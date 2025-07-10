#!/usr/bin/env python3
"""
Fixed NFL Predictor Class
This file contains fixes for the NFLPredictor to match the MLB predictor improvements
"""

import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import scipy.sparse
import bittensor as bt
import json
import time

class NFLPredictorFixed:
    def __init__(
        self,
        model_name="nfl_wager_model",
        preprocessor_path=None,
        team_averages_path=None,
        calibrated_model_path=None,
        id=None,
        db_manager=None,
        miner_stats_handler=None,
        predictions_handler=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store paths but don't load models yet (LAZY LOADING)
        self.model_name = model_name
        
        if preprocessor_path is None:
            preprocessor_path = os.path.join(
                os.path.dirname(__file__), "preprocessor.joblib"
            )
        self.preprocessor_path = preprocessor_path
        
        if team_averages_path is None:
            team_averages_path = os.path.join(
                os.path.dirname(__file__), "team_averages_last_5_games_aug.csv"
            )
        self.team_averages_path = team_averages_path

        if calibrated_model_path is None:
            calibrated_model_path = os.path.join(
                os.path.dirname(__file__), "calibrated_sklearn_model.joblib"
            )
        self.calibrated_model_path = calibrated_model_path

        # Initialize placeholders - models will load when first needed
        self.preprocessor = None
        self.model = None
        self.team_averages = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self._models_loaded = False

        self.db_manager = db_manager
        self.miner_stats_handler = miner_stats_handler
        self.id = id
        
        # NFL model parameters (similar to MLB)
        self.nfl_minimum_wager_amount = 15.0
        self.nfl_maximum_wager_amount = 800
        self.nfl_top_n_games = 8
        self.nfl_kelly_fraction_multiplier = 1.0
        self.nfl_max_bet_percentage = 0.6
        self.nfl_edge_threshold = 0.05
        self.last_param_update = 0
        self.param_refresh_interval = 300
        
        # Initialize model parameters
        if db_manager:
            self.get_model_params(self.db_manager)
        
        print("✅ NFL Predictor initialized (models will load on first prediction)")

    def get_HFmodel(self, model_name):
        """Load the neural network model"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), f"{model_name}.pt")
            config_path = os.path.join(model_dir, "config.json")
            
            # Try different model file formats
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            safetensors_path = os.path.join(model_dir, "model.safetensors")
            
            if not os.path.exists(config_path):
                print(f"NFL config file not found: {config_path}")
                return None
                
            if not os.path.exists(model_path) and not os.path.exists(safetensors_path):
                print(f"NFL Model files not found in: {model_dir}")
                print(f"  Looking for: {model_path} or {safetensors_path}")
                return None
            
            # Load config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Define KellyFractionNet here to avoid import issues
            class KellyFractionNet(nn.Module):
                def __init__(self, input_size):
                    super(KellyFractionNet, self).__init__()
                    self.fc1 = nn.Linear(input_size, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, 32)
                    self.fc4 = nn.Linear(32, 1)
                    self.dropout = nn.Dropout(0.3)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc3(x))
                    x = self.dropout(x)
                    return torch.sigmoid(self.fc4(x)) * 0.5
            
            # Create model
            model = KellyFractionNet(input_size=config['input_size'])
            
            # Load weights - try both formats
            if os.path.exists(model_path):
                print(f"Loading NFL model from: {model_path}")
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            elif os.path.exists(safetensors_path):
                print(f"Loading NFL model from: {safetensors_path}")
                # For safetensors, we need to handle it differently
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(safetensors_path)
                    model.load_state_dict(state_dict)
                except ImportError:
                    print("⚠️ safetensors not available, trying alternative method...")
                    # Fallback: try to load as torch file
                    model.load_state_dict(torch.load(safetensors_path, map_location='cpu'))
            else:
                print("No valid model file found")
                return None
            
            return model
        except Exception as e:
            print(f"Error loading NFL model: {e}")
            return None

    def get_model_params(self, db_manager):
        """Get model parameters from database"""
        try:
            current_time = time.time()
            if current_time - self.last_param_update > self.param_refresh_interval:
                if hasattr(db_manager, 'get_miner_parameters'):
                    params = db_manager.get_miner_parameters(self.id)
                    if params:
                        self.nfl_kelly_fraction_multiplier = params.get('nfl_kelly_fraction_multiplier', 1.0)
                        self.nfl_max_bet_percentage = params.get('nfl_max_bet_percentage', 0.6)
                        self.nfl_edge_threshold = params.get('nfl_edge_threshold', 0.05)
                        self.nfl_minimum_wager_amount = params.get('nfl_minimum_wager_amount', 15.0)
                        self.nfl_maximum_wager_amount = params.get('nfl_maximum_wager_amount', 800)
                        self.nfl_top_n_games = params.get('nfl_top_n_games', 8)
                
                self.last_param_update = current_time
        except Exception as e:
            print(f"Error getting NFL model params: {e}")

    def prepare_raw_data(self, home_teams, away_teams):
        """Prepare raw features for NFL teams"""
        try:
            features = []
            for home_team, away_team in zip(home_teams, away_teams):
                # Get team stats
                home_stats = self.team_averages[self.team_averages['team_name'] == home_team]
                away_stats = self.team_averages[self.team_averages['team_name'] == away_team]
                
                if home_stats.empty or away_stats.empty:
                    print(f"Warning: NFL team stats not found for {home_team} or {away_team}")
                    # Use average stats if team not found
                    home_stats = self.team_averages.mean(numeric_only=True)
                    away_stats = self.team_averages.mean(numeric_only=True)
                else:
                    home_stats = home_stats.iloc[0]
                    away_stats = away_stats.iloc[0]
                
                # Create basic feature vector for NFL
                feature_vector = []
                
                # Add available numeric features
                numeric_cols = self.team_averages.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != 'team_name':
                        feature_vector.append(home_stats.get(col, 0))
                        feature_vector.append(away_stats.get(col, 0))
                
                features.append(feature_vector)
            
            return np.array(features)
        except Exception as e:
            print(f"Error preparing NFL raw data: {e}")
            # Return dummy data with reasonable feature count
            return np.random.random((len(home_teams), 20))

    def preprocess_data(self, home_teams, away_teams):
        """Preprocess data using the trained preprocessor"""
        try:
            raw_data = self.prepare_raw_data(home_teams, away_teams)
            
            # Use preprocessor if available
            if self.preprocessor:
                processed_features = self.preprocessor.transform(raw_data)
                return processed_features
            else:
                # Return raw data if no preprocessor
                return raw_data
        except Exception as e:
            print(f"Error preprocessing NFL data: {e}")
            return np.random.random((len(home_teams), 50))

    def predict_games(self, home_teams, away_teams, odds):
        """Main NFL prediction method"""
        try:
            print(f"🏈 Starting NFL prediction for {len(home_teams)} games...")
            
            # Load models if not already loaded
            if not self._load_models_if_needed():
                print("❌ Failed to load NFL models")
                return []
                
            print("📊 Preparing NFL features...")
            raw_features = self.prepare_raw_data(home_teams, away_teams)
            processed_features = self.preprocess_data(home_teams, away_teams)
            print(f"   Raw features shape: {raw_features.shape}")
            print(f"   Processed features shape: {processed_features.shape}")

            # Get probabilities from calibrated model
            print("🔮 Getting NFL sklearn predictions...")
            sklearn_prediction = self.calibrated_model.predict_proba(raw_features)
            
            if sklearn_prediction.shape[1] == 2:
                sklearn_probs = sklearn_prediction[:, 1]
            else:
                sklearn_probs = sklearn_prediction[:, 0]
            
            print(f"   NFL sklearn probabilities: {sklearn_probs}")

            if scipy.sparse.issparse(processed_features):
                processed_features = processed_features.toarray()

            # Get Kelly fractions from neural network
            print("🧠 Getting NFL neural network predictions...")
            pytorch_features_tensor = torch.tensor(
                processed_features, dtype=torch.float32
            ).to(self.device)

            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                predicted_kelly_fractions = self.model(pytorch_features_tensor)
                if len(predicted_kelly_fractions.shape) > 1:
                    predicted_kelly_fractions = predicted_kelly_fractions.squeeze()

            kelly_fractions = predicted_kelly_fractions.cpu().numpy()
            print(f"   NFL Kelly fractions: {kelly_fractions}")
            
            # Ensure arrays
            kelly_fractions = np.atleast_1d(kelly_fractions)
            sklearn_probs = np.atleast_1d(sklearn_probs)
            
            print("💰 Calculating NFL wagers...")
            wagers = self.recommend_wager_distribution(kelly_fractions, sklearn_probs, odds)
            
            print("📋 Building NFL results...")
            results = []
            for i in range(len(home_teams)):
                # Handle odds format
                if isinstance(odds[i], (list, tuple)) and len(odds[i]) >= 2:
                    home_odds = odds[i][0]
                    away_odds = odds[i][1] if len(odds[i]) == 2 else odds[i][2]
                else:
                    home_odds = 1.85
                    away_odds = 1.95
                
                result = {
                    "Home Team": home_teams[i],
                    "Away Team": away_teams[i],
                    "PredictedOutcome": "Home Win" if sklearn_probs[i] > 0.5 else "Away Win",
                    "ConfidenceScore": float(np.round(sklearn_probs[i], 2)),
                    "KellyFraction": float(np.round(kelly_fractions[i], 4)),
                    "recommendedWager": float(wagers[i]),
                    "HomeOdds": float(home_odds),
                    "AwayOdds": float(away_odds),
                }
                results.append(result)

            print(f"✅ Generated {len(results)} NFL predictions successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Error in NFL predict_games: {e}")
            import traceback
            traceback.print_exc()
            return []

    def recommend_wager_distribution(self, kelly_fractions, sklearn_probs, odds):
        """Calculate recommended wagers for NFL"""
        try:
            # Ensure inputs are numpy arrays
            kelly_fractions = np.atleast_1d(np.array(kelly_fractions))
            sklearn_probs = np.atleast_1d(np.array(sklearn_probs))
            
            # Use mock cash if no miner stats handler
            if self.miner_stats_handler:
                current_miner_cash = self.miner_stats_handler.get_miner_cash()
            else:
                current_miner_cash = 5000.0
            
            max_daily_wager = min(self.nfl_maximum_wager_amount, current_miner_cash)
            min_wager = self.nfl_minimum_wager_amount
            
            kelly_fractions *= self.nfl_kelly_fraction_multiplier
            kelly_fractions = np.clip(kelly_fractions, 0.0, 0.5)

            # Calculate edges
            home_odds_list = []
            for i, game_odds in enumerate(odds):
                if isinstance(game_odds, (list, tuple)) and len(game_odds) >= 1:
                    home_odds_list.append(game_odds[0])
                else:
                    home_odds_list.append(1.85)
            
            home_odds_array = np.array(home_odds_list)
            implied_probs = 1 / home_odds_array
            edges = sklearn_probs - implied_probs
            
            # Filter by edge threshold
            valid_indices = edges >= self.nfl_edge_threshold
            filtered_kelly = kelly_fractions * valid_indices

            # Calculate wagers
            total_kelly = np.sum(filtered_kelly)
            if total_kelly > 0:
                bet_fractions = filtered_kelly / total_kelly
            else:
                bet_fractions = np.ones(len(filtered_kelly)) / len(filtered_kelly)
            
            base_wager = max_daily_wager / len(kelly_fractions) if len(kelly_fractions) > 0 else min_wager
            wagers = bet_fractions * base_wager * (1 + filtered_kelly)
            wagers = np.minimum(wagers, max_daily_wager * self.nfl_max_bet_percentage)
            wagers = np.maximum(wagers, min_wager)
            wagers = np.round(wagers, 2)

            return wagers
            
        except Exception as e:
            print(f"❌ Error calculating NFL wagers: {e}")
            num_games = len(np.atleast_1d(kelly_fractions))
            return np.array([self.nfl_minimum_wager_amount] * num_games)

    def predict_game(self, home_team, away_team, home_odds=1.85, away_odds=1.95):
        """Predict a single NFL game"""
        try:
            if not self._load_models_if_needed():
                return {
                    'error': 'Failed to load NFL models',
                    'home_team': home_team,
                    'away_team': away_team
                }
                
            results = self.predict_games(
                home_teams=[home_team],
                away_teams=[away_team], 
                odds=[[home_odds, away_odds]]
            )
            
            if results and len(results) > 0:
                result = results[0]
                return {
                    'game_id': f"nfl_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}",
                    'home_team': result['Home Team'],
                    'away_team': result['Away Team'],
                    'predicted_outcome': result['PredictedOutcome'],
                    'confidence': result['ConfidenceScore'],
                    'kelly_fraction': result['KellyFraction'],
                    'wager': result['recommendedWager']
                }
            else:
                return {
                    'error': 'No NFL prediction generated',
                    'home_team': home_team,
                    'away_team': away_team
                }
                
        except Exception as e:
            return {
                'error': f'NFL prediction failed: {str(e)}',
                'home_team': home_team,
                'away_team': away_team
            }

    def _load_models_if_needed(self):
        """Load NFL models lazily when first needed"""
        if self._models_loaded:
            return True
            
        try:
            print("🔄 Loading NFL models for first time...")
            
            # Load preprocessor
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                print("✅ NFL preprocessor loaded")
            else:
                print(f"⚠️ NFL preprocessor not found: {self.preprocessor_path}")
                self.preprocessor = None  # Will use raw features
            
            # Load team data
            if os.path.exists(self.team_averages_path):
                self.team_averages = pd.read_csv(self.team_averages_path)
                if 'team' in self.team_averages.columns:
                    self.team_averages = self.team_averages.rename(columns={"team": "team_name"})
                print(f"✅ NFL team data loaded: {len(self.team_averages)} teams")
            else:
                print(f"❌ NFL team data not found: {self.team_averages_path}")
                return False
            
            # Load calibrated model
            if os.path.exists(self.calibrated_model_path):
                self.calibrated_model = joblib.load(self.calibrated_model_path)
                print("✅ NFL calibrated model loaded")
            else:
                print(f"❌ NFL calibrated model not found: {self.calibrated_model_path}")
                return False
            
            # Load neural network model
            self.model = self.get_HFmodel(self.model_name)
            if self.model is None:
                print("❌ Failed to load NFL neural network model")
                return False
            self.model = self.model.to(self.device)
            print("✅ NFL neural network model loaded")
            
            self._models_loaded = True
            print("🎉 All NFL models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading NFL models: {e}")
            import traceback
            traceback.print_exc()
            return False
