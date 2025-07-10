#!/usr/bin/env python3
"""
Fixed MLB Predictor Class
This file contains the complete MLBPredictor class that was missing from model_utils.py
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

class MLBPredictor:
    def __init__(
        self,
        model_name="mlb_wager_model",
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
                os.path.dirname(__file__), "mlb_preprocessor.joblib"
            )
        self.preprocessor_path = preprocessor_path
        
        if team_averages_path is None:
            team_averages_path = os.path.join(
                os.path.dirname(__file__), "mlb_team_stats.csv"
            )
        self.team_averages_path = team_averages_path

        if calibrated_model_path is None:
            calibrated_model_path = os.path.join(
                os.path.dirname(__file__),
                "mlb_calibrated_model.joblib",
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
        self.fuzzy_match_percentage = 80
        self.mlb_minimum_wager_amount = 15.0
        self.mlb_maximum_wager_amount = 800
        self.mlb_top_n_games = 8
        self.last_param_update = 0
        self.param_refresh_interval = 300  # 5 minutes in seconds
        
        # Initialize model parameters
        if db_manager:
            self.get_model_params(self.db_manager)

        # MLB model attributes
        self.mlb_model_on = True
        self.mlb_kelly_fraction_multiplier = 1.0
        self.mlb_max_bet_percentage = 0.6
        self.mlb_edge_threshold = 0.05
        
        print("✅ MLB Predictor initialized (models will load on first prediction)")

    def get_HFmodel(self, model_name):
        """Load the neural network model"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), f"{model_name}.pt")
            config_path = os.path.join(model_dir, "config.json")
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            
            if not os.path.exists(config_path) or not os.path.exists(model_path):
                print(f"Model files not found: {model_dir}")
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
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_model_params(self, db_manager):
        """Get model parameters from database"""
        try:
            current_time = time.time()
            if current_time - self.last_param_update > self.param_refresh_interval:
                if hasattr(db_manager, 'get_miner_parameters'):
                    params = db_manager.get_miner_parameters(self.id)
                    if params:
                        self.mlb_model_on = params.get('mlb_model_on', True)
                        self.mlb_kelly_fraction_multiplier = params.get('mlb_kelly_fraction_multiplier', 1.0)
                        self.mlb_max_bet_percentage = params.get('mlb_max_bet_percentage', 0.6)
                        self.mlb_edge_threshold = params.get('mlb_edge_threshold', 0.05)
                        self.mlb_minimum_wager_amount = params.get('mlb_minimum_wager_amount', 15.0)
                        self.mlb_maximum_wager_amount = params.get('mlb_maximum_wager_amount', 800)
                        self.mlb_top_n_games = params.get('mlb_top_n_games', 8)
                
                self.last_param_update = current_time
        except Exception as e:
            print(f"Error getting model params: {e}")

    def prepare_raw_data(self, home_teams, away_teams):
        """Prepare raw features for the sklearn model (20 features)"""
        try:
            features = []
            for home_team, away_team in zip(home_teams, away_teams):
                # Get team stats
                home_stats = self.team_averages[self.team_averages['team_name'] == home_team]
                away_stats = self.team_averages[self.team_averages['team_name'] == away_team]
                
                if home_stats.empty or away_stats.empty:
                    # Use average stats if team not found
                    print(f"Warning: Team stats not found for {home_team} or {away_team}")
                    home_stats = self.team_averages.mean(numeric_only=True)
                    away_stats = self.team_averages.mean(numeric_only=True)
                else:
                    home_stats = home_stats.iloc[0]
                    away_stats = away_stats.iloc[0]
                
                # Create feature vector with 20 features (as expected by sklearn model)
                feature_vector = [
                    # Home team stats (10 features)
                    home_stats.get('wins', 81),
                    home_stats.get('losses', 81), 
                    home_stats.get('win_percentage', 0.5),
                    home_stats.get('avg_runs_scored', 4.5),
                    home_stats.get('avg_runs_allowed', 4.5),
                    home_stats.get('team_batting_avg', 0.250),
                    home_stats.get('team_era', 4.00),
                    home_stats.get('team_ops', 0.720),
                    home_stats.get('home_wins', 40),
                    home_stats.get('away_wins', 41),
                    
                    # Away team stats (10 features)
                    away_stats.get('wins', 81),
                    away_stats.get('losses', 81),
                    away_stats.get('win_percentage', 0.5),
                    away_stats.get('avg_runs_scored', 4.5),
                    away_stats.get('avg_runs_allowed', 4.5),
                    away_stats.get('team_batting_avg', 0.250),
                    away_stats.get('team_era', 4.00),
                    away_stats.get('team_ops', 0.720),
                    away_stats.get('home_wins', 40),
                    away_stats.get('away_wins', 41),
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
        except Exception as e:
            print(f"Error preparing raw data: {e}")
            return np.array([[0.5] * 20] * len(home_teams))  # 20 features for sklearn

    def prepare_neural_features(self, home_teams, away_teams):
        """Prepare features for the neural network (50 features)"""
        try:
            features = []
            for home_team, away_team in zip(home_teams, away_teams):
                # Get team stats
                home_stats = self.team_averages[self.team_averages['team_name'] == home_team]
                away_stats = self.team_averages[self.team_averages['team_name'] == away_team]
                
                if home_stats.empty or away_stats.empty:
                    # Use average stats if team not found
                    print(f"Warning: Team stats not found for {home_team} or {away_team}")
                    home_stats = self.team_averages.mean(numeric_only=True)
                    away_stats = self.team_averages.mean(numeric_only=True)
                else:
                    home_stats = home_stats.iloc[0]
                    away_stats = away_stats.iloc[0]
                
                # Create feature vector with 50 features (as expected by neural network)
                feature_vector = [
                    # Home team basic stats (10 features)
                    home_stats.get('wins', 81),
                    home_stats.get('losses', 81), 
                    home_stats.get('win_percentage', 0.5),
                    home_stats.get('avg_runs_scored', 4.5),
                    home_stats.get('avg_runs_allowed', 4.5),
                    home_stats.get('team_batting_avg', 0.250),
                    home_stats.get('team_era', 4.00),
                    home_stats.get('team_ops', 0.720),
                    home_stats.get('home_wins', 40),
                    home_stats.get('away_wins', 41),
                    
                    # Home team extended stats (15 features)
                    home_stats.get('team_whip', 1.25),
                    home_stats.get('team_k9', 8.5),
                    home_stats.get('team_bb9', 3.2),
                    home_stats.get('team_hr9', 1.1),
                    home_stats.get('team_avg_rvp', 0.0),
                    home_stats.get('team_slugging', 0.420),
                    home_stats.get('team_obp', 0.330),
                    home_stats.get('team_iso', 0.170),
                    home_stats.get('team_babip', 0.300),
                    home_stats.get('team_lob_pct', 0.72),
                    home_stats.get('team_gb_pct', 0.45),
                    home_stats.get('team_hr_fb', 0.12),
                    home_stats.get('team_wpa', 0.0),
                    home_stats.get('team_clutch', 0.0),
                    home_stats.get('home_field_advantage', 1.05),
                    
                    # Away team basic stats (10 features)
                    away_stats.get('wins', 81),
                    away_stats.get('losses', 81),
                    away_stats.get('win_percentage', 0.5),
                    away_stats.get('avg_runs_scored', 4.5),
                    away_stats.get('avg_runs_allowed', 4.5),
                    away_stats.get('team_batting_avg', 0.250),
                    away_stats.get('team_era', 4.00),
                    away_stats.get('team_ops', 0.720),
                    away_stats.get('home_wins', 40),
                    away_stats.get('away_wins', 41),
                    
                    # Away team extended stats (15 features)
                    away_stats.get('team_whip', 1.25),
                    away_stats.get('team_k9', 8.5),
                    away_stats.get('team_bb9', 3.2),
                    away_stats.get('team_hr9', 1.1),
                    away_stats.get('team_avg_rvp', 0.0),
                    away_stats.get('team_slugging', 0.420),
                    away_stats.get('team_obp', 0.330),
                    away_stats.get('team_iso', 0.170),
                    away_stats.get('team_babip', 0.300),
                    away_stats.get('team_lob_pct', 0.72),
                    away_stats.get('team_gb_pct', 0.45),
                    away_stats.get('team_hr_fb', 0.12),
                    away_stats.get('team_wpa', 0.0),
                    away_stats.get('team_clutch', 0.0),
                    away_stats.get('away_field_disadvantage', 0.95),
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
        except Exception as e:
            print(f"Error preparing neural features: {e}")
            return np.array([[0.5] * 50] * len(home_teams))  # 50 features for neural network

    def preprocess_data(self, home_teams, away_teams):
        """Preprocess data using the trained preprocessor"""
        try:
            raw_data = self.prepare_raw_data(home_teams, away_teams)
            
            # Convert to DataFrame for consistency with 20 features
            feature_names = [
                # Home team features (10)
                'home_wins', 'home_losses', 'home_win_pct', 'home_runs_scored',
                'home_runs_allowed', 'home_batting_avg', 'home_era', 'home_ops',
                'home_home_wins', 'home_away_wins',
                
                # Away team features (10) 
                'away_wins', 'away_losses', 'away_win_pct', 'away_runs_scored', 
                'away_runs_allowed', 'away_batting_avg', 'away_era', 'away_ops',
                'away_home_wins', 'away_away_wins'
            ]
            
            df = pd.DataFrame(raw_data, columns=feature_names)
            
            # Use preprocessor
            processed_features = self.preprocessor.transform(df)
            
            return processed_features
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            # Return dummy processed data
            return np.random.random((len(home_teams), 50))

    def predict_games(self, home_teams, away_teams, odds):
        """Main prediction method"""
        try:
            print(f"🎯 Starting prediction for {len(home_teams)} games...")
            
            # Load models if not already loaded
            if not self._load_models_if_needed():
                print("❌ Failed to load MLB models")
                return []
                
            print("📊 Preparing features...")
            raw_features = self.prepare_raw_data(home_teams, away_teams)  # 20 features for sklearn
            neural_features = self.prepare_neural_features(home_teams, away_teams)  # 50 features for neural net
            processed_features = self.preprocess_data(home_teams, away_teams)  # Use sklearn preprocessor
            print(f"   Raw features shape: {raw_features.shape}")
            print(f"   Neural features shape: {neural_features.shape}")
            print(f"   Processed features shape: {processed_features.shape}")

            # Get probabilities from calibrated model
            print("🔮 Getting sklearn predictions...")
            sklearn_prediction = self.calibrated_model.predict_proba(raw_features)
            print(f"   Sklearn prediction shape: {sklearn_prediction.shape}")
            
            # Handle both binary and multi-class predictions
            if sklearn_prediction.shape[1] == 2:
                sklearn_probs = sklearn_prediction[:, 1]  # Get positive class probability
            else:
                sklearn_probs = sklearn_prediction[:, 0]  # Get first class probability
            
            print(f"   Sklearn probabilities: {sklearn_probs}")

            if scipy.sparse.issparse(processed_features):
                processed_features = processed_features.toarray()

            # Get Kelly fractions from neural network using 50-feature input
            print("🧠 Getting neural network predictions...")
            pytorch_features_tensor = torch.tensor(
                neural_features, dtype=torch.float32  # Use neural_features (50 features)
            ).to(self.device)

            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                predicted_kelly_fractions = self.model(pytorch_features_tensor)
                if len(predicted_kelly_fractions.shape) > 1:
                    predicted_kelly_fractions = predicted_kelly_fractions.squeeze()

            kelly_fractions = predicted_kelly_fractions.cpu().numpy()
            print(f"   Kelly fractions: {kelly_fractions}")
            
            # Ensure Kelly fractions and sklearn_probs are always arrays
            if not isinstance(kelly_fractions, np.ndarray) or kelly_fractions.ndim == 0:
                kelly_fractions = np.array([float(kelly_fractions)])
            elif kelly_fractions.ndim == 1 and len(kelly_fractions) == 1:
                kelly_fractions = kelly_fractions
            else:
                kelly_fractions = kelly_fractions.flatten()
                
            if not isinstance(sklearn_probs, np.ndarray) or sklearn_probs.ndim == 0:
                sklearn_probs = np.array([float(sklearn_probs)])
            elif sklearn_probs.ndim == 1 and len(sklearn_probs) == 1:
                sklearn_probs = sklearn_probs
            else:
                sklearn_probs = sklearn_probs.flatten()
            
            print(f"   Kelly fractions (normalized): {kelly_fractions}")
            print(f"   Sklearn probs (normalized): {sklearn_probs}")
            
            print("💰 Calculating wagers...")
            wagers = self.recommend_wager_distribution(kelly_fractions, sklearn_probs, odds)
            print(f"   Recommended wagers: {wagers}")
            
            print("📋 Building results...")
            results = []
            for i in range(len(home_teams)):
                # Handle odds format - extract home and away odds
                if isinstance(odds[i], (list, tuple)) and len(odds[i]) >= 3:
                    home_odds = odds[i][0]
                    away_odds = odds[i][2]
                elif isinstance(odds[i], (list, tuple)) and len(odds[i]) == 2:
                    home_odds = odds[i][0] 
                    away_odds = odds[i][1]
                else:
                    home_odds = 1.85  # Default
                    away_odds = 1.95  # Default
                
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
                print(f"   Result {i+1}: {result}")

            print(f"✅ Generated {len(results)} predictions successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Error in predict_games: {e}")
            import traceback
            traceback.print_exc()
            return []

    def recommend_wager_distribution(self, kelly_fractions, sklearn_probs, odds):
        """Calculate recommended wagers"""
        try:
            print("💰 Calculating wager distribution...")
            
            # Ensure inputs are numpy arrays
            kelly_fractions = np.atleast_1d(np.array(kelly_fractions))
            sklearn_probs = np.atleast_1d(np.array(sklearn_probs))
            
            print(f"   Kelly fractions shape: {kelly_fractions.shape}")
            print(f"   Sklearn probs shape: {sklearn_probs.shape}")
            
            # Use mock cash if no miner stats handler
            if self.miner_stats_handler:
                current_miner_cash = self.miner_stats_handler.get_miner_cash()
            else:
                current_miner_cash = 5000.0  # Default for testing
            print(f"   Available cash: ${current_miner_cash}")

            max_daily_wager = min(self.mlb_maximum_wager_amount, current_miner_cash)
            min_wager = self.mlb_minimum_wager_amount
            top_n = self.mlb_top_n_games
            
            print(f"   Max daily wager: ${max_daily_wager}")
            print(f"   Min wager: ${min_wager}")
            print(f"   Top N games: {top_n}")
            
            kelly_fractions *= self.mlb_kelly_fraction_multiplier
            kelly_fractions = np.clip(kelly_fractions, 0.0, 0.5)
            print(f"   Adjusted Kelly fractions: {kelly_fractions}")

            # Calculate implied probabilities and edges
            # Handle odds format safely
            home_odds_list = []
            for i, game_odds in enumerate(odds):
                if isinstance(game_odds, (list, tuple)) and len(game_odds) >= 1:
                    home_odds_list.append(game_odds[0])
                else:
                    home_odds_list.append(1.85)  # Default
            
            home_odds_array = np.array(home_odds_list)
            implied_probs = 1 / home_odds_array
            edges = sklearn_probs - implied_probs
            print(f"   Edges: {edges}")
            
            # Dynamic edge threshold for MLB
            median_edge = np.median(edges) if len(edges) > 0 else 0.01
            edge_threshold = max(self.mlb_edge_threshold, median_edge * 0.5)
            print(f"   Edge threshold: {edge_threshold}")

            # Filter by edge threshold
            valid_indices = edges >= edge_threshold
            filtered_kelly = kelly_fractions * valid_indices
            print(f"   Valid indices: {valid_indices}")
            print(f"   Filtered Kelly: {filtered_kelly}")

            # Select top games
            if len(filtered_kelly) > 0:
                top_indices = np.argsort(filtered_kelly)[-top_n:]
                top_kelly_fractions = filtered_kelly[top_indices]
            else:
                top_indices = np.arange(len(kelly_fractions))
                top_kelly_fractions = kelly_fractions
            
            print(f"   Top indices: {top_indices}")
            print(f"   Top Kelly fractions: {top_kelly_fractions}")

            # Calculate wagers
            total_kelly = np.sum(top_kelly_fractions)
            print(f"   Total Kelly: {total_kelly}")
            
            if total_kelly > 0:
                bet_fractions = top_kelly_fractions / total_kelly
            else:
                bet_fractions = np.ones(len(top_kelly_fractions)) / len(top_kelly_fractions)
            
            print(f"   Bet fractions: {bet_fractions}")
            
            base_wager = max_daily_wager / len(top_kelly_fractions) if len(top_kelly_fractions) > 0 else min_wager
            wagers = bet_fractions * base_wager * (1 + top_kelly_fractions)
            wagers = np.minimum(wagers, max_daily_wager * self.mlb_max_bet_percentage)

            # Ensure minimum wager
            wagers = np.maximum(wagers, min_wager)
            wagers = np.round(wagers, 2)
            print(f"   Calculated wagers: {wagers}")

            # Distribute to all games
            final_wagers = np.zeros(len(kelly_fractions))
            if len(top_indices) == len(wagers):
                final_wagers[top_indices] = wagers
            else:
                # Fallback: assign minimum wager to all
                final_wagers = np.full(len(kelly_fractions), min_wager)
            
            print(f"   Final wagers: {final_wagers}")
            return final_wagers
            
        except Exception as e:
            print(f"❌ Error calculating wagers: {e}")
            import traceback
            traceback.print_exc()
            # Return safe default wagers - ensure kelly_fractions is array-like
            num_games = len(np.atleast_1d(kelly_fractions))
            return np.array([self.mlb_minimum_wager_amount] * num_games)

    def predict_game(self, home_team, away_team, home_odds=1.85, away_odds=1.95):
        """
        Predict a single MLB game - wrapper for predict_games method
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team  
            home_odds (float): Odds for home team (default 1.85)
            away_odds (float): Odds for away team (default 1.95)
            
        Returns:
            dict: Prediction results for the single game
        """
        try:
            # Load models if not already loaded
            if not self._load_models_if_needed():
                return {
                    'error': 'Failed to load MLB models',
                    'home_team': home_team,
                    'away_team': away_team
                }
                
            # Use predict_games with single game data
            results = self.predict_games(
                home_teams=[home_team],
                away_teams=[away_team], 
                odds=[[home_odds, 0.0, away_odds]]  # [home, draw, away] format
            )
            
            if results and len(results) > 0:
                result = results[0]
                # Convert to more readable format
                return {
                    'game_id': f"mlb_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}",
                    'home_team': result['Home Team'],
                    'away_team': result['Away Team'],
                    'predicted_outcome': result['PredictedOutcome'],
                    'confidence': result['ConfidenceScore'],
                    'win_probability': result['ConfidenceScore'],
                    'kelly_fraction': result['KellyFraction'],
                    'recommended_stake': result['recommendedWager'],
                    'predicted_odds': result['HomeOdds'] if result['PredictedOutcome'] == 'Home Win' else result['AwayOdds'],
                    'wager': result['recommendedWager']
                }
            else:
                return {
                    'error': 'No prediction generated',
                    'home_team': home_team,
                    'away_team': away_team
                }
                
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'home_team': home_team,
                'away_team': away_team
            }

    def _load_models_if_needed(self):
        """Load models lazily when first needed"""
        if self._models_loaded:
            return True
            
        try:
            print("🔄 Loading MLB models for first time...")
            
            # Load preprocessor
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                print("✅ MLB preprocessor loaded")
            else:
                print(f"❌ MLB preprocessor not found: {self.preprocessor_path}")
                return False
            
            # Load team data
            if os.path.exists(self.team_averages_path):
                self.team_averages = pd.read_csv(self.team_averages_path)
                # Handle both 'team' and 'team_name' column names
                if 'team' in self.team_averages.columns:
                    self.team_averages = self.team_averages.rename(columns={"team": "team_name"})
                print(f"✅ MLB team data loaded: {len(self.team_averages)} teams")
            else:
                print(f"❌ MLB team data not found: {self.team_averages_path}")
                return False
            
            # Load calibrated model
            if os.path.exists(self.calibrated_model_path):
                self.calibrated_model = joblib.load(self.calibrated_model_path)
                print("✅ MLB calibrated model loaded")
            else:
                print(f"❌ MLB calibrated model not found: {self.calibrated_model_path}")
                return False
            
            # Load neural network model
            self.model = self.get_HFmodel(self.model_name)
            if self.model is None:
                print("❌ Failed to load MLB neural network model")
                return False
            self.model = self.model.to(self.device)
            print("✅ MLB neural network model loaded")
            
            self._models_loaded = True
            print("🎉 All MLB models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading MLB models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _ensure_models_loaded(self):
        """Alias for _load_models_if_needed for compatibility"""
        return self._load_models_if_needed()
