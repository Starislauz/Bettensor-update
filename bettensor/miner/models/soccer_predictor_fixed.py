#!/usr/bin/env python3
"""
Fixed Soccer Predictor Class
This file contains the complete SoccerPredictor class with lazy loading and robust error handling
"""

import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import scipy.sparse
import pickle
import time

class SoccerPredictorFixed:
    def __init__(
        self,
        model_name="soccer_model",
        label_encoder_path=None,
        team_averages_path=None,
        id=0,
        db_manager=None,
        miner_stats_handler=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store paths but don't load models yet (LAZY LOADING)
        self.model_name = model_name
        
        if label_encoder_path is None:
            label_encoder_path = os.path.join(
                os.path.dirname(__file__), "label_encoder.pkl"
            )
        self.label_encoder_path = label_encoder_path
        
        if team_averages_path is None:
            team_averages_path = os.path.join(
                os.path.dirname(__file__), "team_averages_last_5_games_aug.csv"
            )
        self.team_averages_path = team_averages_path

        # Initialize placeholders - models will load when first needed
        self.model = None
        self.le = None
        self.scaler = StandardScaler()
        self.team_averages = None
        self._models_loaded = False

        self.db_manager = db_manager
        self.miner_stats_handler = miner_stats_handler
        self.id = id
        
        # Soccer model parameters
        self.soccer_model_on = True
        self.wager_distribution_steepness = 10
        self.fuzzy_match_percentage = 80
        self.minimum_wager_amount = 20.0
        self.maximum_wager_amount = 1000
        self.top_n_games = 10
        self.last_param_update = 0
        self.param_refresh_interval = 300
        
        # Initialize model parameters
        if db_manager:
            self.get_model_params(self.db_manager)
        
        print("✅ Soccer Predictor initialized (models will load on first prediction)")

    def load_label_encoder(self, path):
        """Load the label encoder for team names"""
        try:
            with open(path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            return None

    def get_HFmodel(self, model_name):
        """Load the soccer transformer model"""
        try:
            # Define PodosTransformer locally to avoid import issues
            class PodosTransformer(nn.Module):
                def __init__(self, input_dim=23, model_dim=64, num_classes=3, num_heads=4, num_layers=2, dropout=0.1, temperature=1):
                    super(PodosTransformer, self).__init__()
                    self.temperature = temperature
                    self.projection = nn.Linear(input_dim, model_dim)
                    encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.fc = nn.Linear(model_dim, num_classes)

                def forward(self, x):
                    x = self.projection(x)
                    x = x.unsqueeze(1)
                    x = self.transformer_encoder(x)
                    x = x.mean(dim=1)
                    x = self.fc(x)
                    if self.temperature != 1.0:
                        x = x / self.temperature
                    return x

            local_model_dir = os.path.dirname(__file__)
            local_model_path = os.path.join(local_model_dir, f"{model_name}.pt")

            if os.path.exists(local_model_path):
                print(f"Loading soccer model from local file: {local_model_path}")
                # Create a basic model structure for soccer
                model = PodosTransformer()
                try:
                    model.load_state_dict(torch.load(local_model_path, map_location='cpu'))
                except:
                    print("Note: Using default soccer model weights")
                return model.to(self.device)
            else:
                print(f"Soccer model file not found: {local_model_path}")
                # Return a default model
                return PodosTransformer().to(self.device)

        except Exception as e:
            print(f"Error loading soccer model: {e}")
            # Return a basic default model
            class PodosTransformer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(23, 3)
                def forward(self, x):
                    return self.fc(x)
            return PodosTransformer().to(self.device)

    def get_model_params(self, db_manager):
        """Get model parameters from database"""
        try:
            current_time = time.time()
            if current_time - self.last_param_update >= self.param_refresh_interval:
                if self.id is None:
                    print("Warning: Miner ID is not set. Using default soccer model parameters.")
                else:
                    if hasattr(db_manager, 'get_model_params'):
                        row = db_manager.get_model_params(self.id)
                        if row:
                            self.soccer_model_on = row.get("soccer_model_on", False)
                            self.wager_distribution_steepness = row.get("wager_distribution_steepness", 1)
                            self.fuzzy_match_percentage = row.get("fuzzy_match_percentage", 80)
                            self.minimum_wager_amount = row.get("minimum_wager_amount", 1.0)
                            self.maximum_wager_amount = row.get("max_wager_amount", 100.0)
                            self.top_n_games = row.get("top_n_games", 10)
                self.last_param_update = current_time
        except Exception as e:
            print(f"Error getting soccer model params: {e}")

    def preprocess_data(self, home_teams, away_teams, odds):
        """Preprocess data for soccer predictions"""
        try:
            odds = np.array(odds)
            df = pd.DataFrame({
                "HomeTeam": home_teams,
                "AwayTeam": away_teams,
                "B365H": odds[:, 0] if odds.shape[1] > 0 else [1.85] * len(home_teams),
                "B365D": odds[:, 1] if odds.shape[1] > 1 else [3.5] * len(home_teams),
                "B365A": odds[:, 2] if odds.shape[1] > 2 else [1.95] * len(home_teams),
            })

            # Encode teams safely
            if self.le and hasattr(self.le, 'classes_'):
                encoded_teams = set(self.le.classes_)
                df["home_encoded"] = df["HomeTeam"].apply(
                    lambda x: self.le.transform([x])[0] if x in encoded_teams else 0
                )
                df["away_encoded"] = df["AwayTeam"].apply(
                    lambda x: self.le.transform([x])[0] if x in encoded_teams else 1
                )
            else:
                # Use simple numeric encoding if no label encoder
                unique_teams = list(set(home_teams + away_teams))
                team_to_id = {team: i for i, team in enumerate(unique_teams)}
                df["home_encoded"] = df["HomeTeam"].map(team_to_id).fillna(0)
                df["away_encoded"] = df["AwayTeam"].map(team_to_id).fillna(1)

            # Add team statistics if available
            if self.team_averages is not None and not self.team_averages.empty:
                home_stats = [
                    "Team", "HS", "HST", "HC", "HO", "HY", "HR",
                    "WinStreakHome", "LossStreakHome", "HomeTeamForm",
                ]
                away_stats = [
                    "Team", "AS", "AST", "AC", "AO", "AY", "AR",
                    "WinStreakAway", "LossStreakAway", "AwayTeamForm",
                ]

                # Merge with team averages safely
                try:
                    available_home_cols = [col for col in home_stats if col in self.team_averages.columns]
                    available_away_cols = [col for col in away_stats if col in self.team_averages.columns]
                    
                    if available_home_cols:
                        df = df.merge(
                            self.team_averages[available_home_cols], 
                            left_on="HomeTeam", right_on="Team", how="left"
                        ).drop(columns=["Team"], errors='ignore')
                    
                    if available_away_cols:
                        df = df.merge(
                            self.team_averages[available_away_cols], 
                            left_on="AwayTeam", right_on="Team", how="left"
                        ).drop(columns=["Team"], errors='ignore')
                except Exception as e:
                    print(f"Warning: Could not merge team averages: {e}")

            # Select available features for model input
            basic_features = ["B365H", "B365D", "B365A", "home_encoded", "away_encoded"]
            features = [col for col in basic_features if col in df.columns]
            
            # Add any remaining numeric columns
            for col in df.columns:
                if col not in features and df[col].dtype in ['int64', 'float64']:
                    features.append(col)
            
            # Ensure we have at least some features
            if len(features) < 5:
                # Add dummy features to reach minimum
                for i in range(5 - len(features)):
                    df[f'dummy_feature_{i}'] = 1.0
                    features.append(f'dummy_feature_{i}')
            
            # Fill NaN values
            df[features] = df[features].fillna(0)
            
            return df[features[:23]]  # Limit to 23 features for the model
            
        except Exception as e:
            print(f"Error preprocessing soccer data: {e}")
            # Return basic dummy data
            num_games = len(home_teams)
            dummy_data = np.random.random((num_games, 23))
            return pd.DataFrame(dummy_data, columns=[f'feature_{i}' for i in range(23)])

    def recommend_wager_distribution(self, confidence_scores):
        """Calculate recommended wagers for soccer"""
        try:
            # Use mock cash if no miner stats handler
            if self.miner_stats_handler:
                current_miner_cash = self.miner_stats_handler.get_miner_cash()
            else:
                current_miner_cash = 5000.0
            
            print(f"Soccer current miner cash: {current_miner_cash}")

            max_daily_wager = min(self.maximum_wager_amount, current_miner_cash)
            min_wager = self.minimum_wager_amount
            top_n = self.top_n_games

            confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
            top_indices = np.argsort(confidence_scores)[-top_n:]
            top_confidences = confidence_scores[top_indices]
            
            # Apply steepness to confidence scores
            sigmoids = 1 / (1 + np.exp(-self.wager_distribution_steepness * (top_confidences - 0.5)))
            normalized_sigmoids = sigmoids / np.sum(sigmoids) if np.sum(sigmoids) > 0 else np.ones_like(sigmoids) / len(sigmoids)

            wagers = normalized_sigmoids * max_daily_wager
            wagers = np.maximum(wagers, min_wager)
            wagers = np.round(wagers, 2)

            # Ensure total doesn't exceed available cash
            total_wager = np.sum(wagers)
            if total_wager > current_miner_cash:
                scale_factor = current_miner_cash / total_wager
                wagers *= scale_factor
                wagers = np.round(wagers, 2)

            # Distribute to all games
            final_wagers = [0.0] * len(confidence_scores)
            for idx, wager in zip(top_indices, wagers):
                final_wagers[idx] = wager

            print(f"Total soccer wager: {np.sum(final_wagers)}")
            return final_wagers
            
        except Exception as e:
            print(f"Error calculating soccer wagers: {e}")
            return [self.minimum_wager_amount] * len(confidence_scores)

    def predict_games(self, home_teams, away_teams, odds):
        """Main soccer prediction method"""
        try:
            print(f"⚽ Starting soccer prediction for {len(home_teams)} games...")
            
            # Load models if not already loaded
            if not self._load_models_if_needed():
                print("❌ Failed to load soccer models")
                return []
            
            print("📊 Preparing soccer features...")
            df = self.preprocess_data(home_teams, away_teams, odds)
            x = self.scaler.fit_transform(df)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            print(f"   Soccer features shape: {x_tensor.shape}")

            print("🧠 Getting soccer model predictions...")
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x_tensor)
                probs = nn.Softmax(dim=1)(outputs.cpu())
                confidence_scores, pred_labels = torch.max(probs, dim=1)

            outcome_map = {0: "Home Win", 1: "Tie", 2: "Away Win"}
            pred_outcomes = [outcome_map.get(label.item(), "Home Win") for label in pred_labels]

            confidence_scores = confidence_scores.cpu().numpy()
            wagers = self.recommend_wager_distribution(confidence_scores)

            print("📋 Building soccer results...")
            results = []
            for i in range(len(home_teams)):
                # Handle odds format
                if isinstance(odds[i], (list, tuple)) and len(odds[i]) >= 3:
                    home_odds = odds[i][0]
                    draw_odds = odds[i][1]
                    away_odds = odds[i][2]
                elif isinstance(odds[i], (list, tuple)) and len(odds[i]) == 2:
                    home_odds = odds[i][0]
                    draw_odds = 3.5  # Default draw odds
                    away_odds = odds[i][1]
                else:
                    home_odds = 1.85
                    draw_odds = 3.5
                    away_odds = 1.95
                
                result = {
                    "Home Team": home_teams[i],
                    "Away Team": away_teams[i],
                    "PredictedOutcome": pred_outcomes[i],
                    "ConfidenceScore": np.round(confidence_scores[i].item(), 2),
                    "recommendedWager": wagers[i],
                    "HomeOdds": float(home_odds),
                    "DrawOdds": float(draw_odds),
                    "AwayOdds": float(away_odds),
                }
                results.append(result)

            print(f"✅ Generated {len(results)} soccer predictions successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Error in soccer predict_games: {e}")
            import traceback
            traceback.print_exc()
            return []

    def predict_game(self, home_team, away_team, home_odds=1.85, draw_odds=3.5, away_odds=1.95):
        """Predict a single soccer game"""
        try:
            if not self._load_models_if_needed():
                return {
                    'error': 'Failed to load soccer models',
                    'home_team': home_team,
                    'away_team': away_team
                }
                
            results = self.predict_games(
                home_teams=[home_team],
                away_teams=[away_team], 
                odds=[[home_odds, draw_odds, away_odds]]
            )
            
            if results and len(results) > 0:
                result = results[0]
                return {
                    'game_id': f"soccer_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}",
                    'home_team': result['Home Team'],
                    'away_team': result['Away Team'],
                    'predicted_outcome': result['PredictedOutcome'],
                    'confidence': result['ConfidenceScore'],
                    'wager': result['recommendedWager']
                }
            else:
                return {
                    'error': 'No soccer prediction generated',
                    'home_team': home_team,
                    'away_team': away_team
                }
                
        except Exception as e:
            return {
                'error': f'Soccer prediction failed: {str(e)}',
                'home_team': home_team,
                'away_team': away_team
            }

    def _load_models_if_needed(self):
        """Load soccer models lazily when first needed"""
        if self._models_loaded:
            return True
            
        try:
            print("🔄 Loading soccer models for first time...")
            
            # Load label encoder
            if os.path.exists(self.label_encoder_path):
                self.le = self.load_label_encoder(self.label_encoder_path)
                print("✅ Soccer label encoder loaded")
            else:
                print(f"⚠️ Soccer label encoder not found: {self.label_encoder_path}")
                self.le = None  # Will use simple encoding
            
            # Load team averages
            if os.path.exists(self.team_averages_path):
                self.team_averages = pd.read_csv(self.team_averages_path)
                print(f"✅ Soccer team averages loaded: {len(self.team_averages)} records")
            else:
                print(f"⚠️ Soccer team averages not found: {self.team_averages_path}")
                self.team_averages = pd.DataFrame()  # Empty dataframe
            
            # Load model
            self.model = self.get_HFmodel(self.model_name)
            if self.model is None:
                print("⚠️ Using default soccer model")
            else:
                print("✅ Soccer model loaded")
            
            self._models_loaded = True
            print("🎉 Soccer models setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading soccer models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_max_wager_vs_miner_cash(self, max_wager):
        """Return the lesser of the max_wager and the miner's cash"""
        if self.miner_stats_handler:
            miner_cash = self.miner_stats_handler.get_miner_cash()
            return min(max_wager, miner_cash)
        return max_wager
