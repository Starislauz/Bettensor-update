from dataclasses import dataclass
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import hf_hub_download
from bettensor.miner.database.database_manager import DatabaseManager
import bittensor as bt
import time
import joblib
import scipy.sparse
import warnings
from pathlib import Path

# Filter out scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


@dataclass
class MinerConfig:
    model_prediction: bool = False


class SoccerPredictor:
    def __init__(
        self,
        model_name,
        label_encoder_path=None,
        team_averages_path=None,
        id=0,
        db_manager=None,
        miner_stats_handler=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.get_HFmodel(model_name)
        if label_encoder_path is None:
            label_encoder_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "label_encoder.pkl"
            )
        self.le = self.load_label_encoder(label_encoder_path)
        self.scaler = StandardScaler()
        if team_averages_path is None:
            team_averages_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "team_averages_last_5_games_aug.csv",
            )
        self.team_averages_path = team_averages_path
        self.db_manager = db_manager
        self.miner_stats_handler = miner_stats_handler

        # params
        self.id: int = id
        self.soccer_model_on: bool = False
        self.wager_distribution_steepness: int = 10
        self.fuzzy_match_percentage: int = 80
        self.minimum_wager_amount: float = 20.0
        self.maximum_wager_amount: float = 1000
        self.top_n_games: int = 10
        self.last_param_update = 0
        self.param_refresh_interval = 300  # 5 minutes in seconds
        self.get_model_params(self.db_manager)

    def get_model_params(self, db_manager: DatabaseManager):
        current_time = time.time()
        if current_time - self.last_param_update >= self.param_refresh_interval:
            if self.id is None:
                bt.logging.warning(
                    "Miner ID is not set. Using default model parameters."
                )
            else:
                row = db_manager.get_model_params(self.id)
                if row is None:
                    bt.logging.info(
                        f"No model parameters found for miner ID: {self.id}. Using default values."
                    )
                    db_manager.initialize_default_model_params(self.id)
                else:
                    self.soccer_model_on = row.get("soccer_model_on", False)
                    self.wager_distribution_steepness = row.get(
                        "wager_distribution_steepness", 1
                    )
                    self.fuzzy_match_percentage = row.get("fuzzy_match_percentage", 80)
                    self.minimum_wager_amount = row.get("minimum_wager_amount", 1.0)
                    self.maximum_wager_amount = row.get("max_wager_amount", 100.0)
                    self.top_n_games = row.get("top_n_games", 10)
            self.last_param_update = current_time

    def check_max_wager_vs_miner_cash(self, max_wager):
        """
        Return the lesser of the max_wager and the miner's cash for model wager distribution.

        """
        miner_cash = self.miner_stats_handler.get_miner_cash()
        return min(max_wager, miner_cash)

    def load_label_encoder(self, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def get_HFmodel(self, model_name):
        try:
            import warnings

            warnings.filterwarnings(
                "ignore",
                message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True",
            )

            local_model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            local_model_path = os.path.join(local_model_dir, f"{model_name}.pt")

            if os.path.exists(local_model_path):
                bt.logging.info(f"Loading model from local file: {local_model_path}")
                model = PodosTransformer.from_pretrained(local_model_path)
            else:
                bt.logging.info(
                    f"Downloading model from Hugging Face Hub: Nickel5HF/{model_name}"
                )
                model = PodosTransformer.from_pretrained(f"NicKel5HF/{model_name}")

                # Save the model locally
                os.makedirs(local_model_dir, exist_ok=True)
                model.save_pretrained(local_model_path)
                bt.logging.info(f"Saved model to local file: {local_model_path}")

            return model.to(self.device)

        except Exception as e:
            bt.logging.error(f"Error loading model: {e}")
            return None

    def preprocess_data(self, home_teams, away_teams, odds):
        odds = np.array(odds)
        df = pd.DataFrame(
            {
                "HomeTeam": home_teams,
                "AwayTeam": away_teams,
                "B365H": odds[:, 0],
                "B365D": odds[:, 1],
                "B365A": odds[:, 2],
            }
        )

        encoded_teams = set(self.le.classes_)
        df["home_encoded"] = df["HomeTeam"].apply(
            lambda x: self.le.transform([x])[0] if x in encoded_teams else None
        )
        df["away_encoded"] = df["AwayTeam"].apply(
            lambda x: self.le.transform([x])[0] if x in encoded_teams else None
        )
        df = df.dropna(subset=["home_encoded", "away_encoded"])

        team_averages_df = pd.read_csv(self.team_averages_path)
        home_stats = [
            "Team",
            "HS",
            "HST",
            "HC",
            "HO",
            "HY",
            "HR",
            "WinStreakHome",
            "LossStreakHome",
            "HomeTeamForm",
        ]
        away_stats = [
            "Team",
            "AS",
            "AST",
            "AC",
            "AO",
            "AY",
            "AR",
            "WinStreakAway",
            "LossStreakAway",
            "AwayTeamForm",
        ]

        df = df.merge(
            team_averages_df[home_stats], left_on="HomeTeam", right_on="Team"
        ).drop(columns=["Team"])
        df = df.merge(
            team_averages_df[away_stats], left_on="AwayTeam", right_on="Team"
        ).drop(columns=["Team"])

        features = [
            "HS",
            "AS",
            "HST",
            "AST",
            "HC",
            "AC",
            "HO",
            "AO",
            "HY",
            "AY",
            "HR",
            "AR",
            "B365H",
            "B365D",
            "B365A",
            "home_encoded",
            "away_encoded",
            "WinStreakHome",
            "LossStreakHome",
            "WinStreakAway",
            "LossStreakAway",
            "HomeTeamForm",
            "AwayTeamForm",
        ]
        return df[features]

    def recommend_wager_distribution(self, confidence_scores):
        current_miner_cash = self.miner_stats_handler.get_miner_cash()
        bt.logging.info(f"Current miner cash: {current_miner_cash}")

        max_daily_wager = min(self.maximum_wager_amount, current_miner_cash)
        min_wager = self.minimum_wager_amount
        top_n = self.top_n_games

        confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
        top_indices = np.argsort(confidence_scores)[-top_n:]
        top_confidences = confidence_scores[top_indices]
        sigmoids = 1 / (1 + np.exp(-10 * (top_confidences - 0.5)))
        normalized_sigmoids = sigmoids / np.sum(sigmoids)

        wagers = normalized_sigmoids * max_daily_wager
        wagers = np.maximum(wagers, min_wager)
        wagers = np.round(wagers, 2)

        total_wager = np.sum(wagers)
        if total_wager > current_miner_cash:
            scale_factor = current_miner_cash / total_wager
            wagers *= scale_factor
            wagers = np.round(wagers, 2)

        excess = np.sum(wagers) - current_miner_cash
        while excess > 0.01:
            wagers_above_min = wagers[wagers > min_wager]
            if len(wagers_above_min) > 0:
                reduction = min(excess / len(wagers_above_min), 0.01)
                wagers[wagers > min_wager] -= reduction
            else:
                max_index = np.argmax(wagers)
                wagers[max_index] = max(wagers[max_index] - 0.01, 0)
            
            wagers = np.round(wagers, 2)
            excess = np.sum(wagers) - current_miner_cash

        final_wagers = [0.0] * len(confidence_scores)
        for idx, wager in zip(top_indices, wagers):
            final_wagers[idx] = wager

        bt.logging.info(f"Total wager: {np.sum(final_wagers)}")
        return final_wagers

    def predict_games(
        self,
        home_teams,
        away_teams,
        odds,
        max_daily_wager=None,
        min_wager=None,
        top_n=None,
    ):
        df = self.preprocess_data(home_teams, away_teams, odds)
        x = self.scaler.fit_transform(df)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            probs = nn.Softmax(dim=1)(outputs.cpu())
            confidence_scores, pred_labels = torch.max(probs, dim=1)

        outcome_map = {0: "Home Win", 1: "Tie", 2: "Away Win"}
        pred_outcomes = [outcome_map[label.item()] for label in pred_labels]

        confidence_scores = confidence_scores.cpu().numpy()
        wagers = self.recommend_wager_distribution(confidence_scores)

        results = []
        for i in range(len(home_teams)):
            result = {
                "Home Team": home_teams[i],
                "Away Team": away_teams[i],
                "PredictedOutcome": pred_outcomes[i],
                "ConfidenceScore": np.round(confidence_scores[i].item(), 2),
                "recommendedWager": wagers[i],
            }
            results.append(result)

        return results


class PodosTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        temperature=1,
    ):
        super(PodosTransformer, self).__init__()
        self.temperature = temperature

        self.projection = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
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


# nfl model
class KellyFractionNet(nn.Module, PyTorchModelHubMixin):
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


class NFLPredictor:
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

        if preprocessor_path is None:
            preprocessor_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "preprocessor.joblib"
            )
        self.preprocessor = joblib.load(preprocessor_path)
        self.scaler = StandardScaler()

        self.model = self.get_HFmodel(model_name)
        if self.model is None:
            raise ValueError("Failed to load the NFL model")
        self.model = self.model.to(self.device)

        if team_averages_path is None:
            team_averages_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "team_historical_stats.csv"
            )
        self.team_averages = pd.read_csv(team_averages_path)
        self.team_averages = self.team_averages.rename(columns={"team": "team_name"})

        if calibrated_model_path is None:
            calibrated_model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "calibrated_sklearn_model.joblib",
            )
        self.calibrated_model = joblib.load(calibrated_model_path)

        self.db_manager = db_manager
        self.miner_stats_handler = miner_stats_handler
        self.id = id
        self.fuzzy_match_percentage = 80
        self.nfl_minimum_wager_amount = 20.0
        self.nfl_maximum_wager_amount = 1000
        self.nfl_top_n_games = 10
        self.last_param_update = 0
        self.param_refresh_interval = 300  # 5 minutes in seconds
        self.get_model_params(self.db_manager)

        # new nfl model attributes
        self.nfl_model_on: bool = False
        self.nfl_kelly_fraction_multiplier = 1.0
        self.nfl_max_bet_percentage = 0.7
        self.nfl_edge_threshold = 0.02
        self.bet365_teams = set(self.team_averages["team_name"])
        self.predictions_handler = predictions_handler

    @property
    def le_classes_(self):
        return self.bet365_teams

    def get_HFmodel(self, model_name):
        try:
            local_model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            local_model_path = os.path.join(local_model_dir, f"{model_name}.pt")

            if os.path.exists(local_model_path):
                bt.logging.info(f"Loading model from local file: {local_model_path}")
                model = KellyFractionNet.from_pretrained(local_model_path)
            else:
                bt.logging.info(
                    f"Downloading model from Hugging Face Hub: Nickel5HF/{model_name}"
                )
                model = KellyFractionNet.from_pretrained(f"Nickel5HF/{model_name}")

                os.makedirs(local_model_dir, exist_ok=True)
                model.save_pretrained(local_model_path)
                bt.logging.info(f"Saved model to local file: {local_model_path}")

            return model.to(self.device)

        except Exception as e:
            bt.logging.error(f"Error loading model: {e}")
            return None

    def get_model_params(self, db_manager: DatabaseManager):
        current_time = time.time()
        bt.logging.debug(f"NFLPredictor: Attempting to get model params. Miner ID: {self.id}")
        if current_time - self.last_param_update >= self.param_refresh_interval:
            if self.id is None:
                bt.logging.warning("NFLPredictor: Miner ID is not set. Using default NFL model parameters.")
            else:
                bt.logging.debug(f"NFLPredictor: Fetching model params for miner ID: {self.id}")
                row = db_manager.get_model_params(self.id)
                bt.logging.debug(f"NFLPredictor: Retrieved row: {row}")
                if row is None:
                    bt.logging.info(f"NFLPredictor: No NFL model parameters found for miner ID: {self.id}. Initializing default values.")
                    db_manager.initialize_default_model_params(self.id)
                    row = db_manager.get_model_params(self.id)
                
                if row:
                    self.nfl_model_on = row.get("nfl_model_on", False)
                    self.nfl_minimum_wager_amount = row.get("nfl_minimum_wager_amount", 20.0)
                    self.nfl_maximum_wager_amount = row.get("nfl_max_wager_amount", 1000)
                    self.nfl_top_n_games = row.get("nfl_top_n_games", 10)
                    self.nfl_kelly_fraction_multiplier = row.get("nfl_kelly_fraction_multiplier", 1.0)
                    self.nfl_edge_threshold = row.get("nfl_edge_threshold", 0.02)
                    self.nfl_max_bet_percentage = row.get("nfl_max_bet_percentage", 0.7)
                    
                else:
                    bt.logging.error(f"NFLPredictor: Failed to retrieve or initialize model parameters for miner ID: {self.id}")
            self.last_param_update = current_time
        else:
            bt.logging.debug(f"NFLPredictor: Using cached model params. Current NFL model status: {self.nfl_model_on}")
        
        return self.nfl_model_on

    def preprocess_data(self, home_teams, away_teams):
        df = pd.DataFrame({
            "team_home": home_teams,
            "team_away": away_teams,
            "schedule_week": [1] * len(home_teams),
        })

        df = df.merge(
            self.team_averages, left_on="team_home", right_on="team_name", how="left"
        )
        df = df.merge(
            self.team_averages,
            left_on="team_away",
            right_on="team_name",
            how="left",
            suffixes=("_home", "_away"),
        )
        df = df.drop(columns=["team_name_home", "team_name_away"])

        df["win_loss_diff"] = (
            df["hist_win_loss_perc_home"] - df["hist_win_loss_perc_away"]
        )
        df["points_for_diff"] = (
            df["hist_avg_points_for_home"] - df["hist_avg_points_for_away"]
        )
        df["points_against_diff"] = (
            df["hist_avg_points_against_home"] - df["hist_avg_points_against_away"]
        )
        df["yards_for_diff"] = (
            df["hist_avg_yards_for_home"] - df["hist_avg_yards_for_away"]
        )
        df["yards_against_diff"] = (
            df["hist_avg_yards_against_home"] - df["hist_avg_yards_against_away"]
        )
        df["turnovers_diff"] = (
            df["hist_avg_turnovers_home"] - df["hist_avg_turnovers_away"]
        )

        categorical_features = ["team_home", "team_away", "schedule_week"]
        historical_features = [
            "hist_wins_home",
            "hist_wins_away",
            "hist_losses_home",
            "hist_losses_away",
            "hist_win_loss_perc_home",
            "hist_win_loss_perc_away",
            "hist_avg_points_for_home",
            "hist_avg_points_for_away",
            "hist_avg_points_against_home",
            "hist_avg_points_against_away",
            "hist_avg_yards_for_home",
            "hist_avg_yards_for_away",
            "hist_avg_yards_against_home",
            "hist_avg_yards_against_away",
            "hist_avg_turnovers_home",
            "hist_avg_turnovers_away",
        ]
        interaction_features = [
            "win_loss_diff",
            "points_for_diff",
            "points_against_diff",
            "yards_for_diff",
            "yards_against_diff",
            "turnovers_diff",
        ]
        numerical_features = historical_features + interaction_features
        features = categorical_features + numerical_features

        processed_features = self.preprocessor.transform(df[features])

        assert (
            processed_features.shape[1] == 118
        ), f"Expected 118 features, but got {processed_features.shape[1]}"

        return processed_features

    def prepare_raw_data(self, home_teams, away_teams):
        df = pd.DataFrame({
            "team_home": home_teams,
            "team_away": away_teams,
            "schedule_week": [1] * len(home_teams),  # Add schedule_week column may change
        })

        df = df.merge(
            self.team_averages, left_on="team_home", right_on="team_name", how="left"
        )
        df = df.merge(
            self.team_averages,
            left_on="team_away",
            right_on="team_name",
            how="left",
            suffixes=("_home", "_away"),
        )
        df = df.drop(columns=["team_name_home", "team_name_away"])

        df["win_loss_diff"] = (
            df["hist_win_loss_perc_home"] - df["hist_win_loss_perc_away"]
        )
        df["points_for_diff"] = (
            df["hist_avg_points_for_home"] - df["hist_avg_points_for_away"]
        )
        df["points_against_diff"] = (
            df["hist_avg_points_against_home"] - df["hist_avg_points_against_away"]
        )
        df["yards_for_diff"] = (
            df["hist_avg_yards_for_home"] - df["hist_avg_yards_for_away"]
        )
        df["yards_against_diff"] = (
            df["hist_avg_yards_against_home"] - df["hist_avg_yards_against_away"]
        )
        df["turnovers_diff"] = (
            df["hist_avg_turnovers_home"] - df["hist_avg_turnovers_away"]
        )

        categorical_features = ["team_home", "team_away", "schedule_week"]
        historical_features = [
            "hist_wins_home",
            "hist_wins_away",
            "hist_losses_home",
            "hist_losses_away",
            "hist_win_loss_perc_home",
            "hist_win_loss_perc_away",
            "hist_avg_points_for_home",
            "hist_avg_points_for_away",
            "hist_avg_points_against_home",
            "hist_avg_points_against_away",
            "hist_avg_yards_for_home",
            "hist_avg_yards_for_away",
            "hist_avg_yards_against_home",
            "hist_avg_yards_against_away",
            "hist_avg_turnovers_home",
            "hist_avg_turnovers_away",
        ]
        interaction_features = [
            "win_loss_diff",
            "points_for_diff",
            "points_against_diff",
            "yards_for_diff",
            "yards_against_diff",
            "turnovers_diff",
        ]
        numerical_features = historical_features + interaction_features
        features = categorical_features + numerical_features

        return df[features]

    def predict_games(self, home_teams, away_teams, odds):
        raw_features = self.prepare_raw_data(home_teams, away_teams)
        processed_features = self.preprocess_data(home_teams, away_teams)

        sklearn_probs = self.calibrated_model.predict_proba(raw_features)[:, 1]

        if scipy.sparse.issparse(processed_features):
            processed_features = processed_features.toarray()

        pytorch_features_tensor = torch.tensor(
            processed_features, dtype=torch.float32
        ).to(self.device)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            predicted_kelly_fractions = self.model(pytorch_features_tensor).squeeze()

        kelly_fractions = predicted_kelly_fractions.cpu().numpy()
        wagers = self.recommend_wager_distribution(kelly_fractions, sklearn_probs, odds)
        bt.logging.info(f"pytorch model output: {kelly_fractions}")
        results = []
        for i in range(len(home_teams)):
            result = {
                "Home Team": home_teams[i],
                "Away Team": away_teams[i],
                "PredictedOutcome": "Home Win"
                if sklearn_probs[i] > 0.5
                else "Away Win",
                "ConfidenceScore": float(np.round(sklearn_probs[i], 2)),
                "KellyFraction": float(np.round(kelly_fractions[i], 4)),
                "recommendedWager": float(wagers[i]),
                "HomeOdds": float(odds[i][0]),
                "AwayOdds": float(odds[i][2]),
            }
            results.append(result)

        return results

    def recommend_wager_distribution(self, kelly_fractions, sklearn_probs, odds):
        current_miner_cash = self.miner_stats_handler.get_miner_cash()
        bt.logging.info(f"Initial current miner cash: {current_miner_cash}")
        
        double_checked_cash = self.miner_stats_handler.get_miner_cash()
        if current_miner_cash != double_checked_cash:
            bt.logging.warning(f"Miner cash discrepancy detected. Initial: {current_miner_cash}, Double-checked: {double_checked_cash}")
            current_miner_cash = double_checked_cash

        max_daily_wager = min(self.nfl_maximum_wager_amount, current_miner_cash)
        min_wager = self.nfl_minimum_wager_amount
        top_n = self.nfl_top_n_games
        
        kelly_fractions *= self.nfl_kelly_fraction_multiplier
        kelly_fractions = np.clip(kelly_fractions, 0.0, 0.5)

        implied_probs = 1 / np.array(odds)[:, 0]  # Using home odds
        edges = sklearn_probs - implied_probs
        
        # Dynamic edge threshold
        median_edge = np.median(edges)
        edge_threshold = max(self.nfl_edge_threshold, median_edge * 0.5)
        
        valid_bets = edges > edge_threshold
        top_indices = np.argsort(edges)[-top_n:]
        top_indices = top_indices[valid_bets[top_indices]]

        if len(top_indices) == 0:
            return [0.0] * len(kelly_fractions)

        top_kelly_fractions = kelly_fractions[top_indices]
        top_odds = np.array(odds)[top_indices, 0]  # Using home odds
        top_sklearn_probs = sklearn_probs[top_indices]

        bet_fractions = calculate_kelly_fraction(
            top_sklearn_probs, top_odds, fraction=0.25
        )
        
        wagers = bet_fractions * max_daily_wager * (1 + top_kelly_fractions)

        wagers = np.minimum(wagers, max_daily_wager * self.nfl_max_bet_percentage)

        total_wager = np.sum(wagers)
        max_total_wager = current_miner_cash * self.nfl_max_bet_percentage
        if total_wager > max_total_wager:
            scale_factor = max_total_wager / total_wager
            wagers *= scale_factor

        wagers = np.maximum(wagers, min_wager)
        wagers = np.round(wagers, 2)

        excess = np.sum(wagers) - current_miner_cash
        while excess > 0.01:
            wagers_above_min = wagers[wagers > min_wager]
            if len(wagers_above_min) > 0:
                reduction = min(excess / len(wagers_above_min), 0.01)
                wagers[wagers > min_wager] -= reduction
            else:
                max_index = np.argmax(wagers)
                wagers[max_index] = max(wagers[max_index] - 0.01, 0)
            
            wagers = np.round(wagers, 2)
            excess = np.sum(wagers) - current_miner_cash

        final_wagers = np.zeros(len(kelly_fractions))
        final_wagers[top_indices] = wagers

        bt.logging.info(f"Total wager: {np.sum(final_wagers)}")
        return final_wagers

    def check_max_wager_vs_miner_cash(self, max_wager):
        miner_cash = self.miner_stats_handler.get_miner_cash()
        return min(max_wager, miner_cash)


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
        
        # Store paths for lazy loading
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
        
        # Initialize models as None - load lazily when needed
        self.preprocessor = None
        self.scaler = StandardScaler()
        self.model = None
        self.team_averages = None
        self.calibrated_model = None
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

    def get_HFmodel(self, model_name):
        """Load the neural network model"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), f"{model_name}.pt")
            config_path = os.path.join(model_dir, "config.json")
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            
            if not os.path.exists(config_path) or not os.path.exists(model_path):
                bt.logging.info(f"Model files not found: {model_dir}")
                return None
            
            # Load config
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create model
            model = KellyFractionNet(input_size=config['input_size'])
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            return model
        except Exception as e:
            bt.logging.error(f"Error loading model: {e}")
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
            bt.logging.error(f"Error getting model params: {e}")

    def prepare_raw_data(self, home_teams, away_teams):
        """Prepare raw features for the sklearn model"""
        try:
            features = []
            for home_team, away_team in zip(home_teams, away_teams):
                # Get team stats
                home_stats = self.team_averages[self.team_averages['team_name'] == home_team]
                away_stats = self.team_averages[self.team_averages['team_name'] == away_team]
                
                if home_stats.empty or away_stats.empty:
                    # Use average stats if team not found
                    bt.logging.warning(f"Team stats not found for {home_team} or {away_team}")
                    home_stats = self.team_averages.mean(numeric_only=True)
                    away_stats = self.team_averages.mean(numeric_only=True)
                else:
                    home_stats = home_stats.iloc[0]
                    away_stats = away_stats.iloc[0]
                
                # Create feature vector
                feature_vector = [
                    home_stats.get('wins', 81),
                    home_stats.get('losses', 81), 
                    home_stats.get('win_percentage', 0.5),
                    home_stats.get('avg_runs_scored', 4.5),
                    home_stats.get('avg_runs_allowed', 4.5),
                    home_stats.get('team_batting_avg', 0.250),
                    home_stats.get('team_era', 4.00),
                    home_stats.get('team_ops', 0.720),
                    away_stats.get('wins', 81),
                    away_stats.get('losses', 81),
                    away_stats.get('win_percentage', 0.5),
                    away_stats.get('avg_runs_scored', 4.5),
                    away_stats.get('avg_runs_allowed', 4.5),
                    away_stats.get('team_batting_avg', 0.250),
                    away_stats.get('team_era', 4.00),
                    away_stats.get('team_ops', 0.720),
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
        except Exception as e:
            bt.logging.error(f"Error preparing raw data: {e}")
            return np.array([[0.5] * 16] * len(home_teams))

    def preprocess_data(self, home_teams, away_teams):
        """Preprocess data using the trained preprocessor"""
        try:
            raw_data = self.prepare_raw_data(home_teams, away_teams)
            
            # Convert to DataFrame for consistency
            feature_names = [
                'home_wins', 'home_losses', 'home_win_pct', 'home_runs_scored',
                'home_runs_allowed', 'home_batting_avg', 'home_era', 'home_ops',
                'away_wins', 'away_losses', 'away_win_pct', 'away_runs_scored', 
                'away_runs_allowed', 'away_batting_avg', 'away_era', 'away_ops'
            ]
            
            df = pd.DataFrame(raw_data, columns=feature_names)
            
            # Use preprocessor
            processed_features = self.preprocessor.transform(df)
            
            return processed_features
        except Exception as e:
            bt.logging.error(f"Error preprocessing data: {e}")
            # Return dummy processed data
            return np.random.random((len(home_teams), 50))

    def predict_games(self, home_teams, away_teams, odds):
        """Main prediction method"""
        try:
            # Load models if not already loaded
            if not self._load_models_if_needed():
                return []
                
            raw_features = self.prepare_raw_data(home_teams, away_teams)
            processed_features = self.preprocess_data(home_teams, away_teams)

            # Get probabilities from calibrated model
            sklearn_probs = self.calibrated_model.predict_proba(raw_features)[:, 1]

            if scipy.sparse.issparse(processed_features):
                processed_features = processed_features.toarray()

            # Get Kelly fractions from neural network
            pytorch_features_tensor = torch.tensor(
                processed_features, dtype=torch.float32
            ).to(self.device)

            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                predicted_kelly_fractions = self.model(pytorch_features_tensor).squeeze()

            kelly_fractions = predicted_kelly_fractions.cpu().numpy()
            
            # Handle single prediction case
            if len(home_teams) == 1:
                kelly_fractions = [kelly_fractions] if not isinstance(kelly_fractions, (list, np.ndarray)) else kelly_fractions
                sklearn_probs = [sklearn_probs] if not isinstance(sklearn_probs, (list, np.ndarray)) else sklearn_probs
            
            wagers = self.recommend_wager_distribution(kelly_fractions, sklearn_probs, odds)
            bt.logging.info(f"MLB pytorch model output: {kelly_fractions}")
            
            results = []
            for i in range(len(home_teams)):
                result = {
                    "Home Team": home_teams[i],
                    "Away Team": away_teams[i],
                    "PredictedOutcome": "Home Win" if sklearn_probs[i] > 0.5 else "Away Win",
                    "ConfidenceScore": float(np.round(sklearn_probs[i], 2)),
                    "KellyFraction": float(np.round(kelly_fractions[i], 4)),
                    "recommendedWager": float(wagers[i]),
                    "HomeOdds": float(odds[i][0]),
                    "AwayOdds": float(odds[i][2]),
                }
                results.append(result)

            return results
        except Exception as e:
            bt.logging.error(f"Error in predict_games: {e}")
            import traceback
            traceback.print_exc()
            return []

    def recommend_wager_distribution(self, kelly_fractions, sklearn_probs, odds):
        """Calculate recommended wagers"""
        try:
            # Use mock cash if no miner stats handler
            if self.miner_stats_handler:
                current_miner_cash = self.miner_stats_handler.get_miner_cash()
            else:
                current_miner_cash = 5000.0  # Default for testing

            bt.logging.info(f"MLB Initial current miner cash: {current_miner_cash}")
            
            max_daily_wager = min(self.mlb_maximum_wager_amount, current_miner_cash)
            min_wager = self.mlb_minimum_wager_amount
            top_n = self.mlb_top_n_games
            
            kelly_fractions = np.array(kelly_fractions)
            kelly_fractions *= self.mlb_kelly_fraction_multiplier
            kelly_fractions = np.clip(kelly_fractions, 0.0, 0.5)

            # Calculate implied probabilities and edges
            implied_probs = 1 / np.array(odds)[:, 0]  # Using home odds
            edges = np.array(sklearn_probs) - implied_probs
            
            # Dynamic edge threshold for MLB
            median_edge = np.median(edges)
            edge_threshold = max(self.mlb_edge_threshold, median_edge * 0.5)

            # Filter by edge threshold
            valid_indices = edges >= edge_threshold
            filtered_kelly = kelly_fractions * valid_indices

            # Select top games
            top_indices = np.argsort(filtered_kelly)[-top_n:]
            top_kelly_fractions = filtered_kelly[top_indices]

            # Calculate wagers
            total_kelly = np.sum(top_kelly_fractions)
            if total_kelly > 0:
                bet_fractions = top_kelly_fractions / total_kelly
            else:
                bet_fractions = np.ones(len(top_kelly_fractions)) / len(top_kelly_fractions)
            
            wagers = bet_fractions * max_daily_wager * (1 + top_kelly_fractions)
            wagers = np.minimum(wagers, max_daily_wager * self.mlb_max_bet_percentage)

            # Scale total wager if needed
            total_wager = np.sum(wagers)
            max_total_wager = current_miner_cash * self.mlb_max_bet_percentage
            if total_wager > max_total_wager:
                scale_factor = max_total_wager / total_wager
                wagers *= scale_factor

            # Ensure minimum wager
            wagers = np.maximum(wagers, min_wager)
            wagers = np.round(wagers, 2)

            # Distribute to all games
            final_wagers = np.zeros(len(kelly_fractions))
            final_wagers[top_indices] = wagers

            bt.logging.info(f"Total MLB wager: {np.sum(final_wagers)}")
            return final_wagers
        except Exception as e:
            bt.logging.error(f"Error calculating wagers: {e}")
            return np.array([self.mlb_minimum_wager_amount] * len(kelly_fractions))

    def check_max_wager_vs_miner_cash(self, max_wager):
        if self.miner_stats_handler:
            miner_cash = self.miner_stats_handler.get_miner_cash()
            return min(max_wager, miner_cash)
        return max_wager

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
            bt.logging.info("Loading MLB models for first time...")
            
            # Load preprocessor
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                bt.logging.info("✅ MLB preprocessor loaded")
            else:
                bt.logging.warning(f"MLB preprocessor not found: {self.preprocessor_path}")
                return False
            
            # Load team data
            if os.path.exists(self.team_averages_path):
                self.team_averages = pd.read_csv(self.team_averages_path)
                # Handle both 'team' and 'team_name' column names
                if 'team' in self.team_averages.columns:
                    self.team_averages = self.team_averages.rename(columns={"team": "team_name"})
                bt.logging.info(f"✅ MLB team data loaded: {len(self.team_averages)} teams")
            else:
                bt.logging.warning(f"MLB team data not found: {self.team_averages_path}")
                return False
            
            # Load calibrated model
            if os.path.exists(self.calibrated_model_path):
                self.calibrated_model = joblib.load(self.calibrated_model_path)
                bt.logging.info("✅ MLB calibrated model loaded")
            else:
                bt.logging.warning(f"MLB calibrated model not found: {self.calibrated_model_path}")
                return False
            
            # Load neural network model
            self.model = self.get_HFmodel(self.model_name)
            if self.model is None:
                bt.logging.error("Failed to load MLB neural network model")
                return False
            self.model = self.model.to(self.device)
            bt.logging.info("✅ MLB neural network model loaded")
            
            self._models_loaded = True
            bt.logging.info("🎉 All MLB models loaded successfully!")
            return True
            
        except Exception as e:
            bt.logging.error(f"Error loading MLB models: {e}")
            return False