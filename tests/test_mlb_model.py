import unittest
from unittest.mock import MagicMock, patch
import bittensor
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bettensor.miner.database.predictions import PredictionsHandler
from bettensor.miner.models.model_utils import MLBPredictor, KellyFractionNet
from bettensor.protocol import TeamGame, TeamGamePrediction
from datetime import datetime, timezone
import joblib
import torch
import scipy.sparse


class TestMLBPredictions(unittest.TestCase):
    def setUp(self):
        self.db_manager = MagicMock()
        self.state_manager = MagicMock()
        self.state_manager.miner_uid = "test_miner_uid"
        self.miner_stats_handler = MagicMock()
        self.miner_stats_handler.stats = {
            "miner_cash": 1000.0,
            "miner_lifetime_wins": 0,
            "miner_lifetime_losses": 0,
            "miner_win_loss_ratio": 0.0,
        }
        self.predictions_handler = PredictionsHandler(
            self.db_manager, self.state_manager, "test_hotkey"
        )
        self.predictions_handler.stats_handler = self.miner_stats_handler

    @patch("bettensor.miner.models.model_utils.MLBPredictor.get_HFmodel")
    @patch("bettensor.miner.models.model_utils.joblib.load")
    def test_process_model_predictions_mlb(self, mock_joblib_load, mock_get_HFmodel):
        mock_db_manager = MagicMock()
        mock_db_manager.get_model_params.return_value = {
            "mlb_model_on": True,
            "soccer_model_on": True,
            "nfl_model_on": True,
            "mlb_minimum_wager_amount": 15.0,
            "mlb_max_wager_amount": 800,
            "fuzzy_match_percentage": 80,
            "mlb_top_n_games": 8,
            "mlb_kelly_fraction_multiplier": 1.0,
            "mlb_edge_threshold": 0.025,
            "mlb_max_bet_percentage": 0.6,
            "wager_distribution_steepness": 1.0,
            "minimum_wager_amount": 10.0,
            "max_wager_amount": 500,
            "top_n_games": 5,
            "kelly_fraction_multiplier": 0.5,
            "edge_threshold": 0.01,
            "max_bet_percentage": 0.5,
        }

        # Mock the required model files
        mock_preprocessor = MagicMock()
        mock_calibrated_model = MagicMock()
        mock_calibrated_model.predict_proba.return_value = [[0.3, 0.7], [0.6, 0.4], [0.4, 0.6]]
        
        mock_joblib_load.side_effect = [mock_preprocessor, mock_calibrated_model]
        
        # Mock the neural network model
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        mock_get_HFmodel.return_value = mock_model

        mock_miner_stats_handler = MagicMock()
        mock_miner_stats_handler.get_miner_cash.return_value = 1000
        mock_miner_stats_handler.stats = {
            "miner_lifetime_wins": 10,
            "miner_lifetime_losses": 5,
            "miner_win_loss_ratio": 2.0,
            "miner_cash": 1000.0,
        }

        predictions_handler = PredictionsHandler(
            mock_db_manager, self.state_manager, "test_hotkey"
        )
        predictions_handler.stats_handler = mock_miner_stats_handler

        def real_get_best_match(team_name, encoded_teams, sport):
            return team_name

        predictions_handler.get_best_match = real_get_best_match

        # Create MLB predictor with mocked dependencies
        with patch('pandas.read_csv') as mock_read_csv:
            mock_team_data = MagicMock()
            mock_team_data.rename.return_value = mock_team_data
            mock_read_csv.return_value = mock_team_data
            
            mlb_predictor = MLBPredictor(
                db_manager=mock_db_manager,
                miner_stats_handler=mock_miner_stats_handler,
                predictions_handler=predictions_handler,
            )
            
            # Set up mocked attributes
            mlb_predictor.preprocessor = mock_preprocessor
            mlb_predictor.calibrated_model = mock_calibrated_model
            mlb_predictor.model = mock_model
            mlb_predictor.mlb_teams = {
                "New York Yankees", "Boston Red Sox", "Los Angeles Dodgers",
                "Houston Astros", "Atlanta Braves"
            }

        predictions_handler.models["baseball"] = mlb_predictor

        current_time = datetime.now(timezone.utc).isoformat()
        games = {
            "game1": TeamGame(
                game_id="game1",
                team_a="New York Yankees",
                team_b="Boston Red Sox",
                team_a_odds=2.1,
                team_b_odds=1.8,
                tie_odds=1.0,  # Baseball rarely ties
                sport="baseball",
                league="MLB",
                external_id="ext1",
                create_date=current_time,
                last_update_date=current_time,
                event_start_date=current_time,
                active=True,
                outcome="Unfinished",
                can_tie=False,
                schedule_week=1,
            ),
            "game2": TeamGame(
                game_id="game2",
                team_a="Los Angeles Dodgers",
                team_b="Houston Astros",
                team_a_odds=1.9,
                team_b_odds=1.95,
                tie_odds=1.0,
                sport="baseball",
                league="MLB",
                external_id="ext2",
                create_date=current_time,
                last_update_date=current_time,
                event_start_date=current_time,
                active=True,
                outcome="Unfinished",
                can_tie=False,
                schedule_week=1,
            ),
            "game3": TeamGame(
                game_id="game3",
                team_a="Atlanta Braves",
                team_b="New York Yankees",
                team_a_odds=2.2,
                team_b_odds=1.7,
                tie_odds=1.0,
                sport="baseball",
                league="MLB",
                external_id="ext3",
                create_date=current_time,
                last_update_date=current_time,
                event_start_date=current_time,
                active=True,
                outcome="Unfinished",
                can_tie=False,
                schedule_week=1,
            ),
        }

        # Mock the preprocessor and model outputs
        mock_preprocessor.transform.return_value = [[0.1, 0.2, 0.3]] * 3
        
        with torch.no_grad():
            mock_tensor_output = torch.tensor([0.15, 0.25, 0.20])
            mock_model.return_value = mock_tensor_output

        predictions = predictions_handler.process_model_predictions(games, "baseball")
        
        print("Processed MLB predictions:")
        for game_id, prediction in predictions.items():
            print(f"Game {game_id}:")
            print(f"  Predicted Outcome: {prediction.predicted_outcome}")
            print(f"  Wager: {prediction.wager}")
            print(f"  Team A: {prediction.team_a}")
            print(f"  Team B: {prediction.team_b}")
            print(f"  Team A Odds: {prediction.team_a_odds}")
            print(f"  Team B Odds: {prediction.team_b_odds}")
            print()

        self.assertEqual(len(predictions), 3)
        for game_id, prediction in predictions.items():
            self.assertIsInstance(prediction, TeamGamePrediction)
            self.assertIn(
                prediction.predicted_outcome, [prediction.team_a, prediction.team_b]
            )
            self.assertGreaterEqual(prediction.wager, 0)
            self.assertLess(prediction.wager, 800)

    def test_get_best_match_mlb(self):
        self.predictions_handler.models["baseball"] = MagicMock()
        self.predictions_handler.models["baseball"].fuzzy_match_percentage = 90
        encoded_teams = {
            "New York Yankees",
            "Boston Red Sox",
            "Los Angeles Dodgers",
            "Houston Astros",
        }

        match = self.predictions_handler.get_best_match(
            "New York Yankees", encoded_teams, "baseball"
        )
        self.assertEqual(match, "New York Yankees")

        match = self.predictions_handler.get_best_match(
            "Yankees", encoded_teams, "baseball"
        )
        self.assertEqual(match, "New York Yankees")

        match = self.predictions_handler.get_best_match(
            "Los Angeles Lakers", encoded_teams, "baseball"
        )
        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
