#!/usr/bin/env python3
"""
Quick file status check for model files
"""
import os

# Get the correct models directory path
models_dir = os.path.join(os.getcwd(), "bettensor", "miner", "models")
print(f"Checking models directory: {models_dir}")
print(f"Directory exists: {os.path.exists(models_dir)}")
print()

# Check model files in the correct directory
model_files = [
    "mlb_predictor_fixed.py",
    "nfl_predictor_completely_fixed.py", 
    "soccer_predictor_completely_fixed.py",
    "mlb_team_stats.csv",
    "mlb_calibrated_model.joblib",
    "mlb_preprocessor.joblib",
    "calibrated_sklearn_model.joblib",
    "preprocessor.joblib",
    "label_encoder.pkl",
    "team_historical_stats.csv",
    "team_averages_last_5_games_aug.csv"
]

print("Model files status:")
for file in model_files:
    file_path = os.path.join(models_dir, file)
    status = "✅" if os.path.exists(file_path) else "❌"
    print(f"  {status} {file}")

# Additional model directories check
model_dirs = [
    "mlb_wager_model.pt",
    "nfl_wager_model.pt"
]

print("\nModel directories status:")
for dir_name in model_dirs:
    dir_path = os.path.join(models_dir, dir_name)
    status = "✅" if os.path.exists(dir_path) and os.path.isdir(dir_path) else "❌"
    print(f"  {status} {dir_name}/")
    
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # Check contents of model directories
        for item in os.listdir(dir_path):
            print(f"    📁 {item}")

print(f"\n🎯 Model files location: {models_dir}")
