import os

print("=== BETTENSOR SYSTEM STATUS ===")

models_dir = "bettensor/miner/models"

# Check predictor files
predictors = [
    "mlb_predictor_fixed.py",
    "nfl_predictor_completely_fixed.py", 
    "soccer_predictor_completely_fixed.py"
]

print("\n1. Predictor Files:")
for p in predictors:
    path = os.path.join(models_dir, p)
    status = "✓ EXISTS" if os.path.exists(path) else "✗ MISSING"
    print(f"   {p}: {status}")

# Check model files  
model_files = [
    "mlb_calibrated_model.joblib",
    "mlb_preprocessor.joblib",
    "calibrated_sklearn_model.joblib", 
    "preprocessor.joblib",
    "label_encoder.pkl"
]

print("\n2. Model Files:")
for m in model_files:
    path = os.path.join(models_dir, m)
    status = "✓ EXISTS" if os.path.exists(path) else "✗ MISSING"
    print(f"   {m}: {status}")

# Check neural net model directories
nn_dirs = ["mlb_wager_model.pt", "nfl_wager_model.pt"]
print("\n3. Neural Network Models:")
for nn in nn_dirs:
    path = os.path.join(models_dir, nn)
    status = "✓ EXISTS" if os.path.exists(path) else "✗ MISSING"
    print(f"   {nn}/: {status}")

print("\n=== STATUS COMPLETE ===")
print("If all files show '✓ EXISTS', the system is ready!")
print("If any show '✗ MISSING', those files need to be restored.")
