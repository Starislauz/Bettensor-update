# CLIENT TESTING GUIDE

## Quick Test Instructions

### 1. Setup (One-time)
```powershell
# Navigate to project folder
cd "c:\Users\ANTHONY\Desktop\Bettensor-main\Bettensor-main"

# Install dependencies
pip install -r requirements.txt
```

### 2. Test All Systems
```powershell
# Run comprehensive test to verify everything works
python final_comprehensive_test.py
```
**Expected Result:** All 3 predictors (MLB, NFL, Soccer) should show "WORKING ✅"

### 3. Test Individual Sports
```powershell
# Test MLB predictions
python -c "from bettensor.miner.models.mlb_predictor_fixed import MLBPredictor; p=MLBPredictor(); print('MLB Ready:', p.predict_game({'home_team': 'Yankees', 'away_team': 'Red Sox'}))"

# Test NFL predictions  
python -c "from bettensor.miner.models.nfl_predictor_completely_fixed import NFLPredictor; p=NFLPredictor(); print('NFL Ready:', p.predict_game({'home_team': 'Chiefs', 'away_team': 'Patriots'}))"

# Test Soccer predictions
python -c "from bettensor.miner.models.soccer_predictor_completely_fixed import SoccerPredictor; p=SoccerPredictor(); print('Soccer Ready:', p.predict_game({'home_team': 'Arsenal', 'away_team': 'Chelsea'}))"
```

### 4. Run Automated System
```powershell
# Run the main betting automation
python automated_betting_system.py
```

## What This Software Does

### 🎯 Purpose
**AI-powered sports betting prediction system** using advanced machine learning models.

### 🏆 Supported Sports
- **MLB** (Baseball) - Season predictions
- **NFL** (Football) - Game predictions  
- **Soccer** - Match predictions

### 🧠 AI Models Used
- **Sklearn Models** (Random Forest, XGBoost)
- **Neural Networks** (PyTorch transformers)
- **Calibrated Predictions** (probability calibration)

### 📊 Key Features
- **Game Outcome Predictions** (Win/Loss)
- **Confidence Scores** (0-100%)
- **Kelly Criterion Wager Calculations** (optimal bet sizing)
- **Real-time Predictions** (automated system)

### 💰 Use Cases
1. **Sports Betting** - Predict game outcomes with confidence scores
2. **Fantasy Sports** - Player/team performance predictions
3. **Sports Analysis** - Statistical modeling and insights
4. **Research** - Machine learning for sports analytics

### ⚡ Quick Success Check
If `final_comprehensive_test.py` shows all ✅, the system is ready for production use!

---
**Contact:** If any test fails, check the error messages and ensure all model files are present.
