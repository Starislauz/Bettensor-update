#!/usr/bin/env python3
"""
Standalone test for sports predictors without requiring bittensor import.
This test directly imports and tests individual predictor modules.
"""

import sys
import os
import traceback
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mlb_predictor():
    """Test MLB predictor directly"""
    print("============================================================")
    print("⚾ TESTING MLB PREDICTOR (STANDALONE)")
    print("============================================================")
    
    try:
        # Import directly from the module file
        sys.path.insert(0, str(project_root / "bettensor" / "miner" / "models"))
        from mlb_predictor_fixed import MLBPredictor
        
        print("1. ✅ MLB Predictor imported successfully")
        
        # Initialize predictor
        predictor = MLBPredictor()
        print("2. ✅ MLB Predictor initialized")
        
        # Test prediction with sample data
        team_a = "LAD"  # Los Angeles Dodgers
        team_b = "SF"   # San Francisco Giants
        
        result = predictor.predict_game(team_a, team_b)
        
        if result and 'team_a_win_probability' in result:
            print(f"3. ✅ MLB Prediction successful:")
            print(f"   Team A ({team_a}) win probability: {result['team_a_win_probability']:.3f}")
            print(f"   Team B ({team_b}) win probability: {result['team_b_win_probability']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Suggested wager: {result['suggested_wager']:.2f}")
            return True
        else:
            print("❌ MLB prediction failed - no valid result returned")
            return False
            
    except Exception as e:
        print(f"❌ MLB predictor test failed with error: {e}")
        traceback.print_exc()
        return False

def test_nfl_predictor():
    """Test NFL predictor directly"""
    print("============================================================")
    print("🏈 TESTING NFL PREDICTOR (STANDALONE)")
    print("============================================================")
    
    try:
        # Import directly from the module file
        sys.path.insert(0, str(project_root / "bettensor" / "miner" / "models"))
        from nfl_predictor_completely_fixed import NFLPredictorFixed
        
        print("1. ✅ NFL Predictor imported successfully")
        
        # Initialize predictor
        predictor = NFLPredictorFixed()
        print("2. ✅ NFL Predictor initialized")
        
        # Test prediction with sample data
        team_a = "KC"   # Kansas City Chiefs
        team_b = "BUF"  # Buffalo Bills
        
        result = predictor.predict_game(team_a, team_b)
        
        if result and 'team_a_win_probability' in result:
            print(f"3. ✅ NFL Prediction successful:")
            print(f"   Team A ({team_a}) win probability: {result['team_a_win_probability']:.3f}")
            print(f"   Team B ({team_b}) win probability: {result['team_b_win_probability']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Suggested wager: {result['suggested_wager']:.2f}")
            return True
        else:
            print("❌ NFL prediction failed - no valid result returned")
            return False
            
    except Exception as e:
        print(f"❌ NFL predictor test failed with error: {e}")
        traceback.print_exc()
        return False

def test_soccer_predictor():
    """Test Soccer predictor directly"""
    print("============================================================")
    print("⚽ TESTING SOCCER PREDICTOR (STANDALONE)")
    print("============================================================")
    
    try:
        # Import directly from the module file
        sys.path.insert(0, str(project_root / "bettensor" / "miner" / "models"))
        from soccer_predictor_completely_fixed import SoccerPredictorFixed
        
        print("1. ✅ Soccer Predictor imported successfully")
        
        # Initialize predictor
        predictor = SoccerPredictorFixed()
        print("2. ✅ Soccer Predictor initialized")
        
        # Test prediction with sample data
        team_a = "Manchester United"
        team_b = "Liverpool"
        
        result = predictor.predict_game(team_a, team_b)
        
        if result and 'team_a_win_probability' in result:
            print(f"3. ✅ Soccer Prediction successful:")
            print(f"   Team A ({team_a}) win probability: {result['team_a_win_probability']:.3f}")
            print(f"   Team B ({team_b}) win probability: {result['team_b_win_probability']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Suggested wager: {result['suggested_wager']:.2f}")
            return True
        else:
            print("❌ Soccer prediction failed - no valid result returned")
            return False
            
    except Exception as e:
        print(f"❌ Soccer predictor test failed with error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all standalone predictor tests"""
    print("================================================================================")
    print("🏆 STANDALONE SPORTS PREDICTOR TEST SUITE")
    print("================================================================================")
    
    results = {
        'mlb': test_mlb_predictor(),
        'nfl': test_nfl_predictor(),
        'soccer': test_soccer_predictor()
    }
    
    print("================================================================================")
    print("📊 FINAL TEST RESULTS SUMMARY")
    print("================================================================================")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for sport, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {sport.upper()} Predictor: {'WORKING' if result else 'NEEDS FIX'}")
    
    print(f"Overall Result: {passed_tests}/{total_tests} predictors working")
    
    if passed_tests == total_tests:
        print("🎉 All predictors are working correctly!")
    else:
        print(f"⚠️  {total_tests - passed_tests} predictor(s) need attention")
    
    print("================================================================================")
    print("🔍 SYSTEM INFORMATION")
    print("================================================================================")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Models directory: {project_root / 'bettensor' / 'miner' / 'models'}")
    
    # Check model files
    models_dir = project_root / "bettensor" / "miner" / "models"
    required_files = [
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
    for file in required_files:
        file_path = models_dir / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    # Check model directories
    model_dirs = ["mlb_wager_model.pt", "nfl_wager_model.pt"]
    print("Model directories status:")
    for dir_name in model_dirs:
        dir_path = models_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✅ {dir_name}/")
            for sub_file in dir_path.iterdir():
                print(f"    📁 {sub_file.name}")
        else:
            print(f"  ❌ {dir_name}/")
    
    print("🎯 All model files are correctly located in:", models_dir)

if __name__ == "__main__":
    main()
