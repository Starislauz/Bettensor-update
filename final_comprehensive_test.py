#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE TEST for ALL SPORTS PREDICTORS
This tests MLB, NFL, and Soccer predictors with proper error handling
"""

import sys
import os
import traceback

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mlb_predictor():
    """Test the MLB predictor"""
    print("="*60)
    print("⚾ TESTING MLB PREDICTOR")
    print("="*60)
    
    try:
        print("1. Initializing MLB Predictor...")
        from bettensor.miner.models.mlb_predictor_fixed import MLBPredictor
        
        mlb_predictor = MLBPredictor()
        print("   ✅ MLB Predictor initialized successfully")
        
        print("\n2. Testing single MLB game prediction...")
        result = mlb_predictor.predict_game("New York Yankees", "Boston Red Sox", 1.80, 2.10)
        
        if 'error' in result:
            print(f"   ❌ Single game prediction failed: {result['error']}")
            return False
        else:
            print("   ✅ Single game prediction successful!")
            print(f"      Game: {result['home_team']} vs {result['away_team']}")
            print(f"      Prediction: {result['predicted_outcome']}")
            print(f"      Confidence: {result['confidence']}")
            print(f"      Wager: ${result['wager']}")
        
        print("\n3. Testing multiple MLB games prediction...")
        results = mlb_predictor.predict_games(
            home_teams=["New York Yankees", "Los Angeles Dodgers", "Houston Astros"],
            away_teams=["Boston Red Sox", "San Francisco Giants", "Texas Rangers"],
            odds=[[1.80, 0.0, 2.10], [1.65, 0.0, 2.25], [1.90, 0.0, 1.95]]
        )
        
        if len(results) > 0:
            print(f"   ✅ Multiple games prediction successful! ({len(results)} games)")
            for i, result in enumerate(results):
                print(f"      Game {i+1}: {result['Home Team']} vs {result['Away Team']}")
                print(f"        -> {result['PredictedOutcome']} (Confidence: {result['ConfidenceScore']}, Wager: ${result['recommendedWager']})")
        else:
            print("   ❌ Multiple games prediction failed")
            return False
            
        print("\n🎉 MLB PREDICTOR TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ MLB predictor test failed with error: {e}")
        traceback.print_exc()
        return False

def test_nfl_predictor():
    """Test the NFL predictor"""
    print("\n" + "="*60)
    print("🏈 TESTING NFL PREDICTOR")
    print("="*60)
    
    try:
        print("1. Initializing NFL Predictor...")
        from bettensor.miner.models.nfl_predictor_completely_fixed import NFLPredictorFixed
        
        nfl_predictor = NFLPredictorFixed()
        print("   ✅ NFL Predictor initialized successfully")
        
        print("\n2. Testing single NFL game prediction...")
        result = nfl_predictor.predict_game("Green Bay Packers", "Buffalo Bills", 1.85, 1.95)
        
        if 'error' in result:
            print(f"   ❌ Single game prediction failed: {result['error']}")
            return False
        else:
            print("   ✅ Single game prediction successful!")
            print(f"      Game: {result['home_team']} vs {result['away_team']}")
            print(f"      Prediction: {result['predicted_outcome']}")
            print(f"      Confidence: {result['confidence']}")
            print(f"      Wager: ${result['wager']}")
        
        print("\n3. Testing multiple NFL games prediction...")
        results = nfl_predictor.predict_games(
            home_teams=["Green Bay Packers", "Buffalo Bills", "Carolina Panthers"],
            away_teams=["Buffalo Bills", "Carolina Panthers", "Washington Redskins"],
            odds=[[1.85, 1.95], [1.75, 2.05], [1.90, 1.90]]
        )
        
        if len(results) > 0:
            print(f"   ✅ Multiple games prediction successful! ({len(results)} games)")
            for i, result in enumerate(results):
                print(f"      Game {i+1}: {result['Home Team']} vs {result['Away Team']}")
                print(f"        -> {result['PredictedOutcome']} (Confidence: {result['ConfidenceScore']}, Wager: ${result['recommendedWager']})")
        else:
            print("   ❌ Multiple games prediction failed")
            return False
            
        print("\n🎉 NFL PREDICTOR TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ NFL predictor test failed with error: {e}")
        traceback.print_exc()
        return False

def test_soccer_predictor():
    """Test the Soccer predictor"""
    print("\n" + "="*60)
    print("⚽ TESTING SOCCER PREDICTOR")
    print("="*60)
    
    try:
        print("1. Initializing Soccer Predictor...")
        from bettensor.miner.models.soccer_predictor_completely_fixed import SoccerPredictorFixed
        
        soccer_predictor = SoccerPredictorFixed()
        print("   ✅ Soccer Predictor initialized successfully")
        
        print("\n2. Testing single soccer game prediction...")
        result = soccer_predictor.predict_game("Manchester United", "Liverpool", 2.10, 3.20, 3.40)
        
        if 'error' in result:
            print(f"   ❌ Single game prediction failed: {result['error']}")
            return False
        else:
            print("   ✅ Single game prediction successful!")
            print(f"      Game: {result['home_team']} vs {result['away_team']}")
            print(f"      Prediction: {result['predicted_outcome']}")
            print(f"      Confidence: {result['confidence']}")
            print(f"      Wager: ${result['wager']}")
        
        print("\n3. Testing multiple soccer games prediction...")
        results = soccer_predictor.predict_games(
            home_teams=["Manchester United", "Barcelona", "Real Madrid"],
            away_teams=["Liverpool", "Atletico Madrid", "Bayern Munich"],
            odds=[[2.10, 3.20, 3.40], [1.85, 3.50, 4.20], [1.95, 3.30, 3.80]]
        )
        
        if len(results) > 0:
            print(f"   ✅ Multiple games prediction successful! ({len(results)} games)")
            for i, result in enumerate(results):
                print(f"      Game {i+1}: {result['Home Team']} vs {result['Away Team']}")
                print(f"        -> {result['PredictedOutcome']} (Confidence: {result['ConfidenceScore']}, Wager: ${result['recommendedWager']})")
        else:
            print("   ❌ Multiple games prediction failed")
            return False
            
        print("\n🎉 SOCCER PREDICTOR TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Soccer predictor test failed with error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all predictor tests"""
    print("="*80)
    print("🏆 COMPREHENSIVE SPORTS PREDICTOR TEST SUITE")
    print("="*80)
    
    results = {
        'MLB': test_mlb_predictor(),
        'NFL': test_nfl_predictor(), 
        'Soccer': test_soccer_predictor()
    }
    
    print("\n" + "="*80)
    print("📊 FINAL TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for sport, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status} {sport} Predictor: {'WORKING' if passed_test else 'NEEDS FIX'}")
    
    print(f"\nOverall Result: {passed}/{total} predictors working")
    
    if passed == total:
        print("🎉 ALL PREDICTORS ARE WORKING!")
        print("✅ MLB system is ready for baseball predictions")
        print("✅ NFL system is ready for football predictions") 
        print("✅ Soccer system is ready for soccer predictions")
    else:
        print(f"⚠️  {total - passed} predictor(s) need attention")
        if results['MLB']:
            print("✅ MLB system is ready for baseball predictions")
        if results['NFL']:
            print("✅ NFL system is ready for football predictions")
        if results['Soccer']:
            print("✅ Soccer system is ready for soccer predictions")
    
    print("\n" + "="*80)
    print("🔍 SYSTEM INFORMATION")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Models directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Check model files
    model_files = [
        "mlb_predictor_fixed.py",
        "nfl_predictor_completely_fixed.py", 
        "soccer_predictor_fixed.py",
        "mlb_team_stats.csv",
        "mlb_calibrated_model.joblib",
        "mlb_preprocessor.joblib",
        "calibrated_sklearn_model.joblib",
        "preprocessor.joblib",
        "label_encoder.pkl",
        "team_historical_stats.csv",
        "team_averages_last_5_games_aug.csv"
    ]
    
    print("\nModel files status:")
    for file in model_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"  {status} {file}")

if __name__ == "__main__":
    main()
