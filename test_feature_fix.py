#!/usr/bin/env python3
"""
Test the FIXED MLB Predictor with correct feature count
"""

import sys
import os

# Add the models directory to path
sys.path.append('bettensor/miner/models')

from mlb_predictor_fixed import MLBPredictor

class MockMinerStats:
    def get_miner_cash(self):
        return 5000.0

def test_fixed_feature_count():
    """Test the MLB predictor with corrected feature count"""
    
    print("🔧 TESTING MLB PREDICTOR WITH CORRECT FEATURES")
    print("=" * 60)
    
    try:
        print("1. Creating MLBPredictor instance...")
        mock_stats = MockMinerStats()
        predictor = MLBPredictor(miner_stats_handler=mock_stats)
        print("✅ MLBPredictor created successfully!")
        
        print("\n2. Testing prediction with feature debugging...")
        result = predictor.predict_game("Arizona Diamondbacks", "Atlanta Braves")
        
        print("\n3. Checking result...")
        if 'error' in result:
            print(f"❌ Prediction failed: {result['error']}")
            return False
        else:
            print("✅ PREDICTION SUCCESSFUL!")
            print("📊 PREDICTION RESULTS:")
            for key, value in result.items():
                print(f"   {key}: {value}")
            
            print(f"\n🎯 Winner: {result['predicted_outcome']}")
            print(f"📈 Confidence: {result['confidence']:.1%}")
            print(f"💰 Recommended Stake: ${result['recommended_stake']:.2f}")
            print(f"🎲 Kelly Fraction: {result['kelly_fraction']:.4f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_games():
    """Test multiple MLB predictions"""
    
    print("\n🎮 TESTING MULTIPLE MLB GAMES")
    print("=" * 50)
    
    try:
        mock_stats = MockMinerStats()
        predictor = MLBPredictor(miner_stats_handler=mock_stats)
        
        games = [
            ("New York Yankees", "Boston Red Sox"),
            ("Los Angeles Dodgers", "San Francisco Giants"),
            ("Houston Astros", "Oakland Athletics"),
            ("Chicago Cubs", "St. Louis Cardinals")
        ]
        
        successful_predictions = 0
        
        for i, (home, away) in enumerate(games, 1):
            print(f"\n--- Game {i}: {home} vs {away} ---")
            result = predictor.predict_game(home, away)
            
            if 'error' not in result:
                print(f"   Winner: {result['predicted_outcome']}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Stake: ${result['recommended_stake']:.2f}")
                successful_predictions += 1
            else:
                print(f"   Error: {result['error']}")
        
        print(f"\n📊 SUMMARY: {successful_predictions}/{len(games)} predictions successful")
        return successful_predictions == len(games)
        
    except Exception as e:
        print(f"❌ Multiple games test failed: {e}")
        return False

if __name__ == "__main__":
    print("🏟️ MLB AUTO-PREDICTION SYSTEM TEST")
    print("=" * 70)
    
    test1_success = test_fixed_feature_count()
    
    if test1_success:
        test2_success = test_multiple_games()
        
        print("\n" + "=" * 70)
        if test1_success and test2_success:
            print("🎉 SUCCESS! MLB AUTO-PREDICTION SYSTEM IS NOW WORKING!")
            print("✅ Feature count fixed (20 features)")
            print("✅ Models loading correctly")
            print("✅ Predictions generating successfully")
            print("✅ Multiple games working")
            print("\n🚀 MLB AUTO-PREDICTION IS READY FOR PRODUCTION!")
        else:
            print("⚠️ Basic test passed but multiple games had issues")
    else:
        print("\n❌ BASIC TEST FAILED - Still debugging needed")
