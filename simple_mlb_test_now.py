#!/usr/bin/env python3
"""
Simple test for MLB auto-prediction
"""

import sys
import os

print("🏟️ MLB AUTO-PREDICTION TEST")
print("=" * 40)

try:
    print("Step 1: Adding path...")
    sys.path.insert(0, os.path.join(os.getcwd(), 'bettensor', 'miner', 'models'))
    
    print("Step 2: Importing fixed MLB predictor...")
    from mlb_predictor_fixed import MLBPredictor
    print("✅ Import successful!")
    
    print("Step 3: Creating predictor...")
    class MockStats:
        def get_miner_cash(self):
            return 5000.0
    
    predictor = MLBPredictor(miner_stats_handler=MockStats())
    print("✅ Predictor created!")
    
    print("Step 4: Making prediction...")
    result = predictor.predict_game("New York Yankees", "Boston Red Sox")
    
    print("Step 5: Results:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    if 'error' not in result:
        print("\n🎉 SUCCESS! MLB AUTO-PREDICTION IS WORKING!")
    else:
        print(f"\n⚠️ Prediction error: {result['error']}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
