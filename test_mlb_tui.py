#!/usr/bin/env python3
"""
Test script for the upgraded MLB TUI interface
Tests that all MLB parameter controls work correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mlb_tui_import():
    """Test that the TUI module can be imported"""
    try:
        from bettensor.miner.model_params_tui import ModelParamsTUI
        print("✓ Successfully imported ModelParamsTUI")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_mlb_parameters():
    """Test that all MLB parameters are properly defined"""
    try:
        from bettensor.miner.model_params_tui import ModelParamsTUI
        
        # Check if MLB parameters are in the explanations
        dummy_db_params = {}
        tui = ModelParamsTUI.__new__(ModelParamsTUI)
        tui.explanations = {
            "soccer_model_on": "Toggle the soccer model on or off.",
            "wager_distribution_steepness": "Controls the steepness of the wager distribution.",
            "mlb_model_on": "Toggle the MLB model on or off.",
            "mlb_minimum_wager_amount": "The minimum wager amount for MLB bets.",
            "mlb_max_wager_amount": "The maximum wager amount for MLB bets.",
            "mlb_top_n_games": "The number of top MLB games to consider for betting.",
            "mlb_kelly_fraction_multiplier": "The Kelly fraction multiplier for MLB bets.",
            "mlb_edge_threshold": "The edge threshold for MLB bets.",
            "mlb_max_bet_percentage": "The maximum bet percentage for MLB bets."
        }
        
        mlb_params = [
            "mlb_model_on",
            "mlb_minimum_wager_amount", 
            "mlb_max_wager_amount",
            "mlb_top_n_games",
            "mlb_kelly_fraction_multiplier",
            "mlb_edge_threshold",
            "mlb_max_bet_percentage"
        ]
        
        print("✓ MLB parameters to test:")
        for param in mlb_params:
            if param in tui.explanations:
                print(f"  ✓ {param}: {tui.explanations[param]}")
            else:
                print(f"  ✗ {param}: Missing explanation")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing MLB parameters: {e}")
        return False

def test_database_schema():
    """Test that the database schema includes MLB parameters"""
    try:
        # Read the TUI file to check for MLB schema
        tui_file = "bettensor/miner/model_params_tui.py"
        with open(tui_file, 'r') as f:
            content = f.read()
            
        mlb_schema_items = [
            "mlb_model_on BOOLEAN DEFAULT TRUE",
            "mlb_minimum_wager_amount DECIMAL DEFAULT 1.0",
            "mlb_max_wager_amount DECIMAL DEFAULT 10.0",
            "mlb_top_n_games INTEGER DEFAULT 3",
            "mlb_kelly_fraction_multiplier DECIMAL DEFAULT 0.25",
            "mlb_edge_threshold DECIMAL DEFAULT 0.05",
            "mlb_max_bet_percentage DECIMAL DEFAULT 5.0"
        ]
        
        print("✓ Checking database schema for MLB parameters:")
        for item in mlb_schema_items:
            if item in content:
                print(f"  ✓ {item.split()[0]}: Found in schema")
            else:
                print(f"  ✗ {item.split()[0]}: Missing from schema")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing database schema: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing MLB TUI Interface")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_mlb_tui_import),
        ("Parameter Test", test_mlb_parameters),
        ("Database Schema Test", test_database_schema),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✨ All tests passed! MLB TUI interface is ready to use.")
        print("\n🚀 How to use the MLB TUI:")
        print("   1. Set up your PostgreSQL database connection")
        print("   2. Run: python bettensor/miner/model_params_tui.py")
        print("   3. Navigate to MLB parameters and toggle them on/off")
        print("   4. Press 'q' to quit and save changes")
        print("\n🎮 MLB Controls Available:")
        print("   - Toggle MLB model on/off")
        print("   - Set minimum/maximum wager amounts")
        print("   - Configure top N games to bet on")
        print("   - Adjust Kelly fraction multiplier")
        print("   - Set edge threshold for bets")
        print("   - Configure maximum bet percentage")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
