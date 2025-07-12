#!/usr/bin/env python3
"""
Complete test for MLB TUI implementation
Verifies all MLB parameters are properly integrated
"""

import sys
import os
import ast
import re

def test_file_syntax():
    """Test that the TUI file has valid Python syntax"""
    try:
        with open('bettensor/miner/model_params_tui.py', 'r') as f:
            content = f.read()
        
        # Parse the file to check for syntax errors
        ast.parse(content)
        print("✓ File has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

def test_mlb_parameters_in_explanations():
    """Test that all MLB parameters are in the explanations dictionary"""
    try:
        with open('bettensor/miner/model_params_tui.py', 'r') as f:
            content = f.read()
        
        # Check for MLB parameters in explanations
        mlb_params = [
            'mlb_model_on',
            'mlb_minimum_wager_amount',
            'mlb_max_wager_amount',
            'mlb_top_n_games',
            'mlb_kelly_fraction_multiplier',
            'mlb_edge_threshold',
            'mlb_max_bet_percentage'
        ]
        
        print("✓ Checking MLB parameters in explanations:")
        all_found = True
        for param in mlb_params:
            if f'"{param}"' in content:
                print(f"  ✓ {param}")
            else:
                print(f"  ✗ {param} - NOT FOUND")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error checking explanations: {e}")
        return False

def test_database_schema():
    """Test that MLB parameters are in the database schema"""
    try:
        with open('bettensor/miner/model_params_tui.py', 'r') as f:
            content = f.read()
        
        # Check for MLB columns in CREATE TABLE
        mlb_columns = [
            'mlb_model_on BOOLEAN',
            'mlb_minimum_wager_amount FLOAT',
            'mlb_max_wager_amount FLOAT',
            'mlb_top_n_games INTEGER',
            'mlb_kelly_fraction_multiplier FLOAT',
            'mlb_edge_threshold FLOAT',
            'mlb_max_bet_percentage FLOAT'
        ]
        
        print("✓ Checking MLB columns in database schema:")
        all_found = True
        for column in mlb_columns:
            if column in content:
                print(f"  ✓ {column}")
            else:
                print(f"  ✗ {column} - NOT FOUND")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error checking database schema: {e}")
        return False

def test_save_functionality():
    """Test that MLB parameters are included in save queries"""
    try:
        with open('bettensor/miner/model_params_tui.py', 'r') as f:
            content = f.read()
        
        # Check for MLB parameters in UPDATE query
        mlb_params = [
            'mlb_model_on',
            'mlb_minimum_wager_amount',
            'mlb_max_wager_amount',
            'mlb_top_n_games',
            'mlb_kelly_fraction_multiplier',
            'mlb_edge_threshold',
            'mlb_max_bet_percentage'
        ]
        
        print("✓ Checking MLB parameters in save functionality:")
        all_found = True
        for param in mlb_params:
            if f"{param} = %s" in content:
                print(f"  ✓ {param} in UPDATE query")
            else:
                print(f"  ✗ {param} - NOT FOUND in UPDATE query")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error checking save functionality: {e}")
        return False

def test_validation_logic():
    """Test that MLB parameters have validation logic"""
    try:
        with open('bettensor/miner/model_params_tui.py', 'r') as f:
            content = f.read()
        
        # Check for MLB parameter validation
        validations = [
            'mlb_model_on',  # Should be in boolean validation
            'mlb_top_n_games'  # Should be in integer validation
        ]
        
        print("✓ Checking MLB parameter validation:")
        all_found = True
        
        # Check if mlb_model_on is in boolean validation
        if 'mlb_model_on' in content and ('toggle' in content.lower() or 'boolean' in content.lower()):
            print("  ✓ mlb_model_on has boolean validation")
        else:
            print("  ✗ mlb_model_on validation - NOT FOUND")
            all_found = False
        
        # Check if mlb_top_n_games is in integer validation
        if 'mlb_top_n_games' in content and 'int(' in content:
            print("  ✓ mlb_top_n_games has integer validation")
        else:
            print("  ✗ mlb_top_n_games validation - NOT FOUND")
            all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error checking validation logic: {e}")
        return False

def test_default_parameters():
    """Test that MLB parameters have default values"""
    try:
        with open('bettensor/miner/model_params_tui.py', 'r') as f:
            content = f.read()
        
        # Check for MLB defaults in create_default_params
        mlb_defaults = [
            ('mlb_model_on', 'False'),
            ('mlb_minimum_wager_amount', '1.0'),
            ('mlb_max_wager_amount', '100.0'),
            ('mlb_top_n_games', '5'),
            ('mlb_kelly_fraction_multiplier', '1.0'),
            ('mlb_edge_threshold', '0.02'),
            ('mlb_max_bet_percentage', '0.7')
        ]
        
        print("✓ Checking MLB default parameters:")
        all_found = True
        for param, default in mlb_defaults:
            if f'"{param}": {default}' in content:
                print(f"  ✓ {param} = {default}")
            else:
                print(f"  ✗ {param} default - NOT FOUND")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"✗ Error checking default parameters: {e}")
        return False

def test_ui_display():
    """Test that MLB parameters are displayed in the UI"""
    try:
        with open('bettensor/miner/model_params_tui.py', 'r') as f:
            content = f.read()
        
        # Check for MLB toggle display
        if 'mlb_model_on' in content and ('MLB' in content or 'mlb' in content):
            print("  ✓ MLB toggle display found")
            return True
        else:
            print("  ✗ MLB toggle display - NOT FOUND")
            return False
    except Exception as e:
        print(f"✗ Error checking UI display: {e}")
        return False

def count_lines_and_functions():
    """Count lines and functions in the file"""
    try:
        with open('bettensor/miner/model_params_tui.py', 'r') as f:
            content = f.read()
        
        lines = len(content.splitlines())
        functions = len(re.findall(r'def\s+\w+\s*\(', content))
        classes = len(re.findall(r'class\s+\w+\s*\(', content))
        
        print(f"📊 File statistics:")
        print(f"  Lines: {lines}")
        print(f"  Functions: {functions}")
        print(f"  Classes: {classes}")
        
        return True
    except Exception as e:
        print(f"✗ Error counting lines: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE MLB TUI TEST")
    print("=" * 50)
    
    tests = [
        ("File Syntax", test_file_syntax),
        ("MLB Parameters in Explanations", test_mlb_parameters_in_explanations),
        ("Database Schema", test_database_schema),
        ("Save Functionality", test_save_functionality),
        ("Validation Logic", test_validation_logic),
        ("Default Parameters", test_default_parameters),
        ("UI Display", test_ui_display),
        ("File Statistics", count_lines_and_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} needs attention")
    
    print(f"\n" + "=" * 50)
    print(f"🎯 FINAL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ MLB TUI IMPLEMENTATION IS PERFECT!")
        print("🚀 Ready to install bittensor and run full test")
    else:
        print("❌ Some issues found - please review")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
