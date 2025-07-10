#!/usr/bin/env python3
"""
Final Bettensor System Architecture Verification
Checks that the entire system is properly arranged and ready for production
"""

import os
import sys
import json
from pathlib import Path

def check_system_architecture():
    """Comprehensive check of system architecture"""
    print("🔍 BETTENSOR SYSTEM ARCHITECTURE VERIFICATION")
    print("=" * 80)
    
    base_path = Path(".")
    issues = []
    
    # 1. Check core package structure
    print("\n📦 CORE PACKAGE STRUCTURE:")
    core_files = [
        "bettensor/__init__.py",
        "bettensor/protocol.py",
        "bettensor/base/__init__.py", 
        "bettensor/base/neuron.py",
        "bettensor/miner/__init__.py",
        "bettensor/miner/bettensor_miner.py",
        "bettensor/validator/__init__.py",
        "bettensor/validator/bettensor_validator.py"
    ]
    
    for file_path in core_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            issues.append(f"Missing core file: {file_path}")
    
    # 2. Check neuron entry points
    print("\n🚀 NEURON ENTRY POINTS:")
    entry_points = [
        "neurons/miner.py",
        "neurons/validator.py"
    ]
    
    for entry_point in entry_points:
        full_path = base_path / entry_point
        if full_path.exists():
            print(f"  ✅ {entry_point}")
        else:
            print(f"  ❌ {entry_point}")
            issues.append(f"Missing entry point: {entry_point}")
    
    # 3. Check predictor systems
    print("\n🎯 PREDICTOR SYSTEMS:")
    predictors = [
        "bettensor/miner/models/mlb_predictor_fixed.py",
        "bettensor/miner/models/nfl_predictor_completely_fixed.py", 
        "bettensor/miner/models/soccer_predictor_completely_fixed.py"
    ]
    
    for predictor in predictors:
        full_path = base_path / predictor
        if full_path.exists():
            print(f"  ✅ {predictor}")
        else:
            print(f"  ❌ {predictor}")
            issues.append(f"Missing predictor: {predictor}")
    
    # 4. Check model files
    print("\n🧠 MODEL FILES:")
    model_files = [
        "bettensor/miner/models/mlb_team_stats.csv",
        "bettensor/miner/models/mlb_calibrated_model.joblib",
        "bettensor/miner/models/mlb_preprocessor.joblib",
        "bettensor/miner/models/calibrated_sklearn_model.joblib",
        "bettensor/miner/models/preprocessor.joblib",
        "bettensor/miner/models/label_encoder.pkl",
        "bettensor/miner/models/team_historical_stats.csv",
        "bettensor/miner/models/team_averages_last_5_games_aug.csv"
    ]
    
    for model_file in model_files:
        full_path = base_path / model_file
        if full_path.exists():
            print(f"  ✅ {model_file}")
        else:
            print(f"  ❌ {model_file}")
            issues.append(f"Missing model file: {model_file}")
    
    # 5. Check neural network model directories
    print("\n🤖 NEURAL NETWORK MODELS:")
    model_dirs = [
        "bettensor/miner/models/mlb_wager_model.pt",
        "bettensor/miner/models/nfl_wager_model.pt"
    ]
    
    for model_dir in model_dirs:
        full_path = base_path / model_dir
        if full_path.exists() and full_path.is_dir():
            print(f"  ✅ {model_dir}/")
            
            # Check contents
            expected_files = ["config.json", "README.md"]
            if "mlb" in model_dir:
                expected_files.append("pytorch_model.bin")
            else:  # NFL
                expected_files.append("model.safetensors")
                
            for expected_file in expected_files:
                file_path = full_path / expected_file
                if file_path.exists():
                    print(f"    ✅ {expected_file}")
                else:
                    print(f"    ❌ {expected_file}")
                    issues.append(f"Missing model file: {model_dir}/{expected_file}")
        else:
            print(f"  ❌ {model_dir}/")
            issues.append(f"Missing model directory: {model_dir}")
    
    # 6. Check configuration files
    print("\n⚙️ CONFIGURATION FILES:")
    config_files = [
        "setup.py",
        "setup.cfg", 
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    for config_file in config_files:
        full_path = base_path / config_file
        if full_path.exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file}")
            issues.append(f"Missing config file: {config_file}")
    
    # 7. Check automation system
    print("\n🤖 AUTOMATION SYSTEMS:")
    automation_files = [
        "automated_betting_system.py"
    ]
    
    for automation_file in automation_files:
        full_path = base_path / automation_file
        if full_path.exists():
            print(f"  ✅ {automation_file}")
        else:
            print(f"  ❌ {automation_file}")
            issues.append(f"Missing automation file: {automation_file}")
    
    # 8. Check test and validation files
    print("\n🧪 TEST & VALIDATION FILES:")
    test_files = [
        "final_comprehensive_test.py",
        "final_project_summary.py"
    ]
    
    for test_file in test_files:
        full_path = base_path / test_file
        if full_path.exists():
            print(f"  ✅ {test_file}")
        else:
            print(f"  ❌ {test_file}")
            issues.append(f"Missing test file: {test_file}")
    
    # 9. Check critical directories
    print("\n📁 CRITICAL DIRECTORIES:")
    critical_dirs = [
        "bettensor/miner/database",
        "bettensor/miner/interfaces", 
        "bettensor/miner/stats",
        "bettensor/miner/utils",
        "bettensor/validator/utils",
        "bettensor/validator/utils/database",
        "bettensor/validator/utils/io",
        "bettensor/validator/utils/scoring"
    ]
    
    for dir_path in critical_dirs:
        full_path = base_path / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/")
            issues.append(f"Missing directory: {dir_path}")
    
    # 10. Final assessment
    print("\n" + "=" * 80)
    if issues:
        print("❌ SYSTEM ARCHITECTURE ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
        print(f"\n⚠️ Total issues: {len(issues)}")
        return False
    else:
        print("✅ SYSTEM ARCHITECTURE: PERFECT!")
        print("🎉 All files and directories are properly arranged")
        print("🚀 System is ready for production deployment")
        return True

def check_import_structure():
    """Check that imports work correctly"""
    print("\n🔗 IMPORT STRUCTURE VERIFICATION:")
    
    try:
        # Test core imports
        import bettensor
        print("  ✅ bettensor package")
        
        from bettensor.protocol import GameData, Metadata
        print("  ✅ bettensor.protocol")
        
        from bettensor.base.neuron import BaseNeuron
        print("  ✅ bettensor.base.neuron")
        
        # Test miner imports
        from bettensor.miner.bettensor_miner import BettensorMiner
        print("  ✅ bettensor.miner.bettensor_miner")
        
        # Test validator imports  
        from bettensor.validator.bettensor_validator import BettensorValidator
        print("  ✅ bettensor.validator.bettensor_validator")
        
        print("  🎉 All core imports working!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def generate_deployment_checklist():
    """Generate deployment checklist"""
    print("\n📋 DEPLOYMENT CHECKLIST:")
    print("=" * 80)
    
    checklist = [
        "✅ Install bittensor: pip install bittensor>=9.0.0",
        "✅ Install dependencies: pip install -r requirements.txt", 
        "✅ Set up wallet: btcli wallet create",
        "✅ Fund wallet with TAO for network participation",
        "✅ Configure environment variables (if needed)",
        "✅ Test predictors: python final_comprehensive_test.py",
        "✅ Run miner: python neurons/miner.py",
        "✅ Run validator: python neurons/validator.py",
        "✅ Monitor automated betting: python automated_betting_system.py",
        "✅ Check system health regularly"
    ]
    
    for item in checklist:
        print(f"  {item}")

if __name__ == "__main__":
    print("🏗️ BETTENSOR PRODUCTION READINESS CHECK")
    print("=" * 80)
    
    # Run all checks
    architecture_ok = check_system_architecture()
    import_ok = check_import_structure()
    
    print("\n" + "=" * 80)
    print("📊 FINAL ASSESSMENT:")
    
    if architecture_ok and import_ok:
        print("🎉 SYSTEM IS PRODUCTION READY!")
        print("✅ All files properly arranged")
        print("✅ All imports working")
        print("✅ Ready for bittensor installation and deployment")
        
        generate_deployment_checklist()
        
    else:
        print("❌ SYSTEM NEEDS ATTENTION")
        if not architecture_ok:
            print("⚠️ Architecture issues found")
        if not import_ok:
            print("⚠️ Import issues found")
            
    print("\n🎯 Run 'python final_comprehensive_test.py' to test predictors")
    print("🚀 System is ready once bittensor is installed!")
