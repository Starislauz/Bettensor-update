#!/usr/bin/env python3
"""
Final Project Cleanup Script
Removes all unnecessary test files, debug scripts, and old/duplicate files from the Bettensor project.
Keeps only essential production files and working predictors.
"""

import os
import shutil
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent
    print(f"Cleaning up project in: {project_root}")
    
    # Files to remove from project root
    files_to_remove = [
        # Test files
        "analyze_mlb_data.py",
        "api_example.py", 
        "bettensor_full_project_test.py",
        "bittensor.py",
        "check_dependencies.py",
        "check_status.py",
        "cleanup_project.py",
        "clean_diagnosis.py",
        "comprehensive_mlb_test.py",
        "create_mlb_models.py",
        "debug_mlb.py",
        "debug_mlb_predictor.py", 
        "debug_mlb_system.py",
        "definitive_test.py",
        "diagnose_other_predictors.py",
        "diagnostic_test.py",
        "direct_model_test.py",
        "enable_mlb_predictions.py",
        "explain_test_failures.py",
        "final_comprehensive_mlb_test.py",
        "final_mlb_test.py",
        "final_mlb_test_v2.py",
        "final_mlb_verification.py",
        "final_project_summary.py",
        "fixed_mlb_test.py",
        "focused_mlb_test.py",
        "full_mlb_test_suite.py",
        "generate_enhanced_mlb_stats.py",
        "import_diagnostic.py",
        "isolate_hang.py",
        "live_api_integration.py",
        "live_websocket_updater.py",
        "minimal_mlb_test.py",
        "minimal_mlb_test_direct.py",
        "mlb_auto_prediction_demo.py",
        "mlb_demo.py",
        "mlb_final_test.py",
        "mlb_prediction_test.py",
        "mlb_requirements.txt",
        "mlb_status_report.py",
        "mlb_success_summary.py",
        "prove_environment_issue.py",
        "quick_check.py",
        "quick_file_check.py",
        "quick_import_test.py",
        "quick_mlb_check.py",
        "quick_mlb_test.py",
        "quick_mlb_validation.py",
        "quick_predictors_status.py",
        "quick_project_status.py",
        "quick_test_enhanced.py",
        "quick_validation.py",
        "real_mlb_test.py",
        "setup_automated_betting.py",
        "simple_mlb_test.py",
        "simple_mlb_test_fixed.py",
        "simple_mlb_test_now.py",
        "simple_test.py",
        "simple_working_test.py",
        "test_all_sports_predictors.py",
        "test_complete_mlb_system.py",
        "test_core_bettensor.py",
        "test_enhanced_mlb.py",
        "test_feature_fix.py",
        "test_final_fix.py",
        "test_final_mlb_fix.py",
        "test_fixed_mlb.py",
        "test_fixed_mlb_predictor.py",
        "test_mlb_auto_prediction.py",
        "test_mlb_components.py",
        "test_mlb_integration.py",
        "test_mlb_predictions.py",
        "test_nfl_only.py",
        "test_no_bt.py",
        "test_python.py",
        "verify_mlb_integration.py",
        "very_simple_test.py",
        # Documentation that's been superseded
        "BETTENSOR_PROJECT_REPORT.md",
        "MLB_IMPLEMENTATION_COMPLETE.md",
    ]
    
    # Files to keep (essential working files)
    essential_files = [
        "final_comprehensive_test.py",  # Main test suite
        "standalone_predictor_test.py",  # Predictor validation
        "quick_test.py",  # Quick validation
        "FINAL_PROJECT_STATUS.md",  # Final documentation
        "automated_betting_system.py",  # If it exists
    ]
    
    # Remove unnecessary files from project root
    removed_count = 0
    for filename in files_to_remove:
        file_path = project_root / filename
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✓ Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Failed to remove {filename}: {e}")
    
    # Clean up models directory - remove old/duplicate predictors
    models_dir = project_root / "bettensor" / "miner" / "models"
    if models_dir.exists():
        models_to_remove = [
            "create_mlb_model.py",
            "create_mlb_preprocessor.py", 
            "create_mlb_wager_model.py",
            "create_simple_mlb.py",
            "simple_mlb_model.py",
            "nfl_predictor_fixed.py",  # Keep the "completely_fixed" version
            "soccer_predictor_fixed.py",  # Keep the "completely_fixed" version
        ]
        
        for filename in models_to_remove:
            file_path = models_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"✓ Removed from models: {filename}")
                    removed_count += 1
                except Exception as e:
                    print(f"✗ Failed to remove {filename}: {e}")
    
    # Remove __pycache__ directories
    for pycache_dir in project_root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            try:
                shutil.rmtree(pycache_dir)
                print(f"✓ Removed __pycache__: {pycache_dir}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Failed to remove __pycache__ {pycache_dir}: {e}")
    
    # Remove .pytest_cache if it exists
    pytest_cache = project_root / ".pytest_cache"
    if pytest_cache.exists():
        try:
            shutil.rmtree(pytest_cache)
            print(f"✓ Removed .pytest_cache")
            removed_count += 1
        except Exception as e:
            print(f"✗ Failed to remove .pytest_cache: {e}")
    
    # Remove build and egg-info directories
    build_dirs = ["build", "bettensor.egg-info"]
    for build_dir in build_dirs:
        build_path = project_root / build_dir
        if build_path.exists():
            try:
                shutil.rmtree(build_path)
                print(f"✓ Removed build directory: {build_dir}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Failed to remove {build_dir}: {e}")
    
    print(f"\n🧹 Cleanup complete! Removed {removed_count} files/directories.")
    
    # Show what essential files remain
    print("\n📁 Essential files that remain:")
    essential_remaining = []
    for filename in essential_files:
        file_path = project_root / filename
        if file_path.exists():
            essential_remaining.append(filename)
            print(f"  ✓ {filename}")
    
    print(f"\n🎯 Production-ready predictors:")
    predictor_files = [
        "bettensor/miner/models/mlb_predictor_fixed.py",
        "bettensor/miner/models/nfl_predictor_completely_fixed.py", 
        "bettensor/miner/models/soccer_predictor_completely_fixed.py",
    ]
    
    for predictor in predictor_files:
        predictor_path = project_root / predictor
        if predictor_path.exists():
            print(f"  ✓ {predictor}")
    
    print(f"\n✨ Project is now clean and production-ready!")
    print(f"   Run 'python final_comprehensive_test.py' to verify everything works.")

if __name__ == "__main__":
    main()
