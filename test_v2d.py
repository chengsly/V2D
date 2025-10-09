#!/usr/bin/env python3
"""
Test script for v2d.py
Creates mock data and tests different model configurations
"""

import numpy as np
import pandas as pd
import os
import subprocess
import gzip
from pathlib import Path
import shutil

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
TEST_DIR = Path("test_data")
ANNOT_PREFIX = TEST_DIR / "mock_annot."
BETA2_FILE = TEST_DIR / "mock_beta2.txt"
OUTPUT_PREFIX = TEST_DIR / "test_output"

# Number of SNPs per chromosome
N_SNPS_CHR21 = 100
N_SNPS_CHR22 = 100
N_FEATURES = 5

def create_test_directory():
    """Create test directory if it doesn't exist"""
    TEST_DIR.mkdir(exist_ok=True)
    print(f"Created test directory: {TEST_DIR}")

def create_mock_annotations():
    """
    Create mock annotation files for chromosomes 21 and 22
    Format: CHR, BP, rsid, ID, feature1, feature2, ..., featureN
    """
    print("\n=== Creating Mock Annotation Files ===")
    
    for chr_num in [21, 22]:
        n_snps = N_SNPS_CHR21 if chr_num == 21 else N_SNPS_CHR22
        
        # Generate mock data
        data = {
            'CHR': [chr_num] * n_snps,
            'BP': np.arange(1000000, 1000000 + n_snps * 1000, 1000),  # positions every 1kb
            'rsid': [f'rs{chr_num}{i:06d}' for i in range(n_snps)],
            'ID': [f'snp_{chr_num}_{i}' for i in range(n_snps)]
        }
        
        # Add features (random values between 0 and 1)
        for feat_idx in range(1, N_FEATURES + 1):
            data[f'feature{feat_idx}'] = np.random.rand(n_snps)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save as gzipped file
        annot_file = f"{ANNOT_PREFIX}{chr_num}.annot.gz"
        df.to_csv(annot_file, sep='\t', index=False, compression='gzip')
        print(f"  Created: {annot_file} ({n_snps} SNPs, {N_FEATURES} features)")
    
    return True

def create_mock_beta2():
    """
    Create mock beta2 file with target values
    Format: CHR, BP, rsid, ID, beta2
    The beta2 values will be synthetic - a linear combination of features plus noise
    """
    print("\n=== Creating Mock Beta2 File ===")
    
    # Read all annotation files to get SNP info
    all_snps = []
    for chr_num in [21, 22]:
        annot_file = f"{ANNOT_PREFIX}{chr_num}.annot.gz"
        df = pd.read_csv(annot_file, sep='\t', compression='gzip')
        all_snps.append(df)
    
    combined = pd.concat(all_snps, ignore_index=True)
    
    # Generate synthetic beta2 values
    # Make it a realistic function of the features
    # beta2 = 2*feature1 + 0.5*feature2 - feature3 + noise
    beta2_values = (
        2.0 * combined['feature1'] + 
        0.5 * combined['feature2'] - 
        1.0 * combined['feature3'] +
        0.1 * np.random.randn(len(combined))  # small noise
    )
    
    # Keep only the first 4 columns and add beta2
    beta2_df = combined[['CHR', 'BP', 'rsid', 'ID']].copy()
    beta2_df['beta2'] = beta2_values
    
    # Save as text file
    beta2_df.to_csv(BETA2_FILE, sep='\t', index=False)
    print(f"  Created: {BETA2_FILE} ({len(beta2_df)} SNPs)")
    
    return True

def run_v2d_test(model, model_args, test_name):
    """
    Run v2d.py with specified model and arguments
    
    Args:
        model: Model type (linear, tree, rf, mlp, xgb)
        model_args: Dict of model-specific arguments
        test_name: Name for this test (used in output)
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        'python', 'v2d.py',
        '--model', model,
        '--annot', str(ANNOT_PREFIX),
        '--beta2', str(BETA2_FILE),
        '--out', f"{OUTPUT_PREFIX}_{test_name}",
        '--delimiter', '\t',
        '--print_mse'
    ]
    
    # Add model-specific arguments
    for arg_name, arg_value in model_args.items():
        cmd.extend([f'--{arg_name}', str(arg_value)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print(f"✓ Test PASSED: {test_name}")
            print(f"  Output files created with prefix: {OUTPUT_PREFIX}_{test_name}")
            return True
        else:
            print(f"✗ Test FAILED: {test_name}")
            print(f"  Return code: {result.returncode}")
            print(f"  STDERR:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Test ERROR: {test_name}")
        print(f"  Exception: {e}")
        return False

def run_all_tests():
    """Run tests for all model types"""
    
    test_results = []
    
    # Test 1: Linear model
    test_results.append(
        run_v2d_test(
            model='linear',
            model_args={},
            test_name='linear'
        )
    )
    
    # Test 2: Decision Tree
    test_results.append(
        run_v2d_test(
            model='tree',
            model_args={
                'max_depth': 5,
                'min_samples_leaf': 5
            },
            test_name='tree'
        )
    )
    
    # Test 3: Random Forest
    test_results.append(
        run_v2d_test(
            model='rf',
            model_args={
                'n_estimators': 10,
                'max_depth': 5,
                'min_samples_leaf': 5
            },
            test_name='rf'
        )
    )
    
    # Test 4: MLP (Neural Network)
    test_results.append(
        run_v2d_test(
            model='mlp',
            model_args={
                'n_neurons': 10,
                'n_layers': 2,
                'nbseed': 1
            },
            test_name='mlp'
        )
    )
    
    # Test 5: XGBoost
    test_results.append(
        run_v2d_test(
            model='xgb',
            model_args={
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1,
                'min_child_weight': 1,
                'gamma': 0.0,
                'subsample': 0.8,
                'scale_pos_weight': 1
            },
            test_name='xgb'
        )
    )
    
    # Test 6: Linear with prediction output
    pred_test_dir = TEST_DIR / "pred_test"
    pred_test_dir.mkdir(exist_ok=True)
    test_results.append(
        run_v2d_test(
            model='linear',
            model_args={
                'pred': str(ANNOT_PREFIX)
            },
            test_name='linear_with_pred'
        )
    )
    
    return test_results

def print_summary(test_results):
    """Print summary of test results"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")
    
    print("\nGenerated files:")
    if TEST_DIR.exists():
        for f in sorted(TEST_DIR.glob("*")):
            print(f"  {f}")

def cleanup():
    """Clean up test files (optional)"""
    response = input("\nClean up test files? (y/n): ").lower().strip()
    if response == 'y':
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)
            print(f"✓ Cleaned up {TEST_DIR}")
    else:
        print(f"Test files kept in {TEST_DIR}")

def main():
    """Main test execution"""
    print("="*60)
    print("V2D.PY TEST SUITE")
    print("="*60)
    
    # Setup
    create_test_directory()
    create_mock_annotations()
    create_mock_beta2()
    
    # Run tests
    test_results = run_all_tests()
    
    # Summary
    print_summary(test_results)
    
    # Optional cleanup
    cleanup()

if __name__ == "__main__":
    main()

