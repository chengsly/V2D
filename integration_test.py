#!/usr/bin/env python3
"""
Integration test for v2d.py workflow:
1. Create mock data
2. Train a decision tree model and save it
3. Load and visualize the tree using plot_tree.py
"""

import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path
import shutil

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
TEST_DIR = Path("integration_test_data")
ANNOT_PREFIX = TEST_DIR / "annot."
BETA2_FILE = TEST_DIR / "beta2.txt"
OUTPUT_PREFIX = TEST_DIR / "tree_model"
TREE_PLOT_OUTPUT = TEST_DIR / "tree_visualization.pdf"

# Number of SNPs per chromosome
N_SNPS_CHR21 = 150
N_SNPS_CHR22 = 150
N_FEATURES = 8

def print_step(step_num, description):
    """Print a formatted step header"""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*70}")

def create_test_data():
    """Create mock annotation and beta2 files"""
    print_step(1, "Creating Mock Data")
    
    # Create test directory
    TEST_DIR.mkdir(exist_ok=True)
    print(f"Created directory: {TEST_DIR}")
    
    # Create annotation files for chromosomes 21 and 22
    print("\nCreating annotation files...")
    for chr_num in [21, 22]:
        n_snps = N_SNPS_CHR21 if chr_num == 21 else N_SNPS_CHR22
        
        # Generate mock data
        data = {
            'CHR': [chr_num] * n_snps,
            'BP': np.arange(1000000, 1000000 + n_snps * 1000, 1000),
            'rsid': [f'rs{chr_num}{i:06d}' for i in range(n_snps)],
            'ID': [f'snp_{chr_num}_{i}' for i in range(n_snps)]
        }
        
        # Add features with meaningful names
        feature_names = [
            'conservation_score',
            'recombination_rate',
            'gc_content',
            'distance_to_tss',
            'histone_mark_h3k27ac',
            'dnase_signal',
            'tfbs_count',
            'phylop_score'
        ]
        
        for feat_idx, feat_name in enumerate(feature_names[:N_FEATURES]):
            # Create features with different distributions
            if 'score' in feat_name:
                data[feat_name] = np.random.beta(2, 5, n_snps)  # Skewed towards 0
            elif 'rate' in feat_name:
                data[feat_name] = np.random.exponential(0.3, n_snps)
            elif 'content' in feat_name:
                data[feat_name] = np.random.normal(0.5, 0.15, n_snps).clip(0, 1)
            else:
                data[feat_name] = np.random.rand(n_snps)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save as gzipped file
        annot_file = f"{ANNOT_PREFIX}{chr_num}.annot.gz"
        df.to_csv(annot_file, sep='\t', index=False, compression='gzip')
        print(f"  ✓ Created: {annot_file} ({n_snps} SNPs, {N_FEATURES} features)")
    
    # Create beta2 file
    print("\nCreating beta2 file...")
    all_snps = []
    for chr_num in [21, 22]:
        annot_file = f"{ANNOT_PREFIX}{chr_num}.annot.gz"
        df = pd.read_csv(annot_file, sep='\t', compression='gzip')
        all_snps.append(df)
    
    combined = pd.concat(all_snps, ignore_index=True)
    
    # Generate synthetic beta2 values with a more complex relationship
    # This creates a relationship that a decision tree can learn
    feat_cols = combined.columns[4:]
    beta2_values = (
        3.0 * combined[feat_cols[0]] +           # conservation_score
        2.0 * combined[feat_cols[1]] -           # recombination_rate
        1.5 * combined[feat_cols[4]] +           # histone_mark_h3k27ac
        1.0 * combined[feat_cols[5]] +           # dnase_signal
        0.5 * (combined[feat_cols[0]] > 0.3).astype(float) +  # threshold effect
        0.2 * np.random.randn(len(combined))     # noise
    )
    
    # Keep only the first 4 columns and add beta2
    beta2_df = combined[['CHR', 'BP', 'rsid', 'ID']].copy()
    beta2_df['beta2'] = beta2_values
    
    # Save as text file
    beta2_df.to_csv(BETA2_FILE, sep='\t', index=False)
    print(f"  ✓ Created: {BETA2_FILE} ({len(beta2_df)} SNPs)")
    
    return True

def train_tree_model():
    """Train a decision tree model using v2d.py"""
    print_step(2, "Training Decision Tree Model")
    
    # Build command
    cmd = [
        'python', 'v2d.py',
        '--model', 'tree',
        '--max_depth', '6',
        '--min_samples_leaf', '10',
        '--annot', str(ANNOT_PREFIX),
        '--beta2', str(BETA2_FILE),
        '--out', str(OUTPUT_PREFIX),
        '--delimiter', '\t',
        '--print_mse',
        '--print_model'  # This saves the model
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nRunning v2d.py...")
    
    # Run the command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    if result.returncode == 0:
        print("✓ Model training SUCCESSFUL")
        
        # Check for saved model files
        model_files = list(TEST_DIR.glob("*.joblib"))
        if model_files:
            print(f"\nSaved model files:")
            for mf in model_files:
                print(f"  {mf}")
            return True
        else:
            print("✗ ERROR: No model files were saved!")
            return False
    else:
        print("✗ Model training FAILED")
        print(f"Return code: {result.returncode}")
        print(f"STDERR:\n{result.stderr}")
        return False

def visualize_tree():
    """Visualize the trained tree using plot_tree.py"""
    print_step(3, "Visualizing Decision Tree")
    
    # Find the saved model file (should be .even.joblib or .odd.joblib)
    model_files = list(TEST_DIR.glob("*.joblib"))
    
    if not model_files:
        print("✗ ERROR: No model files found!")
        return False
    
    # Use the first model file (typically the .even.joblib)
    model_file = model_files[0]
    print(f"Using model: {model_file}")
    
    # Use the first annotation file for feature names
    annot_file = f"{ANNOT_PREFIX}21.annot.gz"
    
    # Build command for plot_tree.py
    cmd = [
        'python', 'plot_tree.py',
        '--model_path', str(model_file),
        '--annot_csv', str(annot_file),
        '--delimiter', '\t',
        '--max_depth', '4',  # Limit visualization depth for readability
        '--out', str(TREE_PLOT_OUTPUT),
        '--dpi', '300'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nGenerating tree visualization...")
    
    # Run the command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    if result.returncode == 0:
        print("✓ Tree visualization SUCCESSFUL")
        print(f"  Output: {TREE_PLOT_OUTPUT}")
        return True
    else:
        print("✗ Tree visualization FAILED")
        print(f"Return code: {result.returncode}")
        print(f"STDERR:\n{result.stderr}")
        return False

def check_outputs():
    """Check and summarize all output files"""
    print_step(4, "Verifying Outputs")
    
    expected_files = {
        'log_file': OUTPUT_PREFIX.with_suffix('.log'),
        'mse_file': OUTPUT_PREFIX.with_suffix('.mse.txt'),
        'tree_plot': TREE_PLOT_OUTPUT
    }
    
    all_good = True
    
    for file_type, file_path in expected_files.items():
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {file_type}: {file_path} ({size} bytes)")
        else:
            print(f"✗ {file_type}: {file_path} NOT FOUND")
            all_good = False
    
    # Check for model files
    model_files = list(TEST_DIR.glob("*.joblib"))
    if model_files:
        print(f"✓ model_files: {len(model_files)} file(s)")
        for mf in model_files:
            size = mf.stat().st_size
            print(f"  - {mf} ({size} bytes)")
    else:
        print("✗ model_files: NOT FOUND")
        all_good = False
    
    # Show MSE results if available
    mse_file = expected_files['mse_file']
    if mse_file.exists():
        print(f"\nMSE Results from {mse_file}:")
        with open(mse_file) as f:
            for i, line in enumerate(f):
                print(f"  {line.strip()}")
                if i >= 5:  # Show first few lines
                    print("  ...")
                    break
    
    return all_good

def cleanup():
    """Clean up test files (optional)"""
    print_step(5, "Cleanup")
    
    print(f"\nTest files are in: {TEST_DIR.absolute()}")
    print("To view the tree visualization, open:")
    print(f"  {TREE_PLOT_OUTPUT.absolute()}")
    
    response = input("\nClean up test files? (y/n): ").lower().strip()
    if response == 'y':
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)
            print(f"✓ Cleaned up {TEST_DIR}")
    else:
        print(f"✓ Test files kept in {TEST_DIR}")

def main():
    """Main integration test execution"""
    print("="*70)
    print("V2D.PY INTEGRATION TEST")
    print("Train Tree Model → Save → Load → Visualize")
    print("="*70)
    
    success = True
    
    # Step 1: Create test data
    if not create_test_data():
        print("\n✗ INTEGRATION TEST FAILED: Could not create test data")
        sys.exit(1)
    
    # Step 2: Train tree model
    if not train_tree_model():
        print("\n✗ INTEGRATION TEST FAILED: Model training failed")
        sys.exit(1)
    
    # Step 3: Visualize tree
    if not visualize_tree():
        print("\n✗ INTEGRATION TEST FAILED: Tree visualization failed")
        success = False
    
    # Step 4: Check outputs
    if not check_outputs():
        print("\n✗ INTEGRATION TEST FAILED: Missing expected outputs")
        success = False
    
    # Summary
    print("\n" + "="*70)
    if success:
        print("✓ INTEGRATION TEST PASSED")
        print("="*70)
        print("\nAll steps completed successfully!")
        print(f"- Mock data created in {TEST_DIR}")
        print(f"- Tree model trained and saved")
        print(f"- Tree visualization generated: {TREE_PLOT_OUTPUT}")
    else:
        print("✗ INTEGRATION TEST FAILED")
        print("="*70)
        print("\nSome steps failed. Check error messages above.")
    
    # Optional cleanup
    cleanup()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

