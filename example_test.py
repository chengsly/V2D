#!/usr/bin/env python3
"""
Simple example demonstrating v2d.py usage with minimal mock data
This can be run independently to verify the installation
"""

import numpy as np
import pandas as pd
import subprocess
import os
from pathlib import Path

# Create a minimal test
print("Creating minimal test data...")

# Create test directory
Path("minimal_test").mkdir(exist_ok=True)

# Create tiny annotation files (10 SNPs per chromosome)
for chr_num in [21, 22]:
    data = {
        'CHR': [chr_num] * 10,
        'BP': np.arange(1000000, 1000010) * 1000,
        'rsid': [f'rs{chr_num}{i:03d}' for i in range(10)],
        'ID': [f'id_{chr_num}_{i}' for i in range(10)],
        'feat1': np.random.rand(10),
        'feat2': np.random.rand(10),
        'feat3': np.random.rand(10),
    }
    df = pd.DataFrame(data)
    filename = f"minimal_test/annot.{chr_num}.annot.gz"
    df.to_csv(filename, sep='\t', index=False, compression='gzip')
    print(f"  Created {filename}")

# Create beta2 file
all_annots = []
for chr_num in [21, 22]:
    df = pd.read_csv(f"minimal_test/annot.{chr_num}.annot.gz", sep='\t', compression='gzip')
    all_annots.append(df)

combined = pd.concat(all_annots)
beta2_df = combined[['CHR', 'BP', 'rsid', 'ID']].copy()
# Create synthetic target: simple linear relationship
beta2_df['beta2'] = 2*combined['feat1'] + combined['feat2'] + 0.1*np.random.randn(len(combined))
beta2_df.to_csv("minimal_test/beta2.txt", sep='\t', index=False)
print(f"  Created minimal_test/beta2.txt")

# Run a simple linear model test
print("\nRunning v2d.py with linear model...")
cmd = [
    'python', 'v2d.py',
    '--model', 'linear',
    '--annot', 'minimal_test/annot.',
    '--beta2', 'minimal_test/beta2.txt',
    '--out', 'minimal_test/output',
    '--print_mse'
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✓ SUCCESS! v2d.py ran successfully")
    print("\nOutput files:")
    for f in Path("minimal_test").glob("output*"):
        print(f"  {f}")
    
    # Show MSE results
    if Path("minimal_test/output.mse.txt").exists():
        print("\nMSE Results:")
        with open("minimal_test/output.mse.txt") as f:
            for line in f:
                print(f"  {line.strip()}")
else:
    print("✗ FAILED")
    print(f"Return code: {result.returncode}")
    print(f"STDERR:\n{result.stderr}")

print("\nTo clean up: rm -rf minimal_test/")

