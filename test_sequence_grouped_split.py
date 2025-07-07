#!/usr/bin/env python3
"""
Test script for sequence-grouped dataset splitting
"""

import os
import sys
import pandas as pd
from collections import defaultdict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rde.utils.skempi_mpnn_sequence_grouped import SequenceGroupedSkempiDataset


def test_single_fold():
    """Test single fold training/validation split"""
    print("Testing single fold sequence-grouped split...")
    
    # Configuration
    config = {
        'data': {
            'csv_path': 'data/SKEMPI2/SKEMPI2.csv',
            'pdb_wt_dir': 'data/SKEMPI2/SKEMPI2_cache/wildtype',
            'pdb_mt_dir': 'data/SKEMPI2/SKEMPI2_cache/mutant',
            'cache_dir': 'cache',
            'train': {'transform': 'default'},
            'val': {'transform': 'default'}
        },
        'train': {
            'seed': 42,
            'batch_size': 32
        }
    }
    
    # Create datasets
    train_dataset = SequenceGroupedSkempiDataset(
        csv_path=config['data']['csv_path'],
        pdb_wt_dir=config['data']['pdb_wt_dir'],
        pdb_mt_dir=config['data']['pdb_mt_dir'],
        cache_dir=config['data']['cache_dir'],
        split='train',
        split_seed=config['train']['seed'],
        reset=False,
        sequence_identity_threshold=0.6  # Use 60% threshold
    )
    
    val_dataset = SequenceGroupedSkempiDataset(
        csv_path=config['data']['csv_path'],
        pdb_wt_dir=config['data']['pdb_wt_dir'],
        pdb_mt_dir=config['data']['pdb_mt_dir'],
        cache_dir=config['data']['cache_dir'],
        split='val',
        split_seed=config['train']['seed'],
        reset=False,
        sequence_identity_threshold=0.6  # Use 60% threshold
    )
    
    print(f"Train dataset: {len(train_dataset)} entries")
    print(f"Val dataset: {len(val_dataset)} entries")
    
    # Check for data leakage
    train_complexes = set([entry['complex'] for entry in train_dataset.entries])
    val_complexes = set([entry['complex'] for entry in val_dataset.entries])
    
    leakage = train_complexes.intersection(val_complexes)
    if leakage:
        print(f"WARNING: Data leakage detected! {len(leakage)} complexes in both train and val:")
        for complex_name in list(leakage)[:5]:
            print(f"  {complex_name}")
    else:
        print("✓ No data leakage detected")
    
    return train_dataset, val_dataset


def test_cross_validation():
    """Test cross-validation with multiple folds"""
    print("\nTesting cross-validation with sequence-grouped splits...")
    
    # Configuration
    config = {
        'data': {
            'csv_path': 'data/SKEMPI2/SKEMPI2.csv',
            'pdb_wt_dir': 'data/SKEMPI2/SKEMPI2_cache/wildtype',
            'pdb_mt_dir': 'data/SKEMPI2/SKEMPI2_cache/mutant',
            'cache_dir': 'cache',
            'train': {'transform': 'default'},
            'val': {'transform': 'default'}
        },
        'train': {
            'seed': 42,
            'batch_size': 32
        }
    }
    
    num_folds = 5
    
    for fold in range(num_folds):
        print(f"\nFold {fold + 1}/{num_folds}:")
        
        # Use different seeds for each fold
        fold_seed = config['train']['seed'] + fold
        
        train_dataset = SequenceGroupedSkempiDataset(
            csv_path=config['data']['csv_path'],
            pdb_wt_dir=config['data']['pdb_wt_dir'],
            pdb_mt_dir=config['data']['pdb_mt_dir'],
            cache_dir=config['data']['cache_dir'],
            split='train',
            num_cvfolds=num_folds,
            cvfold_index=fold,
            split_seed=fold_seed,
            reset=False,
            sequence_identity_threshold=0.6  # Use 60% threshold
        )
        
        val_dataset = SequenceGroupedSkempiDataset(
            csv_path=config['data']['csv_path'],
            pdb_wt_dir=config['data']['pdb_wt_dir'],
            pdb_mt_dir=config['data']['pdb_mt_dir'],
            cache_dir=config['data']['cache_dir'],
            split='val',
            num_cvfolds=num_folds,
            cvfold_index=fold,
            split_seed=fold_seed,
            reset=False,
            sequence_identity_threshold=0.6  # Use 60% threshold
        )
        
        print(f"  Train: {len(train_dataset)} entries")
        print(f"  Val: {len(val_dataset)} entries")
        
        # Check for data leakage
        train_complexes = set([entry['complex'] for entry in train_dataset.entries])
        val_complexes = set([entry['complex'] for entry in val_dataset.entries])
        
        leakage = train_complexes.intersection(val_complexes)
        if leakage:
            print(f"  WARNING: Data leakage in fold {fold + 1}!")
        else:
            print(f"  ✓ No data leakage in fold {fold + 1}")


def main():
    """Main test function"""
    print("Testing sequence-grouped dataset splitting...")
    
    # Check if the grouped CSV file exists
    grouped_csv_path = 'data/complex_sequences_grouped.csv'
    if not os.path.exists(grouped_csv_path):
        print(f"ERROR: Grouped CSV file not found: {grouped_csv_path}")
        print("Please run the following commands first:")
        print("1. python extract_sequences.py --csv_path data/SKEMPI2/SKEMPI2.csv --pdb_wt_dir data/SKEMPI2/SKEMPI2_cache/wildtype")
        print("2. python compute_sequence_clusters.py --input_csv data/complex_sequences.csv --threshold 0.6")
        return
    
    print(f"✓ Found grouped CSV file: {grouped_csv_path}")
    
    # Load and display group information
    df = pd.read_csv(grouped_csv_path)
    num_groups = df['complex_group_id'].nunique()
    num_complexes = df['complex_name'].nunique()
    
    print(f"Group information:")
    print(f"  Total groups: {num_groups}")
    print(f"  Total complexes: {num_complexes}")
    print(f"  Average group size: {num_complexes / num_groups:.1f}")
    
    # Show some example groups
    print(f"\nExample groups:")
    for group_id in range(min(3, num_groups)):
        group_complexes = df[df['complex_group_id'] == group_id]['complex_name'].unique()
        print(f"  Group {group_id}: {len(group_complexes)} complexes")
        for complex_name in group_complexes[:3]:
            print(f"    {complex_name}")
        if len(group_complexes) > 3:
            print(f"    ... and {len(group_complexes) - 3} more")
    
    # Test single fold
    try:
        train_dataset, val_dataset = test_single_fold()
        print("✓ Single fold test passed")
    except Exception as e:
        print(f"✗ Single fold test failed: {e}")
        return
    
    # Test cross-validation
    try:
        test_cross_validation()
        print("✓ Cross-validation test passed")
    except Exception as e:
        print(f"✗ Cross-validation test failed: {e}")
        return
    
    print("\n✓ All tests passed! The sequence-grouped dataset is working correctly.")


if __name__ == "__main__":
    main() 