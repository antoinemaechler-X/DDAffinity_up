import functools
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import pickle
import random
import math
from collections import defaultdict
from tqdm.auto import tqdm

from rde.utils.misc import BlackHole
from rde.utils.data_skempi_mpnn import PaddingCollate
from rde.utils.transforms import get_transform
from rde.datasets.skempi_parallel import load_category_entries, SkempiDataset_lmdb


class SequenceGroupedSkempiDataset(SkempiDataset_lmdb):
    """
    Dataset class that groups complexes by sequence identity and splits on groups
    to prevent data leakage from similar structures.
    Works exactly like SkempiDataset_lmdb but uses groups instead of individual complexes.
    """
    
    def __init__(
        self,
        csv_path,
        pdb_wt_dir,
        pdb_mt_dir,
        cache_dir,
        split='train',
        transform=None,
        num_cvfolds=1,
        cvfold_index=0,
        split_seed=42,
        reset=False,
        is_single=2,
        sequence_identity_threshold=0.9,  # 90% sequence identity threshold
        split_csv='data/complex_sequences_grouped.csv',  # NEW ARGUMENT
    ):
        self.sequence_identity_threshold = sequence_identity_threshold
        self.split_csv = split_csv
        
        # Store parameters for later use
        self._csv_path = csv_path
        self._pdb_wt_dir = pdb_wt_dir
        self._pdb_mt_dir = pdb_mt_dir
        self._cache_dir = cache_dir
        self._split = split
        self._transform = transform
        self._num_cvfolds = num_cvfolds
        self._cvfold_index = cvfold_index
        self._split_seed = split_seed
        self._reset = reset
        self._is_single = is_single
        
        # Load entries and create sequence-based splits
        self._load_entries(self._reset)
        
        # Initialize parent class with the filtered entries
        # Don't pass reset=True to avoid triggering structure preprocessing
        super().__init__(
            csv_path=csv_path,
            pdb_wt_dir=pdb_wt_dir,
            pdb_mt_dir=pdb_mt_dir,
            cache_dir=cache_dir,
            split=split,
            transform=transform,
            num_cvfolds=num_cvfolds,
            cvfold_index=cvfold_index,
            split_seed=split_seed,
            reset=False,  # Always False to avoid NumPy compatibility issues
            is_single=is_single
        )
        
        # Override the entries with our sequence-grouped entries
        self.entries = self._filtered_entries
    
    def _load_entries(self, reset=False):
        """Load entries and create sequence-based splits using proper CV logic"""
        # Load all entries
        self.entries_full = load_category_entries(self._csv_path, self._pdb_wt_dir, self._pdb_mt_dir)
        
        # Group complexes by sequence similarity
        complex_groups = self._group_complexes_by_sequence()
        
        # Apply CV splitting logic (exactly like SkempiDataset_lmdb but with groups)
        group_to_entries = {}
        for e in self.entries_full:
            # Find which group this complex belongs to
            # Handle numeric prefix in complex names (e.g., "0_1CSE" -> "1CSE")
            complex_name = e['complex']
            base_complex_name = self._extract_base_complex_name(complex_name)
            
            complex_group = None
            for group in complex_groups:
                if base_complex_name in group:
                    complex_group = tuple(group)  # Make it hashable
                    break
            
            if complex_group is None:
                # Complex not in any group, skip it
                continue
                
            if complex_group not in group_to_entries:
                group_to_entries[complex_group] = []
            group_to_entries[complex_group].append(e)

        # Sort groups and apply CV splitting (exactly like original)
        group_list = sorted(group_to_entries.keys())
        random.Random(self._split_seed).shuffle(group_list)

        split_size = math.ceil(len(group_list) / self._num_cvfolds)
        group_splits = [
            group_list[i*split_size : (i+1)*split_size] 
            for i in range(self._num_cvfolds)
        ]

        val_split = group_splits.pop(self._cvfold_index)
        train_split = sum(group_splits, start=[])
        
        if self._split == 'val':
            groups_this = val_split
        else:
            groups_this = train_split

        # Get all complexes in selected groups
        selected_complexes = set()
        for group in groups_this:
            selected_complexes.update(group)
        
        # Filter entries for selected complexes
        entries = []
        for entry in self.entries_full:
            base_complex_name = self._extract_base_complex_name(entry['complex'])
            if base_complex_name in selected_complexes:
                # Apply single/multiple filtering
                if self._is_single == 0:  # single mutations only
                    if entry['num_muts'] == 1:
                        entries.append(entry)
                elif self._is_single == 1:  # multiple mutations only
                    if entry['num_muts'] > 1:
                        entries.append(entry)
                else:  # all mutations
                    entries.append(entry)
        
        self._filtered_entries = entries
        
        # Log split information
        print(f"\nSequence-grouped CV split summary:")
        print(f"Total complexes: {len(set(e['complex'] for e in self.entries_full))}")
        print(f"Total groups: {len(complex_groups)}")
        print(f"Fold {self._cvfold_index}: {self._split} split has {len(entries)} entries from {len(selected_complexes)} complexes")
        
        # Add detailed CV information
        print(f"\n=== DETAILED CV SPLIT INFO ===")
        print(f"Total groups available: {len(group_list)}")
        print(f"Split size (groups per fold): {split_size}")
        print(f"Number of CV folds: {self._num_cvfolds}")
        print(f"Current fold index: {self._cvfold_index}")
        print(f"Current split: {self._split}")
        
        # Show group distribution for this fold
        print(f"\nGroup distribution for fold {self._cvfold_index}:")
        print(f"  Validation groups: {len(val_split)} groups")
        print(f"  Training groups: {len(train_split)} groups")
        
        # Show some sample groups for verification
        if val_split:
            print(f"  Sample validation groups: {val_split[:3]}")  # Show first 3 groups
        if train_split:
            print(f"  Sample training groups: {train_split[:3]}")  # Show first 3 groups
        
        # Verify no overlap between train and val groups
        train_groups_set = set(train_split)
        val_groups_set = set(val_split)
        overlap = train_groups_set.intersection(val_groups_set)
        if overlap:
            print(f"  ⚠️  WARNING: Group overlap detected: {overlap}")
        else:
            print(f"  ✅ No group overlap - data leakage prevention working")
        
        # Show complex distribution
        print(f"\nComplex distribution:")
        total_complexes = len(set(e['complex'] for e in self.entries_full))
        print(f"  Total complexes in dataset: {total_complexes}")
        print(f"  Complexes in selected groups: {len(selected_complexes)}")
        
        # Calculate complexes not in any group using base names
        all_base_complexes = set()
        for e in self.entries_full:
            base_name = self._extract_base_complex_name(e['complex'])
            all_base_complexes.add(base_name)
        
        # Get all complexes that are in ANY group (not just selected ones)
        all_grouped_complexes = set()
        for group in complex_groups:
            all_grouped_complexes.update(group)
        
        complexes_not_in_groups = len(all_base_complexes) - len(all_grouped_complexes)
        print(f"  Complexes not in any group: {complexes_not_in_groups} (intentionally excluded from sequence grouping)")
        
        # Add explanation for the excluded complexes
        if complexes_not_in_groups > 0:
            print(f"  Note: {complexes_not_in_groups} complexes were excluded from grouping due to sequence similarity computation issues")
        
        # Show mutation distribution
        single_muts = sum(1 for e in entries if e['num_muts'] == 1)
        multi_muts = sum(1 for e in entries if e['num_muts'] > 1)
        print(f"\nMutation distribution in {self._split} split:")
        print(f"  Single mutations: {single_muts}")
        print(f"  Multiple mutations: {multi_muts}")
        print(f"  Total mutations: {len(entries)}")
        
        print(f"=== END CV SPLIT INFO ===\n")
    
    def _extract_base_complex_name(self, complex_name):
        """Extract base complex name by removing numeric prefix (e.g., '0_1CSE' -> '1CSE')"""
        if '_' in complex_name:
            parts = complex_name.split('_', 1)
            if len(parts) == 2 and parts[0].isdigit():
                return parts[1]
        return complex_name
    
    def _group_complexes_by_sequence(self):
        """Group complexes by sequence identity using the pre-computed CSV file"""
        grouped_csv_path = self.split_csv
        if not os.path.exists(grouped_csv_path):
            raise FileNotFoundError(f"Grouped CSV file not found: {grouped_csv_path}. Please run compute_sequence_clusters.py first.")
        print(f"Loading pre-computed sequence groups from CSV: {grouped_csv_path}")
        df = pd.read_csv(grouped_csv_path)
        
        # Get unique complexes and their group IDs
        complex_groups = {}
        for _, row in df.iterrows():
            complex_name = row['complex_name']
            group_id = row['complex_group_id']
            if complex_name not in complex_groups:
                complex_groups[complex_name] = group_id
        
        # Convert to the expected format: list of lists of complex names
        group_to_complexes = defaultdict(list)
        for complex_name, group_id in complex_groups.items():
            group_to_complexes[group_id].append(complex_name)
        
        groups = list(group_to_complexes.values())
        
        print(f"Loaded {len(groups)} groups from CSV")
        print(f"Total complexes: {len(complex_groups)}")
        
        return groups
    
    def _load_structures(self, reset):
        """Override parent method to avoid triggering structure preprocessing with reset=True"""
        # Always use existing cache to avoid NumPy compatibility issues
        if os.path.exists(self.structures_cache):
            print(f"[INFO] Using existing cache at {self.structures_cache}")
            return None
        else:
            print(f"[INFO] Cache not found, preprocessing structures...")
            return super()._preprocess_structures(False)  # Always use False to avoid issues


class SequenceGroupedSkempiDatasetManager:
    """
    Dataset manager that uses sequence-based grouping for cross-validation.
    Works exactly like SkempiDatasetManager but uses groups instead of individual complexes.
    """
    
    def __init__(self, config, num_cvfolds=10, num_workers=4, logger=BlackHole(), reset_split=False, split_csv='data/complex_sequences_grouped.csv'):
        super().__init__()
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.train_loader = []
        self.val_loaders = []
        self.logger = logger
        self.num_workers = num_workers
        self.reset_split = reset_split
        self.split_csv = split_csv
        
        print(f"\n{'='*60}")
        print(f"SEQUENCE-GROUPED CROSS-VALIDATION SETUP")
        print(f"{'='*60}")
        print(f"Number of CV folds: {num_cvfolds}")
        print(f"Random seed: {config.train.seed}")
        print(f"Using groups from: {self.split_csv}")
        
        # Create loaders for each fold (exactly like original)
        for fold in range(num_cvfolds):
            print(f"\n--- Initializing Fold {fold} ---")
            train_loader, val_loader = self.init_loaders(fold)
            self.train_loader.append(train_loader)
            self.val_loaders.append(val_loader)
        
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SETUP COMPLETE")
        print(f"{'='*60}")
        print(f"Created {len(self.train_loader)} train loaders")
        print(f"Created {len(self.val_loaders)} validation loaders")
        print(f"Each fold will have different validation groups to prevent data leakage")
        print(f"{'='*60}\n")
    
    def init_loaders(self, fold):
        config = self.config
        
        # Create dataset with sequence-based grouping (exactly like original)
        dataset_ = functools.partial(
            SequenceGroupedSkempiDataset,
            csv_path=config.data.csv_path,
            pdb_wt_dir=config.data.pdb_wt_dir,
            pdb_mt_dir=config.data.pdb_mt_dir,
            cache_dir=config.data.cache_dir,
            num_cvfolds=self.num_cvfolds,
            cvfold_index=fold,
            split_seed=config.train.seed,  # Use same seed for all folds (like original)
            split_csv=self.split_csv  # Pass the split file
        )
        
        train_dataset = dataset_(
            split='train',
            transform=get_transform(config.data.train.transform)
        )
        val_dataset = dataset_(
            split='val',
            transform=get_transform(config.data.val.transform)
        )
        
        # Verify no data leakage (should be handled by grouping, but double-check)
        train_complexes = set([e['complex'] for e in train_dataset.entries])
        val_complexes = set([e['complex'] for e in val_dataset.entries])
        leakage = train_complexes.intersection(val_complexes)
        assert len(leakage) == 0, f'Data leakage detected in fold {fold}: {leakage}'
        
        # Create data loaders with PaddingCollate like the original
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            collate_fn=PaddingCollate(config.data.val.transform[1].patch_size),
            shuffle=True,
            num_workers=self.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            collate_fn=PaddingCollate(config.data.val.transform[1].patch_size),
            num_workers=self.num_workers
        )
        
        self.logger.info(f'Fold {fold}: Train {len(train_dataset)}, Val {len(val_dataset)}')
        return train_loader, val_loader
    
    def get_train_loader(self, fold):
        return self.train_loader[fold]
    
    def get_val_loader(self, fold):
        return self.val_loaders[fold]


# Convenience function for backward compatibility
def get_sequence_grouped_skempi_dataset(config, num_cvfolds=10, num_workers=4, logger=BlackHole(), reset_split=False):
    """Get a sequence-grouped dataset manager"""
    return SequenceGroupedSkempiDatasetManager(
        config, 
        num_cvfolds=num_cvfolds, 
        num_workers=num_workers, 
        logger=logger,
        reset_split=reset_split
    ) 