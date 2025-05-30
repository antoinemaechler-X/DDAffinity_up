import functools
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import pickle
import random
from sklearn.model_selection import train_test_split

# Patch for numpy deprecations
import numpy as np
np.bool = bool  # This will make np.bool available for BioPython
np.int = int    # This will make np.int available for BioPython

from rde.utils.misc import inf_iterator, BlackHole
from rde.utils.data_skempi_mpnn import PaddingCollate
from rde.utils.transforms import get_transform
from rde.datasets.skempi_parallel_spr import SkempiDatasetSimple  # Import our new class

class SkempiDatasetManagerSimple(object):
    def __init__(self, config, num_workers=4, logger=BlackHole(), test_size=0.1, random_state=42, reset=False):
        super().__init__()
        self.config = config
        self.logger = logger
        self.num_workers = num_workers
        self.test_size = test_size
        self.random_state = random_state
        self.reset = reset
        
        # Initialize loaders
        self.train_loader, self.test_loader = self.init_loaders()

    def init_loaders(self):
        config = self.config
        
        # Log dataset paths
        self.logger.info(f"Loading dataset from:")
        self.logger.info(f"  CSV path: {config.data.csv_path}")
        self.logger.info(f"  WT PDB dir: {config.data.pdb_wt_dir}")
        self.logger.info(f"  MT PDB dir: {config.data.pdb_mt_dir}")
        self.logger.info(f"  Cache dir: {config.data.cache_dir}")
        
        # Check if files exist
        if not os.path.exists(config.data.csv_path):
            raise FileNotFoundError(f"CSV file not found: {config.data.csv_path}")
        if not os.path.exists(config.data.pdb_wt_dir):
            raise FileNotFoundError(f"WT PDB directory not found: {config.data.pdb_wt_dir}")
        if not os.path.exists(config.data.pdb_mt_dir):
            raise FileNotFoundError(f"MT PDB directory not found: {config.data.pdb_mt_dir}")
        if not os.path.exists(config.data.cache_dir):
            self.logger.info(f"Cache directory not found, will be created: {config.data.cache_dir}")
            os.makedirs(config.data.cache_dir, exist_ok=True)
        
        # Load all data using our new dataset class
        self.logger.info("Loading full dataset...")
        full_dataset = SkempiDatasetSimple(
            csv_path=config.data.csv_path,
            pdb_wt_dir=config.data.pdb_wt_dir,
            pdb_mt_dir=config.data.pdb_mt_dir,
            cache_dir=config.data.cache_dir,
            split='train',  # Use 'train' split to get all data
            transform=get_transform(config.data.train.transform),
            num_cvfolds=1,  # Use single fold to get all data
            cvfold_index=0,  # Use first fold
            split_seed=config.train.seed,
            is_single=config.data.is_single,
            reset=self.reset
        )
        
        # Log dataset info
        self.logger.info(f"Full dataset loaded with {len(full_dataset)} entries")
        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty! Check if the CSV file and PDB directories contain the expected data.")
        
        # Get unique complexes and split them
        unique_complexes = list(set(e['complex'] for e in full_dataset.entries))
        self.logger.info(f"Found {len(unique_complexes)} unique complexes")
        
        # Split complexes into train/test
        train_complexes, test_complexes = train_test_split(
            unique_complexes, 
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Filter entries based on complex split
        train_entries = [e for e in full_dataset.entries if e['complex'] in train_complexes]
        test_entries = [e for e in full_dataset.entries if e['complex'] in test_complexes]
        
        self.logger.info(f"Split into {len(train_entries)} training entries and {len(test_entries)} test entries")
        
        # Create train dataset
        train_dataset = SkempiDatasetSimple(
            csv_path=config.data.csv_path,
            pdb_wt_dir=config.data.pdb_wt_dir,
            pdb_mt_dir=config.data.pdb_mt_dir,
            cache_dir=config.data.cache_dir,
            split='train',  # Use 'train' split
            transform=get_transform(config.data.train.transform),
            num_cvfolds=1,
            cvfold_index=0,
            split_seed=config.train.seed,
            is_single=config.data.is_single
        )
        train_dataset.entries = train_entries  # Override entries with our split
        
        # Create test dataset
        test_dataset = SkempiDatasetSimple(
            csv_path=config.data.csv_path,
            pdb_wt_dir=config.data.pdb_wt_dir,
            pdb_mt_dir=config.data.pdb_mt_dir,
            cache_dir=config.data.cache_dir,
            split='train',  # Use 'train' split
            transform=get_transform(config.data.val.transform),
            num_cvfolds=1,
            cvfold_index=0,
            split_seed=config.train.seed,
            is_single=config.data.is_single
        )
        test_dataset.entries = test_entries  # Override entries with our split
        
        # Verify no data leakage
        train_cplx = set([e['complex'] for e in train_dataset.entries])
        test_cplx = set([e['complex'] for e in test_dataset.entries])
        leakage = train_cplx.intersection(test_cplx)
        assert len(leakage) == 0, f'data leakage {leakage}'
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            collate_fn=PaddingCollate(config.data.val.transform[1].patch_size),
            shuffle=True,
            num_workers=self.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            collate_fn=PaddingCollate(config.data.val.transform[1].patch_size),
            num_workers=self.num_workers
        )
        
        self.logger.info(f'Train: {len(train_dataset)} samples, {len(train_complexes)} complexes')
        self.logger.info(f'Test: {len(test_dataset)} samples, {len(test_complexes)} complexes')
        
        return train_loader, test_loader

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader 