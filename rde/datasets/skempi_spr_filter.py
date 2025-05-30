import os
import pandas as pd
import numpy as np
from collections import defaultdict
from easydict import EasyDict
import torch
import pickle
import lmdb
import random
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from tqdm.auto import tqdm

from rde.datasets.skempi_parallel import SkempiDataset_lmdb, _get_structure
from rde.utils.protein.parsers import parse_biopython_structure
from rde.utils.transforms._base import _get_CB_positions

class SkempiSPRFilterDataset(SkempiDataset_lmdb):
    """Dataset class that only includes SPR mutations from SKEMPI2"""
    
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
        reset=False
    ):
        # Load both datasets
        full_df = pd.read_csv('data/SKEMPI2/full_SKEMPI2.csv', sep=';')
        main_df = pd.read_csv(csv_path)
        
        # First: Filter for SPR mutations
        spr_mutations = set()
        for _, row in full_df.iterrows():
            if row['Method'] == 'SPR':
                # Extract PDB code from complex name (e.g. "1CSE_E_I" -> "1CSE")
                pdbcode = row['#Pdb'].split('_')[0]
                spr_mutations.add((pdbcode, row['Mutation(s)_cleaned']))
        
        # Filter main dataset for SPR mutations
        spr_entries = []
        for _, row in main_df.iterrows():
            # Extract PDB code from complex name (e.g. "0_1CSE" -> "1CSE")
            pdbcode = row['#Pdb'].split('_')[-1]
            if (pdbcode, row['Mutation(s)_cleaned']) in spr_mutations:
                spr_entries.append(row.to_dict())
        
        # Get unique complexes from SPR entries
        spr_complexes = list(set(row['#Pdb'].split('_')[-1] for row in spr_entries))
        random.seed(split_seed)
        random.shuffle(spr_complexes)
        
        # Split complexes into folds
        fold_size = len(spr_complexes) // num_cvfolds
        start_idx = cvfold_index * fold_size
        end_idx = start_idx + fold_size if cvfold_index < num_cvfolds - 1 else len(spr_complexes)
        
        # Get complexes for this fold
        if split == 'train':
            fold_complexes = spr_complexes[:start_idx] + spr_complexes[end_idx:]
        else:  # val
            fold_complexes = spr_complexes[start_idx:end_idx]
        
        # Filter entries for this fold's complexes
        fold_entries = [entry for entry in spr_entries 
                       if entry['#Pdb'].split('_')[-1] in fold_complexes]
        
        # Create filtered CSV with only SPR mutations for this fold
        filtered_df = pd.DataFrame(fold_entries)
        filtered_csv_path = csv_path.replace('.csv', '_spr_filtered.csv')
        filtered_df.to_csv(filtered_csv_path, index=False)
        
        # Initialize parent class with filtered CSV
        super().__init__(
            csv_path=filtered_csv_path,
            pdb_wt_dir=pdb_wt_dir,
            pdb_mt_dir=pdb_mt_dir,
            cache_dir=cache_dir,
            split=split,
            transform=transform,
            num_cvfolds=1,  # No need for CV in parent class since we did it here
            cvfold_index=0,
            split_seed=split_seed,
            reset=reset
        )
        
        # Print summary
        print(f"\nSKEMPI2 Dataset Summary:")
        print(f"Total SPR mutations: {len(spr_entries)}")
        print(f"Unique complexes with SPR mutations: {len(spr_complexes)}")
        print(f"Complexes per fold: {fold_size}")
        print(f"Total entries in {split} split: {len(self.entries)}")
        print(f"Unique complexes in {split} split: {len(fold_complexes)}")
        
        # Clean up temporary file
        os.remove(filtered_csv_path)
        
        # Rebuild cache if needed
        if reset or not os.path.exists(self.structures_cache):
            self._preprocess_structures(reset)
    
    def _get_pdbcode(self, complex_name):
        """Extract PDB code from complex name, handling both formats"""
        if '_' in complex_name:
            return complex_name.split('_')[-1]  # Get last part (e.g. "0_1CSE" -> "1CSE")
        return complex_name 