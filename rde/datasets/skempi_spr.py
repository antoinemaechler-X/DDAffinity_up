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

class SkempiSPRDataset(SkempiDataset_lmdb):
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
        # Initialize parent class with same parameters
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
            reset=reset
        )
        
        # Load full SKEMPI2 dataset to get method information
        print("Loading full SKEMPI2 dataset...")
        self.full_skempi_df = pd.read_csv('data/SKEMPI2/full_SKEMPI2.csv', sep=';')
        print(f"Total rows in full dataset: {len(self.full_skempi_df)}")
        print(f"SPR rows in full dataset: {len(self.full_skempi_df[self.full_skempi_df['Method'] == 'SPR'])}")
        
        # Filter entries to only include SPR mutations
        print("\nFiltering entries for SPR mutations...")
        self.entries = self._filter_spr_mutations(self.entries)
        
        # Rebuild cache if needed
        if reset or not os.path.exists(self.structures_cache):
            self._preprocess_structures(reset)
    
    def _extract_pdbcode(self, complex_name):
        """Extract PDB code from complex name, handling different formats"""
        # Try to split by underscore first
        parts = complex_name.split('_')
        if len(parts) > 1:
            # Format is "i_PDBCODE" or similar
            return parts[1]
        else:
            # Format is just "PDBCODE"
            return complex_name
    
    def _filter_spr_mutations(self, entries):
        """Filter entries to only include SPR mutations"""
        # Create mapping from complex+mutstr to method
        method_map = {}
        spr_count = 0
        for _, row in self.full_skempi_df.iterrows():
            if row['Method'] == 'SPR':
                spr_count += 1
                # Get complex name and mutation string
                # In full_SKEMPI2.csv, #Pdb is already in format "PDBCODE_CHAIN1_CHAIN2"
                complex_name = row['#Pdb']
                mutstr = row['Mutation(s)_cleaned']
                # Create key in format "PDBCODE_CHAIN1_CHAIN2_MUTSTR"
                key = f"{complex_name}_{mutstr}"
                method_map[key] = 'SPR'
        print(f"Found {spr_count} SPR mutations in full dataset")
        print(f"Created mapping for {len(method_map)} unique SPR mutations")
        
        # Filter entries
        filtered_entries = []
        not_found = []
        for entry in entries:
            # Get complex name and mutation string from entry
            # entry['complex'] should already be in format "PDBCODE_CHAIN1_CHAIN2"
            complex_name = entry['complex']
            mutstr = entry['mutstr']
            # Create key in same format as method_map
            key = f"{complex_name}_{mutstr}"
            
            if key in method_map:
                filtered_entries.append(entry)
            else:
                not_found.append(key)
                # Print detailed debug info for first few not found entries
                if len(not_found) <= 5:
                    print(f"\nDebug - Entry not found:")
                    print(f"  Complex name from entry: {complex_name}")
                    print(f"  Mutation string: {mutstr}")
                    print(f"  Full key: {key}")
                    print(f"  Available similar keys in method_map:")
                    # Show similar keys that might help debug
                    similar_keys = [k for k in method_map.keys() if complex_name in k or mutstr in k][:3]
                    for k in similar_keys:
                        print(f"    {k}")
        
        print(f"\nFiltering results:")
        print(f"Original entries: {len(entries)}")
        print(f"Filtered entries: {len(filtered_entries)}")
        print(f"Entries not found in SPR dataset: {len(not_found)}")
        if len(not_found) > 0:
            print("\nFirst 5 entries not found:")
            for key in not_found[:5]:
                print(f"  {key}")
        return filtered_entries 