import os
import pandas as pd
import numpy as np
from collections import defaultdict
from easydict import EasyDict
import torch
import pickle
import lmdb
import random
import math
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from tqdm.auto import tqdm

from rde.datasets.skempi_parallel import SkempiDataset_lmdb, _get_structure, load_skempi_entries
from rde.utils.protein.parsers import parse_biopython_structure
from rde.utils.transforms._base import _get_CB_positions

class SkempiPyRosettaDataset(SkempiDataset_lmdb):
    """Dataset class for PyRosetta structures with structure limit"""
    
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
        max_structures=4000  # New parameter to limit structures
    ):
        self.max_structures = max_structures
        
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
    
    def _limit_structures(self, entries):
        """Limit entries to the first max_structures structures"""
        if self.max_structures is None:
            return entries
            
        # Get unique structure IDs (pdbcode) from entries
        structure_ids = set()
        limited_entries = []
        
        for entry in entries:
            structure_id = entry['pdbcode']
            if structure_id not in structure_ids:
                if len(structure_ids) >= self.max_structures:
                    break
                structure_ids.add(structure_id)
            limited_entries.append(entry)
        
        print(f"Limited dataset to {len(structure_ids)} structures ({len(limited_entries)} entries)")
        return limited_entries
    
    def _load_entries(self, reset):
        """Override to apply structure limit after loading entries"""
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)
        
        # Apply structure limit
        self.entries_full = self._limit_structures(self.entries_full)
        
        # Save the limited entries
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(self.entries_full, f)
        
        # Split entries for cross-validation (same as parent class)
        complex_to_entries = {}
        for e in self.entries_full:
            if e['complex'] not in complex_to_entries:
                complex_to_entries[e['complex']] = []
            complex_to_entries[e['complex']].append(e)

        complex_list = sorted(complex_to_entries.keys())
        random.Random(self.split_seed).shuffle(complex_list)

        split_size = math.ceil(len(complex_list) / self.num_cvfolds)
        complex_splits = [
            complex_list[i*split_size : (i+1)*split_size] 
            for i in range(self.num_cvfolds)
        ]

        val_split = complex_splits.pop(self.cvfold_index)
        train_split = sum(complex_splits, start=[])
        if self.split == 'val':
            complexes_this = val_split
        else:
            complexes_this = train_split

        entries = []
        for cplx in complexes_this:
            #  single or multiple
            if self.is_single == 0:
                for complex_item in complex_to_entries[cplx]:
                    if complex_item['num_muts'] > 1:
                        continue
                    else:
                        entries += [complex_item]
            elif self.is_single == 1:
                for complex_item in complex_to_entries[cplx]:
                    if complex_item['num_muts'] == 1:
                        continue
                    else:
                        entries += [complex_item]
            else:
                entries += complex_to_entries[cplx]

        self.entries = entries 