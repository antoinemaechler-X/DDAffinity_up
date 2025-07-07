# -*- coding: utf-8 -*-
"""
Override module for SPR‐only runs: patches _process_structure
in the existing skempi_parallel to catch parse errors and return None.
"""

from rde.datasets import skempi_parallel
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from collections import defaultdict
import random

def _process_structure(pdb_wt_path, pdb_mt_path, pdbcode) -> skempi_parallel.Optional[dict]:
    try:
        structures = defaultdict(dict)
        parser = PDBParser(QUIET=True)
        model = parser.get_structure(None, pdb_wt_path)[0]
        # keep only non‐blank chains
        chains = [c for c in Selection.unfold_entities(model, 'C') if c.id.strip()]
        random.shuffle(chains)
        structures.update(skempi_parallel._get_structure(pdb_wt_path, chains, pdbcode, "wt"))
        structures.update(skempi_parallel._get_structure(pdb_mt_path, chains, pdbcode, "mt"))
        return structures
    except Exception as e:
        print(f"[SPR PATCH] skipping {pdbcode} due to parse error: {e}")
        return None

# Monkey‐patch the original module
skempi_parallel._process_structure = _process_structure
