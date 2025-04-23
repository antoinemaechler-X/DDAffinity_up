#!/usr/bin/env python3
"""
compare_folders_rmsd.py

For each matching PDB file pair in two folders,
compute the overall RMSD (root-mean-square deviation) based on all matching
atoms (by chain, residue number, insertion code, and atom name).
Then, output the mean RMSD over all molecule pairs.

Usage:
  python compute_rmsd.py <folder1> <folder2>
  
Example:
  python compute_rmsd.py data/SKEMPI2/M1340_cache/wildtype_evoef1 data/SKEMPI2/M1340_cache/wildtype_evoef1
"""

import os
import math
import sys

def parse_pdb_coords(path):
    """
    Parse a PDB file and return a dictionary mapping an atom identifier to
    its 3D coordinates.
    
    Atom identifier: (chain_id, res_seq, i_code, atom_name).
    Only lines starting with 'ATOM  ' or 'HETATM' are considered.
    """
    coords = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("ATOM  ") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    chain_id  = line[21].strip()
                    res_seq   = line[22:26].strip()
                    i_code    = line[26].strip()
                    key = (chain_id, res_seq, i_code, atom_name)
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue  # skip malformed coordinates
                    coords[key] = (x, y, z)
    except IOError as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
    return coords

def compute_rmsd_from_coords(coords1, coords2):
    """
    Compute the overall RMSD between two sets of coordinates.
    
    Only atoms common to both structures (intersection of keys) are used.
    Returns None if no common atoms are found.
    """
    common_keys = set(coords1.keys()) & set(coords2.keys())
    if not common_keys:
        return None
    sum_sq = 0.0
    for key in common_keys:
        x1, y1, z1 = coords1[key]
        x2, y2, z2 = coords2[key]
        diff_sq = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
        sum_sq += diff_sq
    rmsd = math.sqrt(sum_sq / len(common_keys))
    return rmsd

def compare_folder_pair(folder1, folder2):
    """
    For each pdb file in folder1 that also exists in folder2 (matched by filename),
    compute the RMSD based on matching 3D coordinates. Then return the average RMSD
    over all file pairs.
    
    Only files ending with '.pdb' (case-insensitive) are considered.
    """
    files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(".pdb")])
    rmsd_values = []
    for f in files1:
        path1 = os.path.join(folder1, f)
        path2 = os.path.join(folder2, f)
        if not os.path.exists(path2):
            # Print a warning or skip if the matching file is not found.
            print(f"Warning: {f} not found in {folder2}. Skipping.", file=sys.stderr)
            continue
        coords1 = parse_pdb_coords(path1)
        coords2 = parse_pdb_coords(path2)
        rmsd = compute_rmsd_from_coords(coords1, coords2)
        if rmsd is not None:
            rmsd_values.append(rmsd)
        else:
            print(f"No common atoms for file {f} between folders. Skipping.", file=sys.stderr)
    if rmsd_values:
        avg_rmsd = sum(rmsd_values) / len(rmsd_values)
    else:
        avg_rmsd = None
    return avg_rmsd, rmsd_values

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    folder1 = sys.argv[1]
    folder2 = sys.argv[2]
    
    avg_rmsd, rmsd_list = compare_folder_pair(folder1, folder2)
    if avg_rmsd is not None:
        print(f"Compared folder '{folder1}' with '{folder2}':")
        print(f"  Number of molecule pairs compared: {len(rmsd_list)}")
        print(f"  Mean RMSD: {avg_rmsd:.3f} Å")
    else:
        print("No valid PDB file pairs were found for comparison.")

if __name__ == "__main__":
    main()
