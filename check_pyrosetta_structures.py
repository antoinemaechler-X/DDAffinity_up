#!/usr/bin/env python3
"""
Script to check available PyRosetta structures and verify the first 4000 structures
exist in both wildtype and optimized folders.
"""
import os
import glob
from collections import defaultdict

def main():
    # Define paths
    wildtype_dir = "data/SKEMPI2/SKEMPI2_cache/wildtype_pyrosetta"
    optimized_dir = "data/SKEMPI2/SKEMPI2_cache/optimized_pyrosetta"
    
    print("Checking PyRosetta structures...")
    print(f"Wildtype directory: {wildtype_dir}")
    print(f"Optimized directory: {optimized_dir}")
    
    # Check if directories exist
    if not os.path.exists(wildtype_dir):
        print(f"ERROR: Wildtype directory does not exist: {wildtype_dir}")
        return
    if not os.path.exists(optimized_dir):
        print(f"ERROR: Optimized directory does not exist: {optimized_dir}")
        return
    
    # Get all PDB files in each directory
    wildtype_files = glob.glob(os.path.join(wildtype_dir, "*.pdb"))
    optimized_files = glob.glob(os.path.join(optimized_dir, "*.pdb"))
    
    print(f"\nFound {len(wildtype_files)} wildtype files")
    print(f"Found {len(optimized_files)} optimized files")
    
    # Extract structure IDs (e.g., "1234_1ABC.pdb" -> "1234_1ABC")
    wildtype_ids = set(os.path.splitext(os.path.basename(f))[0] for f in wildtype_files)
    optimized_ids = set(os.path.splitext(os.path.basename(f))[0] for f in optimized_files)
    
    # Find common structures
    common_ids = wildtype_ids.intersection(optimized_ids)
    print(f"\nCommon structures in both directories: {len(common_ids)}")
    
    # Sort by structure ID (assuming format "number_PDBCODE")
    sorted_common_ids = sorted(common_ids, key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0)
    
    # Check first 4000 structures
    max_structures = 4000
    available_structures = sorted_common_ids[:max_structures]
    
    print(f"\nFirst {len(available_structures)} structures available in both directories:")
    for i, struct_id in enumerate(available_structures[:10]):  # Show first 10
        print(f"  {i+1:4d}: {struct_id}")
    if len(available_structures) > 10:
        print(f"  ... and {len(available_structures) - 10} more")
    
    # Check if we have enough structures
    if len(available_structures) >= max_structures:
        print(f"\n✓ SUCCESS: Found {len(available_structures)} structures (>= {max_structures} required)")
    else:
        print(f"\n⚠ WARNING: Only found {len(available_structures)} structures (< {max_structures} required)")
    
    # Show some statistics
    print(f"\nStructure ID ranges:")
    if available_structures:
        first_id = available_structures[0]
        last_id = available_structures[-1]
        print(f"  First: {first_id}")
        print(f"  Last:  {last_id}")
    
    # Check for any gaps in the sequence
    print(f"\nChecking for gaps in structure sequence...")
    gaps = []
    for i in range(len(available_structures) - 1):
        current_num = int(available_structures[i].split('_')[0])
        next_num = int(available_structures[i + 1].split('_')[0])
        if next_num - current_num > 1:
            gaps.append((current_num, next_num))
    
    if gaps:
        print(f"Found {len(gaps)} gaps in structure sequence:")
        for start, end in gaps[:5]:  # Show first 5 gaps
            print(f"  Gap between {start} and {end}")
        if len(gaps) > 5:
            print(f"  ... and {len(gaps) - 5} more gaps")
    else:
        print("No gaps found in structure sequence")
    
    return available_structures

if __name__ == "__main__":
    available_structures = main() 