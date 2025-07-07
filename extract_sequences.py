#!/usr/bin/env python3
"""
Script to extract protein sequences from PDB files and save to CSV.
Creates a CSV with columns: complex_name, chain_name, sequence
"""

import os
import pandas as pd
from collections import defaultdict
from Bio import PDB
from Bio.PDB.Polypeptide import three_to_one
from tqdm import tqdm
import argparse

from rde.datasets.skempi_parallel import load_category_entries


def extract_sequence_from_pdb(pdb_path, chain_id=None):
    """
    Extract protein sequence from a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Specific chain ID to extract (if None, extract all chains)
    
    Returns:
        dict: {chain_id: sequence} for each chain
    """
    if not os.path.exists(pdb_path):
        return {}
    
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        sequences = {}
        
        # Get all chains
        chains = structure[0].get_chains()
        
        for chain in chains:
            if chain_id is not None and chain.id != chain_id:
                continue
                
            chain_seq = ""
            for residue in chain:
                # Check if residue is a standard amino acid
                if hasattr(residue, 'get_id'):
                    res_id = residue.get_id()
                    if isinstance(res_id, tuple) and len(res_id) > 0 and res_id[0] == " ":
                        try:
                            resname = residue.get_resname()
                            one_letter = three_to_one(resname)
                            chain_seq += one_letter
                        except KeyError:
                            # Non-standard amino acid, skip
                            continue
            
            if chain_seq:
                sequences[chain.id] = chain_seq
        
        return sequences
        
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Extract protein sequences from PDB files')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to SKEMPI CSV file')
    parser.add_argument('--pdb_wt_dir', type=str, required=True, help='Directory containing wild-type PDB files')
    parser.add_argument('--output_csv', type=str, default='data/complex_sequences.csv', 
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    print("Loading entries from CSV...")
    entries = load_category_entries(args.csv_path, args.pdb_wt_dir, args.pdb_wt_dir)
    
    print(f"Found {len(entries)} entries")
    
    # Group entries by complex
    complex_to_entries = defaultdict(list)
    for entry in entries:
        complex_to_entries[entry['complex']].append(entry)
    
    print(f"Found {len(complex_to_entries)} unique complexes")
    
    # Extract sequences for each complex
    results = []
    
    for complex_name, complex_entries in tqdm(complex_to_entries.items(), desc="Extracting sequences"):
        # Use the first entry for each complex
        entry = complex_entries[0]
        pdb_path = entry.get('pdb_wt_path')
        
        if not pdb_path or not os.path.exists(pdb_path):
            print(f"Warning: PDB file not found for complex {complex_name}")
            continue
        
        # Extract sequences from all chains
        sequences = extract_sequence_from_pdb(pdb_path)
        
        if not sequences:
            print(f"Warning: No sequences extracted for complex {complex_name}")
            continue
        
        # Add each chain sequence to results
        for chain_id, sequence in sequences.items():
            results.append({
                'complex_name': complex_name,
                'chain_name': chain_id,
                'sequence': sequence,
                'sequence_length': len(sequence)
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    df.to_csv(args.output_csv, index=False)
    
    print(f"\nExtraction complete!")
    print(f"Total sequences extracted: {len(df)}")
    print(f"Unique complexes: {df['complex_name'].nunique()}")
    print(f"Unique chains: {df['chain_name'].nunique()}")
    print(f"Average sequence length: {df['sequence_length'].mean():.1f}")
    print(f"Output saved to: {args.output_csv}")
    
    # Show some examples
    print(f"\nSample sequences:")
    for i, row in df.head(5).iterrows():
        print(f"  {row['complex_name']}_{row['chain_name']}: {row['sequence'][:50]}... (length: {row['sequence_length']})")


if __name__ == "__main__":
    main() 