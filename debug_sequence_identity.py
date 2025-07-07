#!/usr/bin/env python3
"""
Debug script to test sequence identity calculation
"""

import pandas as pd
from Bio import pairwise2

def compute_sequence_identity(seq1, seq2):
    """
    Compute sequence identity between two sequences using pairwise alignment.
    """
    try:
        print(f"    Computing identity between sequences:")
        print(f"      Seq1: '{seq1[:50]}...' (len: {len(seq1)})")
        print(f"      Seq2: '{seq2[:50]}...' (len: {len(seq2)})")
        
        # Use global alignment with gap penalties
        alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.5, -0.1, one_alignment_only=True)
        if not alignments:
            print(f"      No alignments found!")
            return 0.0
        
        aligned_seq1, aligned_seq2, score, start, end = alignments[0]
        
        print(f"      Alignment score: {score}")
        print(f"      Aligned Seq1: '{aligned_seq1[:100]}...'")
        print(f"      Aligned Seq2: '{aligned_seq2[:100]}...'")
        
        # Count identical positions where both sequences have amino acids (not gaps)
        identical = 0
        total_amino_acids = 0
        
        for a, b in zip(aligned_seq1, aligned_seq2):
            if a != '-' and b != '-':  # Both positions have amino acids
                total_amino_acids += 1
                if a == b:  # Identical amino acids
                    identical += 1
        
        # Use the longer original sequence length as denominator for more intuitive comparison
        max_length = max(len(seq1), len(seq2))
        
        # If there are very few aligned positions, the sequences are probably very different
        if total_amino_acids < max_length * 0.1:  # Less than 10% of longer sequence aligned
            print(f"      Very few aligned positions ({total_amino_acids} < {max_length * 0.1:.1f}), returning 0.0")
            return 0.0
        
        # Return identity based on aligned amino acid positions, normalized by longer sequence length
        identity = identical / max_length
        print(f"      Identical amino acids: {identical}, Total amino acid positions: {total_amino_acids}, Max length: {max_length}, Identity: {identity:.3f}")
        
        return identity
        
    except Exception as e:
        print(f"Error computing sequence identity: {e}")
        return 0.0

def main():
    # Load the sequences
    df = pd.read_csv('data/complex_sequences.csv')
    
    print(f"Loaded {len(df)} sequences")
    print(f"Sample sequences:")
    for i, row in df.head(5).iterrows():
        print(f"  {row['complex_name']}_{row['chain_name']}: {row['sequence'][:50]}... (length: {row['sequence_length']})")
    
    # Test with first few sequences
    sequences = df.head(10)
    
    print(f"\nTesting sequence identity calculation:")
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            seq1 = sequences.iloc[i]['sequence']
            seq2 = sequences.iloc[j]['sequence']
            name1 = f"{sequences.iloc[i]['complex_name']}_{sequences.iloc[i]['chain_name']}"
            name2 = f"{sequences.iloc[j]['complex_name']}_{sequences.iloc[j]['chain_name']}"
            
            identity = compute_sequence_identity(seq1, seq2)
            print(f"  {name1} vs {name2}: {identity:.3f}")
            print(f"    Seq1: {seq1[:30]}... (len: {len(seq1)})")
            print(f"    Seq2: {seq2[:30]}... (len: {len(seq2)})")
            print()

if __name__ == "__main__":
    main() 