#!/usr/bin/env python3
"""
Script to compute sequence similarities and create clusters.
Takes the sequences CSV and adds a group column based on sequence similarity.
"""

import pandas as pd
import numpy as np
from Bio import pairwise2
from tqdm import tqdm
import argparse
from collections import defaultdict


def compute_sequence_identity(seq1, seq2):
    """
    Compute sequence identity between two sequences using pairwise alignment.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
    
    Returns:
        float: Sequence identity (0.0 to 1.0)
    """
    try:
        # Use global alignment with gap penalties
        alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.5, -0.1, one_alignment_only=True)
        if not alignments:
            return 0.0
        
        aligned_seq1, aligned_seq2, score, start, end = alignments[0]
        
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
            return 0.0
        
        # Return identity based on aligned amino acid positions, normalized by longer sequence length
        return identical / max_length
        
    except Exception as e:
        print(f"Error computing sequence identity: {e}")
        return 0.0


def compute_similarity_matrix(sequences):
    """
    Compute pairwise similarity matrix for all sequences.
    Each sequence is compared individually (no concatenation).
    
    Args:
        sequences: List of (complex_name, chain_name, sequence) tuples
    
    Returns:
        tuple: (similarity_matrix, complex_chain_names)
    """
    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    complex_chain_names = []
    
    print(f"Computing pairwise similarities for {n} individual chains...")
    
    for i in tqdm(range(n), desc="Computing similarities"):
        complex_chain_names.append(f"{sequences[i][0]}_{sequences[i][1]}")
        
        for j in range(i+1, n):
            seq1 = sequences[i][2]  # Individual chain sequence
            seq2 = sequences[j][2]  # Individual chain sequence
            
            identity = compute_sequence_identity(seq1, seq2)
            similarity_matrix[i, j] = identity
            similarity_matrix[j, i] = identity
        
        # Self-similarity is 1.0
        similarity_matrix[i, i] = 1.0
    
    return similarity_matrix, complex_chain_names


def cluster_sequences(similarity_matrix, complex_chain_names, threshold=0.9):
    """
    Cluster sequences based on similarity threshold.
    This creates groups of similar individual chains, regardless of complex.
    
    Args:
        similarity_matrix: NxN similarity matrix
        complex_chain_names: List of complex_chain names
        threshold: Sequence identity threshold for clustering
    
    Returns:
        dict: {complex_chain_name: group_id}
    """
    n = len(complex_chain_names)
    visited = [False] * n
    groups = []
    group_id = 0
    
    print(f"Clustering {n} individual chains with {threshold*100}% sequence identity threshold...")
    
    for i in range(n):
        if visited[i]:
            continue
        
        # Start a new group
        group = [complex_chain_names[i]]
        visited[i] = True
        
        # Find all sequences similar to this one
        for j in range(n):
            if not visited[j] and similarity_matrix[i, j] >= threshold:
                group.append(complex_chain_names[j])
                visited[j] = True
        
        groups.append(group)
        group_id += 1
        
        if len(groups) <= 5:  # Show first few groups
            print(f"  Group {len(groups)}: {len(group)} chains (first: {group[0]})")
    
    print(f"Created {len(groups)} groups from {n} individual chains")
    
    # Create mapping from complex_chain_name to group_id
    group_mapping = {}
    for group_id, group in enumerate(groups):
        for complex_chain_name in group:
            group_mapping[complex_chain_name] = group_id
    
    return group_mapping, groups


def create_complex_groups(df, chain_group_mapping):
    """
    Create complex-level groups based on individual chain groups.
    A complex belongs to a group if any of its chains belong to that group.
    
    Args:
        df: DataFrame with complex_name, chain_name, group_id columns
        chain_group_mapping: Mapping from complex_chain_name to group_id
    
    Returns:
        dict: {complex_name: complex_group_id}
    """
    # Get all unique complexes
    complexes = df['complex_name'].unique()
    
    # For each complex, find all groups its chains belong to
    complex_to_chain_groups = {}
    for complex_name in complexes:
        complex_chains = df[df['complex_name'] == complex_name]
        chain_groups = set()
        for _, row in complex_chains.iterrows():
            complex_chain_name = f"{row['complex_name']}_{row['chain_name']}"
            if complex_chain_name in chain_group_mapping:
                chain_groups.add(chain_group_mapping[complex_chain_name])
        complex_to_chain_groups[complex_name] = chain_groups
    
    # Create complex groups: complexes that share any chain group are in the same complex group
    complex_groups = []
    visited_complexes = set()
    
    for complex_name in complexes:
        if complex_name in visited_complexes:
            continue
        
        # Start a new complex group
        complex_group = [complex_name]
        visited_complexes.add(complex_name)
        
        # Find all complexes that share any chain group with this complex
        shared_chain_groups = complex_to_chain_groups[complex_name]
        for other_complex in complexes:
            if other_complex not in visited_complexes:
                other_chain_groups = complex_to_chain_groups[other_complex]
                if shared_chain_groups.intersection(other_chain_groups):
                    complex_group.append(other_complex)
                    visited_complexes.add(other_complex)
        
        complex_groups.append(complex_group)
    
    # Create mapping from complex_name to complex_group_id
    complex_group_mapping = {}
    for group_id, group in enumerate(complex_groups):
        for complex_name in group:
            complex_group_mapping[complex_name] = group_id
    
    return complex_group_mapping, complex_groups


def main():
    parser = argparse.ArgumentParser(description='Compute sequence similarities and create clusters')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV with sequences')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV path (default: input_csv with _grouped suffix)')
    parser.add_argument('--threshold', type=float, default=0.9, help='Sequence identity threshold for clustering (default: 0.9)')
    
    args = parser.parse_args()
    
    # Set output path if not provided
    if args.output_csv is None:
        base_name = args.input_csv.replace('.csv', '')
        args.output_csv = f"{base_name}_grouped.csv"
    
    print(f"Loading sequences from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    print(f"Loaded {len(df)} sequences from {df['complex_name'].nunique()} complexes")
    
    # Prepare sequences for similarity computation
    sequences = []
    for _, row in df.iterrows():
        sequences.append((row['complex_name'], row['chain_name'], row['sequence']))
    
    # Compute similarity matrix
    similarity_matrix, complex_chain_names = compute_similarity_matrix(sequences)
    
    # Show some similarity statistics
    similarities = []
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            similarities.append(similarity_matrix[i, j])
    
    similarities = np.array(similarities)
    print(f"\nSimilarity statistics:")
    print(f"  Min: {similarities.min():.3f}")
    print(f"  Max: {similarities.max():.3f}")
    print(f"  Mean: {similarities.mean():.3f}")
    print(f"  Std: {similarities.std():.3f}")
    print(f"  Similarities >= {args.threshold}: {np.sum(similarities >= args.threshold)}/{len(similarities)} ({np.sum(similarities >= args.threshold)/len(similarities)*100:.1f}%)")
    
    # Create clusters
    group_mapping, groups = cluster_sequences(similarity_matrix, complex_chain_names, args.threshold)
    
    # Add chain-level group information to DataFrame
    df['chain_group_id'] = df.apply(lambda row: group_mapping.get(f"{row['complex_name']}_{row['chain_name']}", -1), axis=1)
    
    # Create complex-level groups
    complex_group_mapping, complex_groups = create_complex_groups(df, group_mapping)
    
    # Add complex-level group information to DataFrame
    df['complex_group_id'] = df['complex_name'].map(complex_group_mapping)
    
    # Save results
    df.to_csv(args.output_csv, index=False)
    
    print(f"\nResults saved to {args.output_csv}")
    print(f"Chain-level groups: {len(groups)}")
    print(f"Complex-level groups: {len(complex_groups)}")
    print(f"Average chain group size: {np.mean([len(g) for g in groups]):.1f}")
    print(f"Average complex group size: {np.mean([len(g) for g in complex_groups]):.1f}")
    
    # Show some examples of similar sequences
    print(f"\nExample chain groups:")
    for i, group in enumerate(groups[:3]):
        print(f"  Chain Group {i}: {len(group)} chains")
        for j, complex_chain in enumerate(group[:3]):
            print(f"    {complex_chain}")
        if len(group) > 3:
            print(f"    ... and {len(group) - 3} more")
    
    print(f"\nExample complex groups:")
    for i, group in enumerate(complex_groups[:3]):
        print(f"  Complex Group {i}: {len(group)} complexes")
        for j, complex_name in enumerate(group[:3]):
            print(f"    {complex_name}")
        if len(group) > 3:
            print(f"    ... and {len(group) - 3} more")


if __name__ == "__main__":
    main() 