#!/usr/bin/env python3
"""
Comprehensive test script for sequence-based grouping.
Generates a detailed CSV file with complex information, mutations, sequences, and group assignments.
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from tqdm.auto import tqdm
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rde.utils.skempi_mpnn_sequence_grouped import SequenceGroupedSkempiDataset
from rde.datasets.skempi_parallel import load_category_entries

# Import configuration
try:
    from test_config import DATA_CONFIG, ANALYSIS_CONFIG, ALTERNATIVE_CONFIGS
except ImportError:
    print("Warning: test_config.py not found, using default configuration")
    DATA_CONFIG = {
        'csv_path': "data/skempi_v2.csv",
        'pdb_wt_dir': "data/pdb_wt",
        'pdb_mt_dir': "data/pdb_mt",
        'cache_dir': "cache",
        'output_csv_path': "detailed_sequence_analysis.csv"
    }
    ANALYSIS_CONFIG = {
        'sequence_identity_threshold': 0.7,
        'reset': True,
        'show_debug_info': True,
        'max_groups_to_show': 10,
        'max_complexes_per_group': 3
    }
    ALTERNATIVE_CONFIGS = {}


def create_detailed_analysis_csv(csv_path, pdb_wt_dir, pdb_mt_dir, cache_dir, output_csv_path, 
                                sequence_identity_threshold=0.7, reset=False, show_debug_info=True,
                                max_groups_to_show=10, max_complexes_per_group=3):
    """
    Create a detailed CSV file with complex information, mutations, sequences, and group assignments.
    
    Args:
        csv_path: Path to the main CSV file
        pdb_wt_dir: Directory containing wild-type PDB files
        pdb_mt_dir: Directory containing mutant PDB files
        cache_dir: Directory for caching results
        output_csv_path: Path for the output CSV file
        sequence_identity_threshold: Threshold for grouping similar sequences
        reset: Whether to reset cached results
        show_debug_info: Whether to show detailed debug information
        max_groups_to_show: Number of groups to show in summary
        max_complexes_per_group: Number of complexes to show per group in summary
    """
    
    print("Loading entries...")
    entries_full = load_category_entries(csv_path, pdb_wt_dir, pdb_mt_dir)
    print(f"Loaded {len(entries_full)} total entries")
    
    if show_debug_info:
        # Show sample entry structure
        if entries_full:
            sample_entry = entries_full[0]
            print(f"\nSample entry structure:")
            for key, value in sample_entry.items():
                if isinstance(value, (str, int, float)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value)}")
    
    # Create dataset instance to access all the grouping functions
    dataset = SequenceGroupedSkempiDataset(
        csv_path=csv_path,
        pdb_wt_dir=pdb_wt_dir,
        pdb_mt_dir=pdb_mt_dir,
        cache_dir=cache_dir,
        split='train',  # Doesn't matter for this analysis
        reset=reset,
        sequence_identity_threshold=sequence_identity_threshold
    )
    
    print("\nExtracting complex sequences...")
    complex_sequences = dataset._extract_complex_sequences()
    
    print("\nComputing sequence similarities...")
    complex_list = list(complex_sequences.keys())
    similarity_matrix = dataset._compute_sequence_similarities(complex_list, complex_sequences)
    
    print("\nClustering complexes into groups...")
    groups = dataset._cluster_complexes(complex_list, similarity_matrix)
    
    # Create mapping from complex to group
    complex_to_group = {}
    for group_idx, group in enumerate(groups):
        for complex_name in group:
            complex_to_group[complex_name] = group_idx
    
    print(f"\nCreated {len(groups)} groups")
    for i, group in enumerate(groups[:max_groups_to_show]):
        print(f"  Group {i}: {len(group)} complexes - {group[:3]}...")
    
    # Prepare data for CSV
    csv_data = []
    
    print("\nProcessing entries for CSV...")
    for entry in tqdm(entries_full, desc="Processing entries"):
        complex_name = entry['complex']
        
        # Get the full complex name (with prefix if it exists)
        full_complex_name = complex_name
        
        # Extract the cut name (remove prefix if it exists)
        # Look for pattern like "0_1CSE", "1_1CSE", etc.
        if '_' in complex_name:
            parts = complex_name.split('_', 1)
            if len(parts) == 2 and parts[0].isdigit():
                cut_name = parts[1]
            else:
                cut_name = complex_name
        else:
            cut_name = complex_name
        
        # Get the parsed sequence
        parsed_sequence = complex_sequences.get(complex_name, "N/A")
        
        # Get the group assignment
        group_id = complex_to_group.get(complex_name, -1)
        
        # Create row data
        row_data = {
            'full_complex_name': full_complex_name,
            'cut_complex_name': cut_name,
            'mutation': entry['mutstr'],
            'parsed_sequence': parsed_sequence,
            'group_id': group_id,
            'group_size': len(groups[group_id]) if group_id >= 0 else 0,
            'num_mutations': entry['num_muts'],
            'ddG': entry['ddG'],
            'pdbcode': entry['pdbcode'],
            'id': entry['id']
        }
        
        csv_data.append(row_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    
    print(f"\nSaving detailed analysis to {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total entries: {len(df)}")
    print(f"Unique complexes: {df['full_complex_name'].nunique()}")
    print(f"Unique cut names: {df['cut_complex_name'].nunique()}")
    print(f"Total groups: {len(groups)}")
    print(f"Average group size: {df['group_size'].mean():.2f}")
    print(f"Median group size: {df['group_size'].median():.2f}")
    print(f"Max group size: {df['group_size'].max()}")
    print(f"Min group size: {df['group_size'].min()}")
    
    # Show group size distribution
    print(f"\nGroup size distribution:")
    group_sizes = df['group_size'].value_counts().sort_index()
    for size, count in group_sizes.items():
        print(f"  Groups with {size} complexes: {count}")
    
    # Show some examples of similar complexes
    print(f"\nExamples of similar complexes (first {min(5, len(groups))} groups):")
    for i, group in enumerate(groups[:5]):
        print(f"  Group {i} ({len(group)} complexes):")
        for j, complex_name in enumerate(group[:max_complexes_per_group]):
            seq = complex_sequences.get(complex_name, "N/A")
            seq_preview = seq[:50] + "..." if len(seq) > 50 else seq
            print(f"    {j+1}. {complex_name}: {seq_preview}")
        if len(group) > max_complexes_per_group:
            print(f"    ... and {len(group) - max_complexes_per_group} more")
        print()
    
    return df, groups, complex_sequences


def analyze_sequence_similarities(df, groups, complex_sequences, output_dir):
    """Additional analysis of sequence similarities within and between groups."""
    
    print("\nAnalyzing sequence similarities...")
    
    # Create similarity analysis file
    similarity_analysis = []
    
    for group_idx, group in enumerate(groups):
        if len(group) > 1:
            # Analyze similarities within this group
            group_similarities = []
            for i, complex1 in enumerate(group):
                for j, complex2 in enumerate(group[i+1:], i+1):
                    seq1 = complex_sequences.get(complex1, "")
                    seq2 = complex_sequences.get(complex2, "")
                    
                    if seq1 and seq2 and seq1 != complex1 and seq2 != complex2:
                        # Compute similarity (simplified version)
                        from Bio import pairwise2
                        try:
                            alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -0.5, -0.1, one_alignment_only=True)
                            if alignments:
                                aligned_seq1, aligned_seq2, score, start, end = alignments[0]
                                identical = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
                                total = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' and b != '-')
                                similarity = identical / total if total > 0 else 0.0
                                group_similarities.append(similarity)
                        except:
                            pass
            
            if group_similarities:
                similarity_analysis.append({
                    'group_id': group_idx,
                    'group_size': len(group),
                    'avg_similarity': np.mean(group_similarities),
                    'min_similarity': np.min(group_similarities),
                    'max_similarity': np.max(group_similarities),
                    'complexes': ';'.join(group)
                })
    
    # Save similarity analysis
    if similarity_analysis:
        similarity_df = pd.DataFrame(similarity_analysis)
        similarity_path = os.path.join(output_dir, 'group_similarity_analysis.csv')
        similarity_df.to_csv(similarity_path, index=False)
        print(f"Saved group similarity analysis to {similarity_path}")
        
        print(f"\nGroup similarity statistics:")
        print(f"Average within-group similarity: {similarity_df['avg_similarity'].mean():.3f}")
        print(f"Minimum within-group similarity: {similarity_df['min_similarity'].min():.3f}")
        print(f"Maximum within-group similarity: {similarity_df['max_similarity'].max():.3f}")


def main():
    """Main function to run the detailed analysis."""
    
    parser = argparse.ArgumentParser(description='Run detailed sequence analysis')
    parser.add_argument('--config', type=str, default='medium', 
                       choices=['strict', 'lenient', 'medium', 'custom'],
                       help='Configuration to use (strict=90%%, lenient=50%%, medium=70%%)')
    parser.add_argument('--csv-path', type=str, help='Path to CSV file')
    parser.add_argument('--pdb-wt-dir', type=str, help='Path to wild-type PDB directory')
    parser.add_argument('--pdb-mt-dir', type=str, help='Path to mutant PDB directory')
    parser.add_argument('--cache-dir', type=str, help='Path to cache directory')
    parser.add_argument('--output', type=str, help='Output CSV path')
    parser.add_argument('--threshold', type=float, help='Sequence identity threshold')
    parser.add_argument('--no-reset', action='store_true', help='Use cached results')
    
    args = parser.parse_args()
    
    # Start with default configuration
    config = DATA_CONFIG.copy()
    analysis_config = ANALYSIS_CONFIG.copy()
    
    # Apply configuration based on argument
    if args.config in ALTERNATIVE_CONFIGS:
        config.update(ALTERNATIVE_CONFIGS[args.config])
        analysis_config.update(ALTERNATIVE_CONFIGS[args.config])
    
    # Override with command line arguments
    if args.csv_path:
        config['csv_path'] = args.csv_path
    if args.pdb_wt_dir:
        config['pdb_wt_dir'] = args.pdb_wt_dir
    if args.pdb_mt_dir:
        config['pdb_mt_dir'] = args.pdb_mt_dir
    if args.cache_dir:
        config['cache_dir'] = args.cache_dir
    if args.output:
        config['output_csv_path'] = args.output
    if args.threshold:
        analysis_config['sequence_identity_threshold'] = args.threshold
    if args.no_reset:
        analysis_config['reset'] = False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(config['output_csv_path']) if os.path.dirname(config['output_csv_path']) else '.', exist_ok=True)
    
    print("Starting detailed sequence analysis...")
    print(f"Configuration: {args.config}")
    print(f"CSV path: {config['csv_path']}")
    print(f"PDB WT dir: {config['pdb_wt_dir']}")
    print(f"PDB MT dir: {config['pdb_mt_dir']}")
    print(f"Cache dir: {config['cache_dir']}")
    print(f"Output CSV: {config['output_csv_path']}")
    print(f"Sequence identity threshold: {analysis_config['sequence_identity_threshold']}")
    print(f"Reset cache: {analysis_config['reset']}")
    
    try:
        # Run the detailed analysis
        df, groups, complex_sequences = create_detailed_analysis_csv(
            csv_path=config['csv_path'],
            pdb_wt_dir=config['pdb_wt_dir'],
            pdb_mt_dir=config['pdb_mt_dir'],
            cache_dir=config['cache_dir'],
            output_csv_path=config['output_csv_path'],
            sequence_identity_threshold=analysis_config['sequence_identity_threshold'],
            reset=analysis_config['reset'],
            show_debug_info=analysis_config['show_debug_info'],
            max_groups_to_show=analysis_config['max_groups_to_show'],
            max_complexes_per_group=analysis_config['max_complexes_per_group']
        )
        
        # Additional analysis
        output_dir = os.path.dirname(config['output_csv_path']) if os.path.dirname(config['output_csv_path']) else '.'
        analyze_sequence_similarities(df, groups, complex_sequences, output_dir)
        
        print(f"\nAnalysis complete! Results saved to {config['output_csv_path']}")
        print(f"Total entries processed: {len(df)}")
        print(f"Total groups created: {len(groups)}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 