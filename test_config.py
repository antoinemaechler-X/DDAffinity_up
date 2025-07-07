#!/usr/bin/env python3
"""
Configuration file for the detailed sequence analysis test.
Adjust these paths and parameters as needed for your environment.
"""

# Data paths - adjust these to match your file structure
DATA_CONFIG = {
    'csv_path': "data/SKEMPI2/SKEMPI2.csv",  # Path to the main CSV file
    'pdb_wt_dir': "data/SKEMPI2/SKEMPI2_cache/wildtype",  # Directory containing wild-type PDB files
    'pdb_mt_dir': "data/SKEMPI2/SKEMPI2_cache/optimized",  # Directory containing mutant PDB files
    'cache_dir': "cache",              # Directory for caching results
    'output_csv_path': "detailed_sequence_analysis.csv"  # Output CSV file
}

# Analysis parameters
ANALYSIS_CONFIG = {
    'sequence_identity_threshold': 0.7,  # 70% sequence identity for grouping
    'reset': True,                       # Force recomputation of groups (ignore cache)
    'show_debug_info': True,             # Show detailed debug information
    'max_groups_to_show': 10,            # Number of groups to show in summary
    'max_complexes_per_group': 3         # Number of complexes to show per group in summary
}

# Alternative configurations for testing different thresholds
ALTERNATIVE_CONFIGS = {
    'strict': {
        'sequence_identity_threshold': 0.9,  # 90% - very strict grouping
        'output_csv_path': "detailed_sequence_analysis_strict.csv"
    },
    'lenient': {
        'sequence_identity_threshold': 0.5,  # 50% - more lenient grouping
        'output_csv_path': "detailed_sequence_analysis_lenient.csv"
    },
    'medium': {
        'sequence_identity_threshold': 0.7,  # 70% - medium grouping
        'output_csv_path': "detailed_sequence_analysis_medium.csv"
    }
} 