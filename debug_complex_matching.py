#!/usr/bin/env python3
"""
Debug script to understand why some complexes are not matching the groups.
"""

import pandas as pd
from collections import defaultdict

def extract_base_complex_name(complex_name):
    """Extract base complex name by removing numeric prefix (e.g., '0_1CSE' -> '1CSE')"""
    if '_' in complex_name:
        parts = complex_name.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            return parts[1]
    return complex_name

def main():
    print("=== DEBUGGING COMPLEX MATCHING ===")
    
    # Load the grouped CSV
    print("Loading grouped CSV...")
    grouped_df = pd.read_csv('data/complex_sequences_grouped.csv')
    grouped_complexes = set(grouped_df['complex_name'].unique())
    print(f"Grouped CSV has {len(grouped_complexes)} unique complexes")
    print(f"Sample grouped complexes: {list(grouped_complexes)[:10]}")
    
    # Load the main dataset
    print("\nLoading main dataset...")
    main_df = pd.read_csv('data/SKEMPI2/SKEMPI2.csv')
    main_complexes = set(main_df['#Pdb'].unique())
    print(f"Main dataset has {len(main_complexes)} unique complexes")
    print(f"Sample main complexes: {list(main_complexes)[:10]}")
    
    # Extract base names from main dataset
    print("\nExtracting base names from main dataset...")
    base_complexes = set()
    for complex_name in main_complexes:
        base_name = extract_base_complex_name(complex_name)
        base_complexes.add(base_name)
    
    print(f"After extracting base names: {len(base_complexes)} unique base complexes")
    print(f"Sample base complexes: {list(base_complexes)[:10]}")
    
    # Find mismatches
    print("\nFinding mismatches...")
    missing_from_groups = base_complexes - grouped_complexes
    missing_from_main = grouped_complexes - base_complexes
    
    print(f"Complexes in main dataset but not in groups: {len(missing_from_groups)}")
    if missing_from_groups:
        print(f"Missing from groups: {sorted(list(missing_from_groups))}")
    
    print(f"Complexes in groups but not in main dataset: {len(missing_from_main)}")
    if missing_from_main:
        print(f"Missing from main: {sorted(list(missing_from_main))}")
    
    # Show some examples of the mismatches
    if missing_from_groups:
        print(f"\nExamples of complexes not in groups:")
        for complex_name in sorted(list(missing_from_groups))[:10]:
            # Find all entries with this base name
            matching_entries = []
            for _, row in main_df.iterrows():
                base_name = extract_base_complex_name(row['#Pdb'])
                if base_name == complex_name:
                    matching_entries.append(row['#Pdb'])
            print(f"  {complex_name}: {len(matching_entries)} entries - {matching_entries[:3]}...")
    
    # Check if there are any patterns in the mismatches
    print(f"\nAnalyzing patterns in mismatches...")
    if missing_from_groups:
        print("Complexes missing from groups:")
        for complex_name in sorted(list(missing_from_groups)):
            print(f"  {complex_name}")
    
    print("=== END DEBUGGING ===")

if __name__ == "__main__":
    main() 