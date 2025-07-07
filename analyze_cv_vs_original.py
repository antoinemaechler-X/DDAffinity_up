#!/usr/bin/env python3
import pandas as pd
import numpy as np

def analyze_cv_vs_original():
    """Analyze the discrepancy between CV results and original SKEMPI2 dataset."""
    
    # Load the data
    cv_results = pd.read_csv("logs_skempi_grouped/[grouped-10fold]_06_27_14_34_51/checkpoints/results_75.csv")
    skempi_original = pd.read_csv("data/SKEMPI2/SKEMPI2.csv")
    
    print(f"CV Results: {len(cv_results)} entries")
    print(f"SKEMPI2 Original: {len(skempi_original)} entries")
    print(f"Difference: {len(cv_results) - len(skempi_original)} entries")
    
    # Check for duplicates in CV results
    print(f"\n=== Duplicate Analysis ===")
    cv_duplicates = cv_results.duplicated(subset=['complex', 'mutstr'], keep=False)
    print(f"Duplicate entries in CV results: {cv_duplicates.sum()}")
    
    if cv_duplicates.sum() > 0:
        print("\nSample duplicate entries:")
        duplicates = cv_results[cv_duplicates].sort_values(['complex', 'mutstr'])
        print(duplicates.head(10))
    
    # Check unique mutations in both datasets
    print(f"\n=== Unique Mutation Analysis ===")
    cv_unique = cv_results[['complex', 'mutstr']].drop_duplicates()
    skempi_unique = skempi_original[['#Pdb', 'Mutation(s)_cleaned']].drop_duplicates()
    
    print(f"Unique mutations in CV results: {len(cv_unique)}")
    print(f"Unique mutations in SKEMPI2: {len(skempi_unique)}")
    
    # Convert SKEMPI2 format to match CV format for comparison
    def convert_skempi_to_cv_format(row):
        # Extract complex from #Pdb (e.g., "0_1CSE" -> "1CSE")
        complex_name = row['#Pdb'].split('_')[1] if '_' in row['#Pdb'] else row['#Pdb']
        return complex_name, row['Mutation(s)_cleaned']
    
    skempi_converted = [convert_skempi_to_cv_format(row) for _, row in skempi_unique.iterrows()]
    skempi_converted_df = pd.DataFrame(skempi_converted, columns=['complex', 'mutstr'])
    
    # Find mutations in CV but not in SKEMPI2
    cv_set = set(cv_unique.apply(tuple, axis=1))
    skempi_set = set(skempi_converted_df.apply(tuple, axis=1))
    
    cv_only = cv_set - skempi_set
    skempi_only = skempi_set - cv_set
    
    print(f"\nMutations in CV but not in SKEMPI2: {len(cv_only)}")
    print(f"Mutations in SKEMPI2 but not in CV: {len(skempi_only)}")
    
    if len(cv_only) > 0:
        print("\nSample mutations in CV but not in SKEMPI2:")
        for i, (complex_name, mutstr) in enumerate(list(cv_only)[:10]):
            print(f"  {complex_name}: {mutstr}")
    
    if len(skempi_only) > 0:
        print("\nSample mutations in SKEMPI2 but not in CV:")
        for i, (complex_name, mutstr) in enumerate(list(skempi_only)[:10]):
            print(f"  {complex_name}: {mutstr}")
    
    # Check for cross-validation duplicates (same mutation appearing in multiple folds)
    print(f"\n=== Cross-Validation Analysis ===")
    mutation_counts = cv_results.groupby(['complex', 'mutstr']).size()
    print(f"Average appearances per mutation: {mutation_counts.mean():.2f}")
    print(f"Max appearances per mutation: {mutation_counts.max()}")
    print(f"Mutations appearing multiple times: {(mutation_counts > 1).sum()}")
    
    # Show distribution of appearances
    print(f"\nDistribution of mutation appearances:")
    for i in range(1, int(mutation_counts.max()) + 1):
        count = (mutation_counts == i).sum()
        print(f"  {i} time(s): {count} mutations")
    
    # Check if the number of duplicates matches the expected CV fold structure
    print(f"\n=== Expected CV Structure ===")
    expected_duplicates = len(cv_unique) * 9  # Each mutation should appear in 9 out of 10 folds
    actual_duplicates = len(cv_results) - len(cv_unique)
    print(f"Expected duplicates (9-fold CV): {expected_duplicates}")
    print(f"Actual duplicates: {actual_duplicates}")
    print(f"Difference: {actual_duplicates - expected_duplicates}")

if __name__ == "__main__":
    analyze_cv_vs_original() 