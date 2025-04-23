#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt

def compute_correlations(csv_path):
    """Compute Pearson and Spearman correlations from a CSV file."""
    df = pd.read_csv(csv_path)
    # Compute Pearson and Spearman correlations from df, using pandas corr method.
    pearson_corr = df[['ddG', 'ddG_pred']].corr(method='pearson').iloc[0, 1]
    spearman_corr = df[['ddG', 'ddG_pred']].corr(method='spearman').iloc[0, 1]
    return pearson_corr, spearman_corr

def main():
    # Base directory where the CSV result files reside.
    base_dir = "logs_skempi/[10-fold-16]_04_22_22_31_26/checkpoints"
    
    # Build list of iteration indices: 0, 5, 10, ..., 70.
    iterations = list(range(0, 71, 5))
    
    pearson_values = []
    spearman_values = []
    
    for it in iterations:
        csv_file = f"results_{it}.csv"
        csv_path = os.path.join(base_dir, csv_file)
        
        if os.path.exists(csv_path):
            pearson, spearman = compute_correlations(csv_path)
            pearson_values.append(pearson)
            spearman_values.append(spearman)
            print(f"{csv_file}: Pearson={pearson:.4f}, Spearman={spearman:.4f}")
        else:
            print(f"{csv_file} not found.")
            pearson_values.append(None)
            spearman_values.append(None)

    # Create a single figure with the two curves.
    plt.figure(figsize=(10,6))
    plt.plot(iterations, pearson_values, marker='o', label='Pearson', linewidth=2)
    plt.plot(iterations, spearman_values, marker='s', label='Spearman', linewidth=2)
    plt.xlabel('Iteration (results file index)')
    plt.ylabel('Correlation Value')
    plt.title('Evolution of Pearson and Spearman Correlations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.653, color='red', linestyle='--')
    plt.axhline(y=0.627, color='red', linestyle='--')
    
    # Save and display the plot.
    output_path = os.path.join(base_dir, "correlation_evolution.png")
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")
    plt.show()

if __name__ == '__main__':
    main()
