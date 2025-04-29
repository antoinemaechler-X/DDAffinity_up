#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr

# import scoring modules
from burial_score import burial_score, burial_scores
from interface_score import interface_score, interface_scores


def load_data(csv_path, pdb_dir, num_lines, mode, neighbor_count, sigma, protein):
    df = pd.read_csv(csv_path)
    df = df[df['#Pdb_origin'] == protein]
    if num_lines is not None:
        df = df.iloc[:num_lines]
    X1, X2, y, labels = [], [], [], []
    for idx, row in df.iterrows():
        pdb_file = os.path.join(pdb_dir, f"{row['#Pdb']}.pdb")
        muts = row['Mutation(s)_cleaned'].strip('"').split(',')
        try:
            ddg_val = abs(float(row['ddG']))
        except:
            continue
        if mode == 'individual':
            for mut in muts:
                b = burial_score(pdb_file, mut, neighbor_count=neighbor_count)
                i = interface_score(pdb_file, mut, sigma=sigma)
                X1.append(b); X2.append(i); y.append(ddg_val); labels.append(mut)
        else:
            b_list = burial_scores(pdb_file, muts, neighbor_count=neighbor_count)
            i_list = interface_scores(pdb_file, muts, sigma=sigma)
            if not b_list or not i_list:
                continue
            if mode == 'mean':
                b = np.mean(b_list); i = np.mean(i_list)
            elif mode == 'sum':
                b = np.sum(b_list); i = np.sum(i_list)
            else:  # max
                b = np.max(b_list); i = np.max(i_list)
            X1.append(b); X2.append(i); y.append(ddg_val); labels.append(';'.join(muts))
    X = np.column_stack([X1, X2])
    return X, np.array(y), labels


def main():
    parser = argparse.ArgumentParser(description="Predict ddG from burial/interface scores with annotations")
    parser.add_argument('-p', '--protein', type=str, required=True,
                        help='PDB code to filter (e.g. 1A22)')
    parser.add_argument('-n', '--num_lines', type=int, default=None,
                        help='Number of CSV rows to process after filtering')
    parser.add_argument('-m', '--mode', choices=['mean', 'max', 'individual', 'sum'], default='mean',
                        help='Aggregation mode')
    parser.add_argument('-k', '--neighbors', type=int, default=5,
                        help='Number of neighbors for burial score')
    parser.add_argument('-s', '--sigma', type=float, default=5.0,
                        help='σ (Å) for Gaussian weighting in interface score')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(here, 'plots_centrality_interface')
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(here, '..', 'data', 'SKEMPI2', 'M1340.csv')
    pdb_dir = os.path.join(here, '..', 'data', 'SKEMPI2', 'M1340_cache', 'wildtype')

    X, y, labels = load_data(csv_path, pdb_dir, args.num_lines,
                              args.mode, args.neighbors, args.sigma, args.protein)

    model = LinearRegression()
    model.fit(X, y)
    lam, mu = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X)

    r2 = model.score(X, y)
    pearson_r, _ = pearsonr(y, y_pred)
    spearman_r, _ = spearmanr(y, y_pred)
    print(f"Model: y = {lam:.4f}*burial + {mu:.4f}*interface + {intercept:.4f}")
    print(f"R²={r2:.4f}, Pearson r={pearson_r:.4f}, Spearman ρ={spearman_r:.4f}")

    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.7)
    for t, p, lab in zip(y, y_pred, labels):
        ax.text(t, p, lab, fontsize=6, alpha=0.7)
    mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlabel('True Absolute ddG')
    ax.set_ylabel('Predicted Absolute ddG')
    ax.set_title(f"{args.protein} | mode={args.mode} k={args.neighbors} σ={args.sigma}")
    metrics = f"R²={r2:.2f}\nPearson={pearson_r:.2f}\nSpearman={spearman_r:.2f}"
    ax.text(0.05, 0.95, metrics, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.tight_layout()
    out = f"ddG_predicted_{args.protein}_{args.mode}_k{args.neighbors}_s{args.sigma}.png"
    filepath = os.path.join(plots_dir, out)
    plt.savefig(filepath)
    print(f"Saved {filepath}")

if __name__ == '__main__':
    main()
