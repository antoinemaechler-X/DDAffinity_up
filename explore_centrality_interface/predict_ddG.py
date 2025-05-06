#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr, spearmanr

from burial_score import burial_score, burial_scores
from interface_score import interface_score, interface_scores

def load_data(csv_path, pdb_dir, num_lines, mode,
              neighbor_count, sigma_interface, protein):
    df = pd.read_csv(csv_path)
    df = df[df['#Pdb_origin'] == protein]
    if num_lines is not None:
        df = df.iloc[:num_lines]
    X1, X2, y, labels, lengths = [], [], [], [], []

    for _, row in df.iterrows():
        pdb_file = os.path.join(pdb_dir, f"{row['#Pdb']}.pdb")
        muts = row['Mutation(s)_cleaned'].strip('"').split(',')
        n_muts = len(muts)
        try:
            ddg_val = abs(float(row['ddG']))
        except:
            continue

        if mode == 'individual':
            for mut in muts:
                b = burial_score(pdb_file, mut, neighbor_count=neighbor_count)
                i = interface_score(pdb_file, mut, sigma_interface=sigma_interface)
                X1.append(b); X2.append(i); y.append(ddg_val)
                labels.append(mut)
                lengths.append(n_muts)
        else:
            b_list = burial_scores(pdb_file, muts, neighbor_count=neighbor_count)
            i_list = interface_scores(pdb_file, muts, sigma_interface=sigma_interface)
            if not b_list or not i_list:
                continue
            if mode == 'mean':
                b = np.mean(b_list); i = np.mean(i_list)
            elif mode == 'sum':
                b = np.sum(b_list); i = np.sum(i_list)
            else:  # max
                b = np.max(b_list); i = np.max(i_list)
            X1.append(b); X2.append(i); y.append(ddg_val)
            labels.append(";".join(muts))
            lengths.append(n_muts)

    X = np.column_stack([X1, X2])
    return X, np.array(y), labels, lengths

def main():
    parser = argparse.ArgumentParser(
        description="Predict ddG via regression from normalized scores"
    )
    parser.add_argument('-p', '--protein', required=True,
                        help='PDB code (e.g. 1A22)')
    parser.add_argument('-n', '--num_lines', type=int, default=None,
                        help='Rows after filtering')
    parser.add_argument('-m', '--mode',
                        choices=['mean','max','individual','sum'],
                        default='mean')
    parser.add_argument('-k', '--neighbors', type=int, default=9,
                        help='k for burial (default:9)')
    parser.add_argument('-s', '--sigma-interface', type=float,
                        dest='sigma_interface', default=1.0,
                        help='σ for interface (default:1.0)')
    parser.add_argument('-d', '--degree', type=int, choices=[1,2],
                        default=1,
                        help='Degree of regression: 1=linear, 2=poly')
    parser.add_argument('--distinguish', action='store_true', default=False,
                        help='Color‐code points by number of mutations (1,2,3,...)')
    parser.add_argument('--single', action='store_true', default=False,
                        help='Only include data points with exactly one mutation')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(here, 'plots_centrality_interface')
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(here, '..', 'data', 'SKEMPI2', 'SKEMPI2.csv')
    pdb_dir = os.path.join(here, '..', 'data', 'SKEMPI2', 'SKEMPI2_cache', 'wildtype')

    X, y, labels, lengths = load_data(
        csv_path, pdb_dir, args.num_lines,
        args.mode, args.neighbors,
        args.sigma_interface, args.protein
    )

    # filter to single-mutation points if requested
    if args.single:
        mask = [l == 1 for l in lengths]
        X = X[mask]
        y = y[mask]
        # labels and lengths no longer needed for plotting text
        labels = [lab for lab, m in zip(labels, mask) if m]
        lengths = [l for l in lengths if l == 1]

    if args.degree == 2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train = poly.fit_transform(X)
        terms = poly.get_feature_names_out(['burial','interface'])
    else:
        X_train = X
        terms = ['burial','interface']

    model = LinearRegression().fit(X_train, y)
    y_pred = model.predict(X_train)

    r2 = model.score(X_train, y)
    pr, _ = pearsonr(y, y_pred)
    sr, _ = spearmanr(y, y_pred)

    fig, ax = plt.subplots()
    if args.distinguish and not args.single:
        unique_counts = sorted(set(lengths))
        cmap = plt.cm.get_cmap('tab10', len(unique_counts))
        color_map = {cnt: cmap(i) for i, cnt in enumerate(unique_counts)}
        colors = [color_map[cnt] for cnt in lengths]
        for cnt in unique_counts:
            ax.scatter([], [], c=[color_map[cnt]], label=f"{cnt} muts")
        ax.legend(title="Num mutations")
        ax.scatter(y, y_pred, c=colors, alpha=0.7)
    else:
        ax.scatter(y, y_pred, alpha=0.7)

    mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlabel('True Absolute ddG')
    ax.set_ylabel('Predicted Absolute ddG')
    ax.set_title(
        f"{args.protein} | deg={args.degree}, mode={args.mode}, "
        f"k={args.neighbors}, σ={args.sigma_interface}"
    )

    # no text labels on points to reduce clutter

    metrics = f"R²={r2:.2f}\nPearson={pr:.2f}\nSpearman={sr:.2f}"
    ax.text(0.05, 0.95, metrics, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    out_fn = (
        f"ddG_reg_deg{args.degree}_{args.protein}_{args.mode}"
        f"{'_single' if args.single else ''}"
        f"_k{args.neighbors}_s{args.sigma_interface}.png"
    )
    out_path = os.path.join(plots_dir, out_fn)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
