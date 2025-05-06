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

def get_first_N_proteins(csv_path, N):
    """
    Return the first N unique protein codes from '#Pdb_origin' in the CSV.
    """
    df = pd.read_csv(csv_path)
    return df['#Pdb_origin'].dropna().unique().tolist()[:N]

def regress_and_plot_multiple(N, csv_path, pdb_dir,
                              mode='mean',
                              neighbors=9, sigma_interface=1.0,
                              degree=1, distinguish=False, single=False):
    """
    Run regression on combined data from the first N proteins and plot real vs. predicted,
    with R², Pearson and Spearman shown in a box.
    If distinguish=True, color‐code points by number of mutations.
    If single=True, only include points with exactly one mutation.
    """
    proteins = get_first_N_proteins(csv_path, N)
    X1, X2, y_list, lengths = [], [], [], []

    for p in proteins:
        df = pd.read_csv(csv_path)
        df = df[df['#Pdb_origin'] == p]
        for _, row in df.iterrows():
            pdb_file = os.path.join(pdb_dir, f"{row['#Pdb']}.pdb")
            muts = row['Mutation(s)_cleaned'].strip('"').split(',')
            n_muts = len(muts)
            try:
                ddg = abs(float(row['ddG']))
            except:
                continue

            b_list = burial_scores(pdb_file, muts, neighbor_count=neighbors)
            i_list = interface_scores(pdb_file, muts, sigma_interface=sigma_interface)

            if mode == 'mean':
                b, i = np.mean(b_list), np.mean(i_list)
            elif mode == 'sum':
                b, i = np.sum(b_list), np.sum(i_list)
            else:  # max
                b, i = np.max(b_list), np.max(i_list)

            X1.append(b)
            X2.append(i)
            y_list.append(ddg)
            lengths.append(n_muts)

    # convert to arrays
    X = np.column_stack([X1, X2])
    y = np.array(y_list)
    lengths = np.array(lengths)

    # filter to single-mutation points if requested
    if single:
        mask = (lengths == 1)
        X = X[mask]
        y = y[mask]
        lengths = lengths[mask]
        # once filtered, distinguish is moot because all lengths==1
        distinguish = False

    # polynomial transform if needed
    if degree == 2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X)

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    r2 = model.score(X, y)
    pr, _ = pearsonr(y, y_pred)
    sr, _ = spearmanr(y, y_pred)

    fig, ax = plt.subplots()
    if distinguish:
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
        f'{N} proteins: mode={mode}, k={neighbors}, '
        f'σ_interface={sigma_interface}, deg={degree}'
    )

    metrics = f"R²={r2:.2f}\nPearson={pr:.2f}\nSpearman={sr:.2f}"
    ax.text(0.05, 0.95, metrics, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    out_dir = os.path.join(os.path.dirname(__file__), 'plots_centrality_interface')
    os.makedirs(out_dir, exist_ok=True)
    single_tag = '_single' if single else ''
    fn = f'Regression_{N}_prot_deg{degree}_k{neighbors}_s{sigma_interface}{single_tag}.png'
    out_path = os.path.join(out_dir, fn)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run regression across multiple proteins and plot"
    )
    parser.add_argument('-N', '--num_proteins', type=int, default=30,
                        help='Number of proteins to include (default: 30)')
    parser.add_argument('-m', '--mode',
                        choices=['mean', 'sum', 'max'], default='mean',
                        help="Aggregation mode (default: mean)")
    parser.add_argument('-k', '--neighbors', type=int, default=9,
                        help='k for burial score (default: 9)')
    parser.add_argument('-s', '--sigma-interface', dest='sigma_interface',
                        type=float, default=1.0,
                        help='σ for interface Gaussian (default: 1.0)')
    parser.add_argument('-d', '--degree', type=int, choices=[1, 2],
                        default=1, help='Regression degree (1 or 2)')
    parser.add_argument('--distinguish', action='store_true', default=False,
                        help='Color‐code points by number of mutations')
    parser.add_argument('--single', action='store_true', default=False,
                        help='Only include single‐mutation points in regression')
    parser.add_argument('--csv', type=str,
                        default='data/SKEMPI2/M1340.csv',
                        help='Path to CSV (default: data/SKEMPI2/M1340.csv)')
    parser.add_argument('--pdb_dir', type=str,
                        default='data/SKEMPI2/M1340_cache/wildtype',
                        help='Path to PDB directory')
    args = parser.parse_args()

    regress_and_plot_multiple(
        args.num_proteins,
        args.csv,
        args.pdb_dir,
        mode=args.mode,
        neighbors=args.neighbors,
        sigma_interface=args.sigma_interface,
        degree=args.degree,
        distinguish=args.distinguish,
        single=args.single
    )

if __name__ == '__main__':
    main()


#python explore_centrality_interface/multi_tests.py --num_proteins 10 --mode mean --neighbors 9 --sigma-interface 1.0 --degree 2 --single --distinguish --csv data/SKEMPI2/SKEMPI2.csv --pdb_dir data/SKEMPI2/SKEMPI2_cache/wildtype