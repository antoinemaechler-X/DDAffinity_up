#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# import scoring modules
from burial_score import burial_score, burial_scores
from interface_score import interface_score, interface_scores


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ddG vs burial and Gaussian-weighted interface scores with mutation labels"
    )
    parser.add_argument(
        "-p", "--protein",
        type=str,
        required=True,
        help="PDB code to filter (e.g. 1A22)"
    )
    parser.add_argument(
        "-n", "--num_lines",
        type=int,
        default=None,
        help="Number of CSV rows to process after filtering (default: all)"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["mean", "max", "individual", "sum"],
        default="mean",
        help=(
            "Aggregation mode for multiple mutations: 'mean' (default),"
            " 'max', 'sum', or 'individual' (one point per mutation)"
        )
    )
    parser.add_argument(
        "-k", "--neighbors",
        type=int,
        default=5,
        help="Number of neighbors considered for burial score (default: 5)"
    )
    parser.add_argument(
        "-s", "--sigma",
        type=float,
        default=5.0,
        help="Standard deviation (Å) for Gaussian weighting in interface score (default: 5.0)"
    )
    args = parser.parse_args()

    # determine paths relative to this script
    here = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(here, 'plots_centrality_interface')
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(here, '..', 'data', 'SKEMPI2', 'M1340.csv')
    pdb_dir = os.path.join(here, '..', 'data', 'SKEMPI2', 'M1340_cache', 'wildtype')

    # read and filter data by protein code
    df = pd.read_csv(csv_path)
    df = df[df['#Pdb_origin'] == args.protein]
    if args.num_lines is not None:
        df = df.iloc[:args.num_lines]

    ddG_list = []
    burial_list = []
    interface_list = []
    labels = []

    for idx, row in df.iterrows():
        pdb_id = row['#Pdb']
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        muts = row['Mutation(s)_cleaned'].strip('"').split(',')
        # absolute ddG
        try:
            ddg = abs(float(row['ddG']))
        except Exception:
            ddg = float('nan')

        if args.mode == 'individual':
            for mut in muts:
                try:
                    b = burial_score(pdb_file, mut, neighbor_count=args.neighbors)
                    i = interface_score(pdb_file, mut, sigma=args.sigma)
                except Exception as e:
                    print(f"[Warning] row {idx}, mutation {mut}: {e}")
                    b, i = float('nan'), float('nan')
                burial_list.append(b)
                interface_list.append(i)
                ddG_list.append(ddg)
                labels.append(mut)
        else:
            try:
                b_scores = burial_scores(pdb_file, muts, neighbor_count=args.neighbors)
                i_scores = interface_scores(pdb_file, muts, sigma=args.sigma)
                if args.mode == 'mean':
                    b = sum(b_scores) / len(b_scores) if b_scores else float('nan')
                    i = sum(i_scores) / len(i_scores) if i_scores else float('nan')
                elif args.mode == 'sum':
                    b = sum(b_scores) if b_scores else float('nan')
                    i = sum(i_scores) if i_scores else float('nan')
                else:  # max
                    b = max(b_scores) if b_scores else float('nan')
                    i = max(i_scores) if i_scores else float('nan')
            except Exception as e:
                print(f"[Warning] row {idx} (PDB {pdb_id}): {e}")
                b, i = float('nan'), float('nan')
            burial_list.append(b)
            interface_list.append(i)
            ddG_list.append(ddg)
            labels.append(";".join(muts))

    # plotting function with labels
    def make_scatter(x, y, labels, xlabel, ylabel, title, filename):
        plt.figure()
        plt.scatter(x, y)
        for xi, yi, lab in zip(x, y, labels):
            plt.text(xi, yi, lab, fontsize=6, alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        filepath = os.path.join(plots_dir, filename)

        plt.savefig(filepath)
        print(f"Saved plot {filepath}")

    mode_label = args.mode.capitalize()
    # ddG vs burial
    make_scatter(
        burial_list,
        ddG_list,
        labels,
        'Burial Score',
        'Absolute ddG',
        f'Absolute ddG vs Burial Score ({mode_label}, k={args.neighbors}, Protein={args.protein})',
        f'ddG_vs_burial_{args.protein}_{args.mode}_k{args.neighbors}_labeled.png'
    )
    # ddG vs interface
    make_scatter(
        interface_list,
        ddG_list,
        labels,
        'Gaussian Interface Score',
        'Absolute ddG',
        f'Absolute ddG vs Gaussian Interface Score ({mode_label}, σ={args.sigma}, Protein={args.protein})',
        f'ddG_vs_interface_{args.protein}_{args.mode}_sigma{args.sigma}_labeled.png'
    )

if __name__ == '__main__':
    main()
