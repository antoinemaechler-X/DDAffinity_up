#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from burial_score import burial_score, burial_scores
from interface_score import interface_score, interface_scores

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ddG vs normalized burial & interface scores"
    )
    parser.add_argument("-p", "--protein", type=str, required=True,
                        help="PDB code to filter (e.g. 1A22)")
    parser.add_argument("-n", "--num_lines", type=int, default=None,
                        help="Rows to process after filtering")
    parser.add_argument("-m", "--mode",
                        choices=["mean", "max", "individual", "sum"],
                        default="mean",
                        help="Aggregation mode")
    parser.add_argument("-k", "--neighbors", type=int, default=9,
                        help="k for burial (default: 9)")
    parser.add_argument("-s", "--sigma-interface", type=float,
                        dest="sigma_interface", default=1.0,
                        help="σ for interface Gaussian (default: 1.0)")
    parser.add_argument("--distinguish", action="store_true", default=False,
                        help="Color‐code points by number of mutations (1,2,3,...)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(here, 'plots_centrality_interface')
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(here, '..', 'data', 'SKEMPI2', 'SKEMPI2.csv')
    pdb_dir = os.path.join(here, '..', 'data', 'SKEMPI2', 'SKEMPI2_cache', 'wildtype')

    df = pd.read_csv(csv_path)
    df = df[df['#Pdb_origin'] == args.protein]
    if args.num_lines is not None:
        df = df.iloc[:args.num_lines]

    ddG_list = []
    burial_list = []
    interface_list = []
    labels = []
    lengths = []

    for idx, row in df.iterrows():
        pdb_file = os.path.join(pdb_dir, f"{row['#Pdb']}.pdb")
        muts = row['Mutation(s)_cleaned'].strip('"').split(',')
        n_muts = len(muts)
        try:
            ddg = abs(float(row['ddG']))
        except:
            ddg = float('nan')

        if args.mode == 'individual':
            for mut in muts:
                try:
                    b = burial_score(pdb_file, mut, neighbor_count=args.neighbors)
                    i = interface_score(pdb_file, mut,
                                        sigma_interface=args.sigma_interface)
                except Exception as e:
                    print(f"[Warn] row {idx}, mut {mut}: {e}")
                    b, i = float('nan'), float('nan')
                burial_list.append(b)
                interface_list.append(i)
                ddG_list.append(ddg)
                labels.append(mut)
                lengths.append(n_muts)
        else:
            try:
                b_scores = burial_scores(pdb_file, muts,
                                         neighbor_count=args.neighbors)
                i_scores = interface_scores(pdb_file, muts,
                                            sigma_interface=args.sigma_interface)
                if args.mode == 'mean':
                    b = sum(b_scores) / len(b_scores)
                    i = sum(i_scores) / len(i_scores)
                elif args.mode == 'sum':
                    b = sum(b_scores)
                    i = sum(i_scores)
                else:  # max
                    b = max(b_scores)
                    i = max(i_scores)
            except Exception as e:
                print(f"[Warn] row {idx}: {e}")
                b, i = float('nan'), float('nan')
            burial_list.append(b)
            interface_list.append(i)
            ddG_list.append(ddg)
            labels.append(";".join(muts))
            lengths.append(n_muts)

    def make_scatter(x, y, labels, lengths, xlabel, ylabel, title, fn):
        plt.figure()
        if args.distinguish:
            # assign a color to each unique mutation count
            unique_counts = sorted(set(lengths))
            cmap = plt.cm.get_cmap('tab10', len(unique_counts))
            color_map = {cnt: cmap(i) for i, cnt in enumerate(unique_counts)}
            colors = [color_map[cnt] for cnt in lengths]
            # plot dummy points for legend
            for cnt in unique_counts:
                plt.scatter([], [], c=[color_map[cnt]], label=f"{cnt} muts")
            plt.legend(title="Num mutations")
            # actual scatter
            plt.scatter(x, y, c=colors, alpha=0.7)
        else:
            plt.scatter(x, y, alpha=0.7)

        for xi, yi, lab in zip(x, y, labels):
            plt.text(xi, yi, lab, fontsize=6, alpha=0.7)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out = os.path.join(plots_dir, fn)
        plt.savefig(out)
        print(f"Saved {out}")

    mode_label = args.mode.capitalize()
    make_scatter(
        burial_list, ddG_list, labels, lengths,
        'Normalized Burial', 'Absolute ddG',
        f'ddG vs Burial ({mode_label}, k={args.neighbors})',
        f'ddG_vs_burial_{args.protein}_{args.mode}_k{args.neighbors}.png'
    )
    make_scatter(
        interface_list, ddG_list, labels, lengths,
        'Normalized Interface', 'Absolute ddG',
        f'ddG vs Interface ({mode_label}, σ={args.sigma_interface})',
        f'ddG_vs_interface_{args.protein}_{args.mode}_s{args.sigma_interface}.png'
    )

if __name__ == '__main__':
    main()
