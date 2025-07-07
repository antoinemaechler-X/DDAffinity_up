#!/usr/bin/env python3
import argparse
import pandas as pd
import os

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter SKEMPI2.csv to only mutations determined *exclusively* by SPR."
    )
    p.add_argument(
        "--skempi",
        default="data/SKEMPI2/SKEMPI2.csv",
        help="Original SKEMPI2 CSV (comma-separated)."
    )
    p.add_argument(
        "--full",
        default="data/SKEMPI2/full_SKEMPI2.csv",
        help="Full SKEMPI2 CSV with Method column (semicolon-separated)."
    )
    p.add_argument(
        "--out",
        default="data/SKEMPI2/SKEMPI2_SPR_exclusive.csv",
        help="Output path for filtered CSV."
    )
    p.add_argument(
        "--pdb-wt-dir",
        default=None,
        help="(Optional) Directory containing wild-type PDB files."
    )
    p.add_argument(
        "--pdb-mt-dir",
        default=None,
        help="(Optional) Directory containing mutant PDB files."
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) load original SKEMPI2 (comma‐sep) and full SKEMPI2 (semicolon‐sep)
    df_orig = pd.read_csv(args.skempi, sep=",", dtype=str)
    df_full = pd.read_csv(args.full, sep=";", dtype=str)

    # 2) build join_key in both
    df_orig["join_key"] = (
        df_orig["#Pdb_origin"].str.upper() + "_" +
        df_orig["Partner1"].str.upper()   + "_" +
        df_orig["Partner2"].str.upper()
    )
    df_full["join_key"] = df_full["#Pdb"].str.upper()

    # 3) group df_full by (join_key, Mutation(s)_cleaned); keep only groups whose METHOD‐set == {'SPR'}
    grouped = df_full.groupby(["join_key", "Mutation(s)_cleaned"])
    def is_exclusive_spr(subdf):
        methods = set(subdf["Method"].dropna().astype(str).str.upper())
        return methods == {"SPR"}

    mask_excl = grouped.filter(is_exclusive_spr)
    exclusive_pairs = set(zip(
        mask_excl["join_key"],
        mask_excl["Mutation(s)_cleaned"]
    ))

    # 4) filter df_orig by membership in exclusive_pairs
    df_spr = df_orig[df_orig.apply(
        lambda row: (row["join_key"], row["Mutation(s)_cleaned"]) in exclusive_pairs,
        axis=1
    )].copy()

    if df_spr.empty:
        raise RuntimeError("No mutations found that are exclusively SPR.")

    # 5) optional: drop any whose PDB files don’t exist
    if args.pdb_wt_dir and args.pdb_mt_dir:
        keep = []
        dropped = []
        for idx, row in df_spr.iterrows():
            pdb_id = row["#Pdb"]
            wt_path = os.path.join(args.pdb_wt_dir, pdb_id + ".pdb")
            mt_path = os.path.join(args.pdb_mt_dir, pdb_id + ".pdb")
            if os.path.isfile(wt_path) and os.path.isfile(mt_path):
                keep.append(idx)
            else:
                dropped.append(pdb_id)
        if dropped:
            print(f"Warning: dropping {len(dropped)} entries due to missing PDB files")
            for pdb in dropped[:10]:
                print("  ", pdb)
            if len(dropped) > 10:
                print("  ...")
        df_spr = df_spr.loc[keep]

    # 6) write out using the same columns as original SKEMPI2.csv
    out_cols = pd.read_csv(args.skempi, nrows=0).columns.tolist()
    df_spr[out_cols].to_csv(args.out, index=False)
    print(f"Wrote {len(df_spr)} exclusive‐SPR entries to {args.out}")

if __name__ == "__main__":
    main()
