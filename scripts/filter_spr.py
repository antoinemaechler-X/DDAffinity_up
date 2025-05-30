#!/usr/bin/env python3
import argparse
import pandas as pd
import os

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter SKEMPI2.csv to only SPR‐measured entries (preserve ddG)."
    )
    p.add_argument("--skempi",       default="data/SKEMPI2/SKEMPI2.csv")
    p.add_argument("--full",         default="data/SKEMPI2/full_SKEMPI2.csv")
    p.add_argument("--out",          default="data/SKEMPI2/SKEMPI2_SPR.csv")
    p.add_argument("--pdb-wt-dir",   default=None)
    p.add_argument("--pdb-mt-dir",   default=None)
    return p.parse_args()

def main():
    args = parse_args()

    # 1) load
    df_orig = pd.read_csv(args.skempi, sep=",", dtype=str)
    df_full = pd.read_csv(args.full, sep=";", dtype=str)

    # 2) build join keys
    df_orig["join_key"] = (
        df_orig["#Pdb_origin"].str.upper() + "_" +
        df_orig["Partner1"].str.upper()   + "_" +
        df_orig["Partner2"].str.upper()
    )
    df_full["join_key"] = df_full["#Pdb"].str.upper()

    # 3) build set of (join_key, mutation) for SPR
    spr_full = df_full[df_full["Method"] == "SPR"]
    spr_pairs = set(zip(
        spr_full["join_key"],
        spr_full["Mutation(s)_cleaned"]
    ))

    # 4) filter original by membership
    mask = df_orig.apply(
        lambda row: (row["join_key"], row["Mutation(s)_cleaned"]) in spr_pairs,
        axis=1
    )
    df_spr = df_orig[mask].copy()
    if df_spr.empty:
        raise RuntimeError("No SPR entries found — check your Method or join keys")

    # 5) optional PDB‐existence check
    if args.pdb_wt_dir and args.pdb_mt_dir:
        keep = []
        dropped = []
        for idx, row in df_spr.iterrows():
            pdb_id = row["#Pdb"]         # e.g. "0_1CSE"
            wt = os.path.join(args.pdb_wt_dir, pdb_id + ".pdb")
            mt = os.path.join(args.pdb_mt_dir, pdb_id + ".pdb")
            if os.path.isfile(wt) and os.path.isfile(mt):
                keep.append(idx)
            else:
                dropped.append(pdb_id)
        if dropped:
            print(f"Warning: dropping {len(dropped)} entries missing PDB files")
        df_spr = df_spr.loc[keep]

    # 6) write out exactly same columns as original
    out_cols = pd.read_csv(args.skempi, nrows=0).columns.tolist()
    df_spr[out_cols].to_csv(args.out, index=False)
    print(f"Wrote {len(df_spr)} SPR entries to {args.out}")

if __name__ == "__main__":
    main()
