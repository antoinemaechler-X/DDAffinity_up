#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb, get_fa_scorefxn
from pyrosetta.toolbox import mutate_residue
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from tqdm import tqdm

def parse_mutation(mstr):
    m = re.match(r"^([A-Z])([A-Za-z0-9])(\d+)([A-Z])$", mstr)
    if not m:
        raise ValueError(f"Bad mutation format '{mstr}'")
    return m.groups()  # orig, chain, resnum, new

def find_residue(pose, chain, resnum):
    pb = pose.pdb_info()
    for i in range(1, pose.total_residue()+1):
        if pb.chain(i)==chain and pb.number(i)==resnum:
            return i
    return None

def process_row(idx, row, pdbs_dir, out_dir, mode):
    entry_id = row['#Pdb']            # e.g. "0_1CSE"
    origin   = row['#Pdb_origin']     # e.g. "1CSE"
    out_path = os.path.join(out_dir, f"{entry_id}.pdb")

    # skip if already done
    if os.path.isfile(out_path):
        print(f"[{idx}] SKIP {entry_id} (exists)")
        return

    wt_path = os.path.join(pdbs_dir, f"{origin}.pdb")
    if not os.path.isfile(wt_path):
        print(f"[{idx}] ERROR: missing WT {wt_path}")
        return

    # load structure
    pose = pose_from_pdb(wt_path)

    # optional mutate step
    if mode == "optimized":
        orig, chain, num_str, new = parse_mutation(row['Mutation(s)_cleaned'])
        seqpos = find_residue(pose, chain, int(num_str))
        if seqpos is None:
            print(f"[{idx}] ERROR: can't map {chain}{num_str} in {origin}")
            return
        mutate_residue(pose, seqpos, new, pack_radius=8.0)

    # common minimization
    scorefxn = get_fa_scorefxn()
    mm = MoveMap(); mm.set_bb(True); mm.set_chi(True)
    mvm = MinMover()
    mvm.movemap(mm)
    mvm.score_function(scorefxn)
    mvm.min_type("lbfgs_armijo_nonmonotone")
    mvm.apply(pose)

    # write out
    pose.dump_pdb(out_path)
    print(f"[{idx}] {mode.upper()} → {entry_id}.pdb")

def main():
    pyrosetta.init(extra_options="-mute all")

    p = argparse.ArgumentParser()
    p.add_argument("--csv",    required=True, help="SKEMPI2.csv")
    p.add_argument("--pdbs",   required=True, help="WT PDB dir")
    p.add_argument("--outdir", required=True, help="Output dir")
    p.add_argument(
      "--mode",
      choices=["optimized","wildtype"],
      default="optimized",
      help="optimized: mutate+minimize; wildtype: only minimize"
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.mode}"):
        try:
            process_row(idx, row, args.pdbs, args.outdir, args.mode)
        except Exception as e:
            print(f"[{idx}] EXCEPTION: {e}")

if __name__=="__main__":
    main()
