#!/usr/bin/env python3
"""
compare_coords.py

Compare 3D coordinates in two PDB files line by line.  
For any matching atom (by chain, residue number, atom name), report
if the X, Y, or Z coordinate differs by even the tiniest amount.

Usage:
  python compare_coords.py <wt.pdb> <mt.pdb>
"""
import sys

def parse_pdb_coords(path):
    coords = {}
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                # PDB columns: 
                #  1-6 record name, 7-11 serial, 13-16 atom name, 17 altLoc,
                #  18-20 resName, 22 chainID, 23-26 resSeq, 27 iCode,
                #  31-38 x, 39-46 y, 47-54 z
                atom_name = line[12:16].strip()
                chain_id  = line[21].strip()
                res_seq   = line[22:26].strip()
                i_code    = line[26].strip()
                key = (chain_id, res_seq, i_code, atom_name)
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue  # skip malformed
                coords[key] = (x, y, z)
    return coords

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    wt_path, mt_path = sys.argv[1], sys.argv[2]

    wt = parse_pdb_coords(wt_path)
    mt = parse_pdb_coords(mt_path)

    all_keys = set(wt) & set(mt)
    diffs = []

    for key in sorted(all_keys):
        x1,y1,z1 = wt[key]
        x2,y2,z2 = mt[key]
        if (x1!=x2) or (y1!=y2) or (z1!=z2):
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            mag = (dx*dx + dy*dy + dz*dz)**0.5
            diffs.append((key, (x1,y1,z1), (x2,y2,z2), mag))

    # also catch atoms missing in one file
    only_wt = set(wt) - set(mt)
    only_mt = set(mt) - set(wt)

    if not diffs and not only_wt and not only_mt:
        print("✅ No coordinate differences detected between the two files.")
        return

    if diffs:
        print("Coordinate differences found:")
        for (chain, res, icode, atom), (x1,y1,z1), (x2,y2,z2), mag in diffs:
            print(f" - {chain}{res}{icode or ''} {atom}:")
            print(f"     WT  = ({x1:.6f}, {y1:.6f}, {z1:.6f})")
            print(f"     MUT = ({x2:.6f}, {y2:.6f}, {z2:.6f})")
            print(f"     Δ   = {mag:.6f} Å")
    if only_wt:
        print("\nAtoms in WT but missing in MT:")
        for key in sorted(only_wt):
            chain, res, icode, atom = key
            print(f" - {chain}{res}{icode or ''} {atom}")
    if only_mt:
        print("\nAtoms in MT but missing in WT:")
        for key in sorted(only_mt):
            chain, res, icode, atom = key
            print(f" - {chain}{res}{icode or ''} {atom}")

if __name__ == '__main__':
    main()
