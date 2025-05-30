#!/usr/bin/env python

"""
shift_score.py

A combined shift scoring script measuring structural changes after optimization.

Tunables:
  - CHI_DEFINITIONS: Side-chain chi angle atom tuples per residue type.
  - CONTACT_CUTOFF: Distance (Å) for defining heavy-atom contacts.
  - w_dihedral: Weight for dihedral score (CLI: --w_dihedral).
  - w_contact: Weight for contact score (CLI: --w_contact).
"""

import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.vectors import calc_dihedral, Vector

# ============================ Tunable parameters ============================
# Define which dihedrals to track per residue type
CHI_DEFINITIONS = {
    'ARG': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','NE'), ('CG','CD','NE','CZ')],
    'ASN': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
    'ASP': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
    'GLN': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'GLU': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
    'HIS': [('N','CA','CB','CG'), ('CA','CB','CG','ND1')],
    'ILE': [('N','CA','CB','CG1'), ('CA','CB','CG1','CD1')],
    'LEU': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'LYS': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','CE'), ('CG','CD','CE','NZ')],
    'MET': [('N','CA','CB','CG'), ('CA','CB','CG','SD'), ('CB','CG','SD','CE')],
    'PHE': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'PRO': [('N','CA','CB','CG'), ('CA','CB','CG','CD')],
    'SER': [('N','CA','CB','OG')],
    'THR': [('N','CA','CB','OG1')],
    'TRP': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'TYR': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
    'VAL': [('N','CA','CB','CG1')]
}
CONTACT_CUTOFF = 10  # Å; cutoff for heavy-atom contact definition
# ===========================================================================


def _get_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure('struct', pdb_path)


def _get_mutation_residue(structure, mutation):
    chain_id = mutation[1]
    res_pos = int(mutation[2:-1])
    for model in structure:
        chain = model[chain_id]
        return chain[(" ", res_pos, " ")]
    raise KeyError(f"Residue {mutation} not found in {pdb_path}")


def compute_dihedral_score(pdb_wt, pdb_opt, mutation):
    """
    Compute softmax-normalized sum of absolute chi-angle changes (°) for all side chains except the mutated residue.
    """
    wt, opt = _get_structure(pdb_wt), _get_structure(pdb_opt)
    mut_res = _get_mutation_residue(wt, mutation)
    total_delta = 0.0
    # Compare only first model
    for res_wt, res_opt in zip(wt[0].get_residues(), opt[0].get_residues()):
        if res_wt.id[0] != ' ' or res_wt == mut_res:
            continue
        name = res_wt.get_resname()
        if name not in CHI_DEFINITIONS:
            continue
        for atom_names in CHI_DEFINITIONS[name]:
            try:
                coords_wt = [Vector(res_wt[a].get_coord()) for a in atom_names]
                coords_opt = [Vector(res_opt[a].get_coord()) for a in atom_names]
                chi_wt = np.degrees(calc_dihedral(*coords_wt))
                chi_opt = np.degrees(calc_dihedral(*coords_opt))
                diff = abs(chi_opt - chi_wt) % 360
                delta = min(diff, 360 - diff)
                # ignore symmetric 180° flips in aromatics
                if name in ('PHE','TYR','HIS') and abs(delta - 180) < 1e-3:
                    continue
                total_delta += delta
            except KeyError:
                continue
    
    raw_score = total_delta/1000
    
    # Get all mutations for this protein
    all_mutations = _get_all_mutations(pdb_wt)
    all_scores = []
    
    # Get scores for all mutations
    for mut in all_mutations:
        if mut == mutation:
            all_scores.append(raw_score)
            continue
        try:
            score = compute_dihedral_score(pdb_wt, pdb_opt, mut)  # Recursive call
            all_scores.append(score)
        except (ValueError, KeyError):
            continue
    
    if all_scores:
        # Apply softmax: exp(x)/sum(exp(x))
        exp_scores = np.exp(all_scores)
        return float(exp_scores[0] / exp_scores.sum())  # Return normalized score for current mutation
    
    return raw_score


def compute_contact_score(pdb_wt, pdb_opt, mutation):
    """
    Compute softmax-normalized sum of distance changes (Å) for all gained or lost residue-residue contacts.
    """
    wt_model, opt_model = _get_structure(pdb_wt)[0], _get_structure(pdb_opt)[0]
    # heavy atoms only
    atoms_wt = [a for a in wt_model.get_atoms() if a.element != 'H']
    atoms_opt = [a for a in opt_model.get_atoms() if a.element != 'H']
    ns_wt, ns_opt = NeighborSearch(atoms_wt), NeighborSearch(atoms_opt)
    def res_id(atom): return (atom.get_parent().get_parent().id, atom.get_parent().id[1])
    contacts = {}
    for label, atoms, ns in [('wt', atoms_wt, ns_wt), ('opt', atoms_opt, ns_opt)]:
        contacts[label] = set()
        for atom in atoms:
            for nbr in ns.search(atom.get_coord(), CONTACT_CUTOFF):
                if nbr is atom: continue
                id1, id2 = res_id(atom), res_id(nbr)
                if id1 == id2: continue
                contacts[label].add(tuple(sorted((id1, id2))))
    gained = contacts['opt'] - contacts['wt']
    lost   = contacts['wt']  - contacts['opt']
    score = 0.0
    # weight by change in minimal inter-atomic distance
    for pair in gained.union(lost):
        pts_wt = [a.get_coord() for a in atoms_wt if res_id(a) in pair]
        pts_opt= [a.get_coord() for a in atoms_opt if res_id(a) in pair]
        d_wt = min(np.linalg.norm(a-b) for a in pts_wt for b in pts_wt if not np.allclose(a,b))
        d_opt= min(np.linalg.norm(a-b) for a in pts_opt for b in pts_opt if not np.allclose(a,b))
        score += abs(d_opt - d_wt)
    
    raw_score = 1000 * score
    
    # Get all mutations for this protein
    all_mutations = _get_all_mutations(pdb_wt)
    all_scores = []
    
    # Get scores for all mutations
    for mut in all_mutations:
        if mut == mutation:
            all_scores.append(raw_score)
            continue
        try:
            score = compute_contact_score(pdb_wt, pdb_opt, mut)  # Recursive call
            all_scores.append(score)
        except (ValueError, KeyError):
            continue
    
    if all_scores:
        # Apply softmax: exp(x)/sum(exp(x))
        exp_scores = np.exp(all_scores)
        return float(exp_scores[0] / exp_scores.sum())  # Return normalized score for current mutation
    
    return raw_score


def combined_shift_score(pdb_wt: str,
                         pdb_opt: str,
                         mutation: str,
                         w_dihedral: float = 1.0,
                         w_contact: float = 2.0) -> float:
    """
    Return weighted sum of softmax-normalized dihedral and contact scores.
    """
    d = compute_dihedral_score(pdb_wt, pdb_opt, mutation)
    c = compute_contact_score(pdb_wt, pdb_opt, mutation)
    return w_dihedral * d + w_contact * c


def combined_shift_scores(pdb_wt: str,
                          pdb_opt: str,
                          mutations: list,
                          w_dihedral: float = 1.0,
                          w_contact: float = 2.0) -> list:
    """
    Compute combined shift scores for multiple mutations.
    """
    return [combined_shift_score(pdb_wt, pdb_opt, m, w_dihedral, w_contact)
            for m in mutations]


def shift_score(pdb_wt: str, pdb_opt: str, mutation: str, sigma_shift: float = 10.0, min_delta: float = 0.4, temperature: float = 1.0) -> float:
    """
    Compute the Gaussian-weighted mean displacement of all atoms (excluding the mutated residue)
    between wildtype and optimized structures, normalized by a softmax centered on the mean.

    Parameters:
        pdb_wt: Path to the wildtype PDB file.
        pdb_opt: Path to the optimized PDB file.
        mutation: Mutation string in format '<orig><chain><resnum><new>' e.g. 'RA88A'.
        sigma_shift: Gaussian width (Angstroms) for weighting by distance from mutation site.
        min_delta: Minimum atomic displacement (Å) to consider. Atoms moving less than this are ignored.
        temperature: Temperature parameter for softmax normalization (higher = softer).

    Returns:
        Normalized weighted mean displacement (float).
    """
    coords_wt, mut_center = _get_atom_coords_and_mut_center(pdb_wt, mutation)
    coords_opt, _ = _get_atom_coords_and_mut_center(pdb_opt, mutation)

    # ensure matching atoms
    keys = set(coords_wt.keys()) & set(coords_opt.keys())
    if not keys:
        raise ValueError("No matching atoms found between wildtype and optimized PDBs.")

    deltas = []
    weights = []
    for key in keys:
        c_wt = coords_wt[key]
        c_opt = coords_opt[key]
        delta = np.linalg.norm(c_opt - c_wt)
        
        # Skip atoms with displacement less than min_delta
        if delta < min_delta:
            continue
            
        # distance from mutation center
        d = np.linalg.norm(c_wt - mut_center)
        w = np.exp(- (d ** 2) / (2 * sigma_shift ** 2))
        deltas.append(delta)
        weights.append(w)

    deltas = np.array(deltas)
    weights = np.array(weights)

    if len(deltas) == 0 or weights.sum() == 0:
        return 0.0

    # Calculate weighted mean displacement
    raw_score = float(np.dot(deltas, weights) / weights.sum())

    # Get all mutations for this protein to compute mean
    protein = mutation[1:]  # Get protein code from mutation
    all_mutations = []
    for mut in _get_all_mutations(pdb_wt):  # You'll need to implement this function
        if mut == mutation:
            continue
        try:
            score = shift_score(pdb_wt, pdb_opt, mut, sigma_shift, min_delta, temperature=None)  # Recursive call without normalization
            all_mutations.append(score)
        except (ValueError, KeyError):
            continue

    if not all_mutations:
        return raw_score

    # Compute mean and standard deviation of all mutation scores
    mean_score = np.mean(all_mutations)
    std_score = np.std(all_mutations) if len(all_mutations) > 1 else 1.0

    # Normalize using softmax centered on mean
    normalized_score = (raw_score - mean_score) / (std_score * temperature)
    return float(1.0 / (1.0 + np.exp(-normalized_score)))


def _get_all_mutations(pdb_path: str) -> list:
    """
    Get all possible single-point mutations for a protein.
    Returns list of mutation strings in format '<orig><chain><resnum><new>'.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_path)
    mutations = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if residue.id[0] != ' ':  # Skip hetero atoms
                    continue
                res_name = residue.get_resname()
                res_pos = residue.id[1]
                # For each residue, create a mutation to each other amino acid
                for new_aa in 'ACDEFGHIKLMNPQRSTVWY':
                    if new_aa != res_name[0]:  # Skip if same as original
                        mutations.append(f"{res_name[0]}{chain_id}{res_pos}{new_aa}")
    
    return mutations


def shift_scores(pdb_wt: str, pdb_opt: str, mutations: list, sigma_shift: float = 30.0, min_delta: float = 0.4, temperature: float = 1.0) -> list:
    """
    Compute shift_score for multiple mutations.

    Parameters:
        pdb_wt: Path to the wildtype PDB file.
        pdb_opt: Path to the optimized PDB file.
        mutations: List of mutation strings.
        sigma_shift: Gaussian width for weighting.
        min_delta: Minimum atomic displacement (Å) to consider.
        temperature: Temperature parameter for softmax normalization.

    Returns:
        List of normalized weighted mean displacements.
    """
    return [shift_score(pdb_wt, pdb_opt, mut, sigma_shift, min_delta, temperature) for mut in mutations]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compute dihedral, contact, and combined scores for mutations.")
    p.add_argument("wildtype_pdb", nargs='?', default="data/SKEMPI2/SKEMPI2_cache/wildtype/174_1BRS.pdb",
                   help="Path to wildtype PDB file")
    p.add_argument("optimized_pdb", nargs='?', default="data/SKEMPI2/SKEMPI2_cache/optimized/174_1BRS.pdb",
                   help="Path to optimized PDB file")
    p.add_argument("mutations", nargs='*', default=["HA100L"],
                   help="Mutation strings, e.g. HA100L")
    p.add_argument("--w_dihedral", type=float, default=1.0,
                   help="Weight for dihedral score")
    p.add_argument("--w_contact", type=float, default=2.0,
                   help="Weight for contact score")
    args = p.parse_args()
    if not args.mutations:
        p.error("At least one mutation must be provided")
    for m in args.mutations:
        d_score = compute_dihedral_score(args.wildtype_pdb, args.optimized_pdb, m)
        c_score = compute_contact_score(args.wildtype_pdb, args.optimized_pdb, m)
        combined = args.w_dihedral * d_score + args.w_contact * c_score
        print(f"{m}\tDihedral: {d_score:.3f}\tContact: {c_score:.3f}\tCombined: {combined:.3f}")
