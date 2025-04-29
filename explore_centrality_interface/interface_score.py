import numpy as np
from Bio.PDB import PDBParser

def calc_virtual_cb(n_coord: np.ndarray, ca_coord: np.ndarray, c_coord: np.ndarray) -> np.ndarray:
    """
    Calculate pseudo C-beta coordinates for glycine residues.

    Approximates the position of the Cβ by placing it 1.522 Å from the Cα
    in the bisector direction of the N-Cα-C plane.
    """
    v_n = n_coord - ca_coord
    v_c = c_coord - ca_coord
    v_n /= np.linalg.norm(v_n)
    v_c /= np.linalg.norm(v_c)
    bisec = v_n + v_c
    bisec /= np.linalg.norm(bisec)
    length = 1.522
    return ca_coord + bisec * length

def _get_cb_coordinates(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_path)
    model = next(structure.get_models())
    cb_coords = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            if 'CA' not in residue or 'N' not in residue or 'C' not in residue:
                continue
            ca = residue['CA'].get_coord()
            if 'CB' in residue:
                cb = residue['CB'].get_coord()
            else:
                cb = calc_virtual_cb(residue['N'].get_coord(), ca, residue['C'].get_coord())
            key = (chain.id, residue.id[1], residue.id[2])
            cb_coords[key] = cb
    keys = list(cb_coords.keys())
    coords = np.vstack([cb_coords[k] for k in keys])
    return cb_coords, keys, coords

def _compute_interface_score_for_mutation(cb_coords, keys, coords, mutation: str, sigma: float) -> float:
    """
    Compute interface score using Gaussian-weighted mean over the 20 nearest neighbors.

    Parameters:
    -------------
    sigma : float
        Standard deviation for Gaussian weighting (Å).
    """
    # parse mutation: wtAA(0), chainID(1), resNum(2:-1), mtAA(-1)
    if len(mutation) < 4:
        raise ValueError(f"Mutation string '{mutation}' too short.")
    chain_id = mutation[1]
    try:
        pos = int(mutation[2:-1])
    except ValueError:
        raise ValueError(f"Cannot parse residue number from '{mutation}'.")
    # allow any insertion code
    candidates = [k for k in cb_coords if k[0] == chain_id and k[1] == pos]
    if not candidates:
        raise KeyError(f"Residue {chain_id}{pos} not found in PDB.")
    mut_key = candidates[0]
    mut_cb = cb_coords[mut_key]
    # distances from mutation to all Cβ
    dists_mut = np.linalg.norm(coords - mut_cb, axis=1)
    # select 20 closest neighbors (including self)
    idx_sorted = np.argsort(dists_mut)[:20]
    # compute Gaussian weights
    weights = np.exp(- (dists_mut[idx_sorted] ** 2) / (2 * sigma ** 2))
    # for each neighbor, compute count of inter-chain residues within 10 Å
    counts = []
    for idx in idx_sorted:
        # distances from neighbor to all
        d_all = np.linalg.norm(coords - coords[idx], axis=1)
        count = sum(
            1 for j, d in enumerate(d_all)
            if j != idx and d < 10.0 and keys[j][0] != chain_id
        )
        counts.append(count)
    counts = np.array(counts, dtype=float)
    # return weighted average
    if np.sum(weights) == 0:
        return 0.0
    return float(np.dot(weights, counts) / np.sum(weights))

def interface_score(pdb_path: str, mutation: str, sigma: float = 3.0) -> float:
    """
    Compute Gaussian-weighted interface score for a single mutation.

    sigma: standard deviation for Gaussian weights (Å).
    """
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)
    return _compute_interface_score_for_mutation(cb_coords, keys, coords, mutation, sigma)

def interface_scores(pdb_path: str, mutations: list, sigma: float = 3.0) -> list:
    """
    Compute Gaussian-weighted interface scores for multiple mutations.

    Returns a list of floats in same order as mutations.
    """
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)
    return [
        _compute_interface_score_for_mutation(cb_coords, keys, coords, mut, sigma)
        for mut in mutations
    ]


#print(interface_scores("data/SKEMPI2/M1340_cache/wildtype/0_1A22.pdb", ["KA161A","FA165A", "KB80A", "EB79A"], sigma=3))