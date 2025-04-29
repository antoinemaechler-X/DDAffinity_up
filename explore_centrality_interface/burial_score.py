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

def _compute_centrality(idx: int, coords: np.ndarray) -> int:
    dists = np.linalg.norm(coords - coords[idx], axis=1)
    dists = np.delete(dists, idx)
    return int(np.sum(dists < 10.0))

def _compute_score_for_mutation(cb_coords, keys, coords, mutation: str, neighbor_count=5) -> float:
    try:
        pos = int(mutation[2:-1])
        chain_id = mutation[1]
    except Exception:
        raise ValueError(f"Cannot parse mutation string '{mutation}'. Expected format like 'EB79A'.")
    mut_key = (chain_id, pos, ' ')
    if mut_key not in cb_coords:
        raise KeyError(f"Residue {mutation} not found or incomplete in PDB.")
    mut_cb = cb_coords[mut_key]
    dists = np.linalg.norm(coords - mut_cb, axis=1)
    self_idx = keys.index(mut_key)
    sorted_idx = np.argsort(dists)
    #sorted_idx = [i for i in sorted_idx if i != self_idx]
    neighbor_idx = sorted_idx[:neighbor_count]
    score = sum(_compute_centrality(i, coords) for i in neighbor_idx)
    return float(score)

def burial_score(pdb_path: str, mutation: str, neighbor_count=5) -> float:
    """
    Compute burial score for a single mutation.
    """
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)
    return _compute_score_for_mutation(cb_coords, keys, coords, mutation, neighbor_count)

def burial_scores(pdb_path: str, mutations: list, neighbor_count=5) -> list:
    """
    Compute burial scores for multiple mutations at once.

    Parameters:
    -----------
    pdb_path : str
        Path to the PDB file.
    mutations : list of str
        List of mutation strings, e.g. ["EB79A", "KB80A"].

    Returns:
    --------
    list of float
        Burial scores in the same order as the input mutations.
    """
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)
    scores = []
    for mutation in mutations:
        scores.append(_compute_score_for_mutation(cb_coords, keys, coords, mutation, neighbor_count))
    return scores

#print(burial_scores("data/SKEMPI2/M1340_cache/wildtype/0_1A22.pdb", ["KA161A","FA165A", "KB80A", "EB79A"], neighbor_count=5))