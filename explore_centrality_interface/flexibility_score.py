import re
import numpy as np
from Bio.PDB import PDBParser

# Global caches to avoid recomputing expensive operations
_flexibility_predictions_cache = None
_pdb_structure_cache = {}
_cb_coords_cache = {}
_flex_scores_cache = {}
_chain_mapping_cache = {}

def _load_flexibility_predictions():
    """Load flexibility predictions file once and cache it."""
    global _flexibility_predictions_cache
    if _flexibility_predictions_cache is None:
        with open("data/SKEMPI2/SKEMPI2_cache/skempi_output_wildtype-3D-all-predictions.txt", "r") as f:
            _flexibility_predictions_cache = f.read()
    return _flexibility_predictions_cache

def calc_virtual_cb(n_coord: np.ndarray, ca_coord: np.ndarray, c_coord: np.ndarray) -> np.ndarray:
    """
    Calculate pseudo C-beta coordinates for glycine residues.
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
    """Get C-beta coordinates for all residues in the structure."""
    global _cb_coords_cache
    
    if pdb_path in _cb_coords_cache:
        return _cb_coords_cache[pdb_path]
    
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
    
    # Cache the result
    _cb_coords_cache[pdb_path] = (cb_coords, keys, coords)
    return cb_coords, keys, coords

def _parse_pdb_id(pdb_path: str) -> str:
    """Extract PDB ID from path like 'data/SKEMPI2/SKEMPI2_cache/wildtype/64_1IAR.pdb'"""
    match = re.search(r'(\d+)_([A-Z0-9]+)\.pdb$', pdb_path)
    if not match:
        raise ValueError(f"Could not extract PDB ID from path: {pdb_path}")
    return f"{match.group(1)}-{match.group(2)}"

def _get_chain_mapping(pdb_path: str) -> dict[str, str]:
    """
    Create a mapping between PDB chain IDs and flexibility score chain IDs.
    Uses sequence alignment and residue count matching to find the best mapping.
    """
    global _chain_mapping_cache
    
    if pdb_path in _chain_mapping_cache:
        return _chain_mapping_cache[pdb_path]
    
    # Read flexibility scores to get all available chains
    content = _load_flexibility_predictions()
    
    pdb_id = _parse_pdb_id(pdb_path)
    pattern = f">{pdb_id}_([A-Z])\n"
    flex_chains = set(re.findall(pattern, content))
    
    if not flex_chains:
        raise KeyError(f"No flexibility scores found for PDB ID {pdb_id}")
    
    # Get residue sequences for each chain in PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_path)
    model = next(structure.get_models())
    
    # Get residue sequences and counts for PDB chains
    pdb_chains = {}
    for chain in model:
        residues = list(chain.get_residues())
        if not residues:
            continue
        # Get sequence of residue names
        seq = [res.resname for res in residues if res.id[0] == ' ']  # Only standard residues
        pdb_chains[chain.id] = {
            'count': len(seq),
            'seq': seq
        }
    
    # Get residue counts for flexibility score chains
    flex_chain_data = {}
    for chain in flex_chains:
        scores = _read_flexibility_scores(pdb_id, chain)
        flex_chain_data[chain] = {
            'count': len(scores),
            'scores': scores
        }
    
    # Create mapping based on residue count matching and score patterns
    mapping = {}
    used_flex_chains = set()
    
    # Sort chains by residue count to try to match largest chains first
    sorted_pdb_chains = sorted(pdb_chains.items(), key=lambda x: x[1]['count'], reverse=True)
    sorted_flex_chains = sorted(flex_chains, key=lambda x: flex_chain_data[x]['count'], reverse=True)
    
    for pdb_chain, pdb_data in sorted_pdb_chains:
        best_match = None
        best_score = float('-inf')
        
        for flex_chain in sorted_flex_chains:
            if flex_chain in used_flex_chains:
                continue
                
            flex_data = flex_chain_data[flex_chain]
            
            # Skip if counts are too different
            if abs(pdb_data['count'] - flex_data['count']) > 10:  # Allow some flexibility
                continue
            
            # Calculate a similarity score based on count difference and score pattern
            count_diff = abs(pdb_data['count'] - flex_data['count'])
            count_score = 1.0 / (1.0 + count_diff)
            
            # Use the shorter length for comparison
            min_len = min(len(pdb_data['seq']), len(flex_data['scores']))
            
            # Calculate correlation between flexibility scores
            if min_len > 10:  # Only if we have enough residues
                pdb_scores = np.array([1.0 if res in ['GLY', 'PRO'] else 0.0 for res in pdb_data['seq'][:min_len]])
                flex_scores = np.array(flex_data['scores'][:min_len])
                pattern_score = np.corrcoef(pdb_scores, flex_scores)[0,1]
                if np.isnan(pattern_score):
                    pattern_score = 0.0
            else:
                pattern_score = 0.0
            
            # Combined score
            total_score = count_score * 0.7 + pattern_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_match = flex_chain
        
        if best_match:
            mapping[pdb_chain] = best_match
            used_flex_chains.add(best_match)
    
    if not mapping:
        raise ValueError(f"Could not find any valid chain mappings for {pdb_id}")
    
    # Cache the result
    _chain_mapping_cache[pdb_path] = mapping
    return mapping

def _read_flexibility_scores(pdb_id: str, chain: str) -> list[float]:
    """Read flexibility scores for a specific PDB ID and chain from the predictions file."""
    content = _load_flexibility_predictions()
    
    # Find the section for this PDB ID and chain
    pattern = f">{pdb_id}_{chain}\n(.*?)(?=>|$)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise KeyError(f"No flexibility scores found for {pdb_id}_{chain}")
    
    # Parse the comma-separated values
    scores_str = match.group(1).strip()
    scores = [float(x.strip()) for x in scores_str.split(",")]
    return scores

def _get_all_flexibility_scores(pdb_path: str, cb_coords: dict, keys: list) -> dict:
    """Get flexibility scores for all residues in the structure."""
    global _flex_scores_cache
    
    cache_key = f"{pdb_path}_{len(keys)}"  # Include number of keys to ensure uniqueness
    if cache_key in _flex_scores_cache:
        return _flex_scores_cache[cache_key]
    
    pdb_id = _parse_pdb_id(pdb_path)
    chain_mapping = _get_chain_mapping(pdb_path)
    
    # Get flexibility scores for each chain
    chain_scores = {}
    for chain in set(k[0] for k in keys):
        if chain not in chain_mapping:
            raise KeyError(f"No flexibility scores mapping found for chain {chain}")
        flex_chain = chain_mapping[chain]
        chain_scores[chain] = _read_flexibility_scores(pdb_id, flex_chain)
    
    # Create a mapping from (chain, res, ins) to flexibility score
    flex_scores = {}
    for key in keys:
        chain, res, ins = key
        if res <= len(chain_scores[chain]):
            flex_scores[key] = chain_scores[chain][res - 1]
        else:
            flex_scores[key] = 0.0  # Default for residues without scores
    
    # Cache the result
    _flex_scores_cache[cache_key] = flex_scores
    return flex_scores

def _compute_weighted_flexibility_score(cb_coords: dict, keys: list, coords: np.ndarray, 
                                      flex_scores: dict, mutation: str, sigma: float = 2.5) -> float:
    """
    Compute weighted flexibility score based on distance to neighbors.
    """
    if len(mutation) < 4:
        raise ValueError(f"Mutation string '{mutation}' too short")
    
    chain_id = mutation[1]
    try:
        pos = int(mutation[2:-1])
    except ValueError:
        raise ValueError(f"Cannot parse residue number from '{mutation}'")
    
    # Find the mutation residue
    candidates = [k for k in cb_coords if k[0] == chain_id and k[1] == pos]
    if not candidates:
        raise KeyError(f"Residue {chain_id}{pos} not found in PDB")
    mut_key = candidates[0]
    mut_cb = cb_coords[mut_key]
    
    # Find 9 closest neighbors
    dists = np.linalg.norm(coords - mut_cb, axis=1)
    idx9 = np.argsort(dists)[:9]
    
    # Calculate Gaussian weights
    weights = np.exp(-(dists[idx9]**2) / (2 * sigma**2))
    
    # Get flexibility scores for neighbors
    neighbor_scores = [flex_scores[keys[i]] for i in idx9]
    
    # Compute weighted average
    if weights.sum() == 0:
        return 0.0
    return float(np.dot(weights, neighbor_scores) / weights.sum())

def clear_caches():
    """Clear all caches to free memory."""
    global _flexibility_predictions_cache, _pdb_structure_cache, _cb_coords_cache, _flex_scores_cache, _chain_mapping_cache
    _flexibility_predictions_cache = None
    _pdb_structure_cache.clear()
    _cb_coords_cache.clear()
    _flex_scores_cache.clear()
    _chain_mapping_cache.clear()

def flexibility_score(pdb_path: str, mutation: str, sigma: float = 2.5) -> float:
    """
    Compute weighted flexibility score for a single mutation.
    The score is based on the flexibility scores of the 9 closest neighbors,
    weighted by their distance to the mutation site using a Gaussian function.
    
    Args:
        pdb_path: Path to the PDB file
        mutation: Mutation string in format like 'HA1A'
        sigma: Standard deviation for Gaussian weighting (default: 2.5)
    
    Returns:
        Weighted flexibility score
    """
    # Get C-beta coordinates (cached)
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)
    
    # Get flexibility scores for all residues (cached)
    flex_scores = _get_all_flexibility_scores(pdb_path, cb_coords, keys)
    
    # Compute weighted score
    return _compute_weighted_flexibility_score(cb_coords, keys, coords, flex_scores, mutation, sigma)

def flexibility_scores(pdb_path: str, mutations: list[str], sigma: float = 2.5) -> list[float]:
    """
    Compute weighted flexibility scores for multiple mutations.
    
    Args:
        pdb_path: Path to the PDB file
        mutations: List of mutation strings
        sigma: Standard deviation for Gaussian weighting (default: 2.5)
    
    Returns:
        List of weighted flexibility scores
    """
    # Get C-beta coordinates (cached)
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)
    
    # Get flexibility scores for all residues (cached)
    flex_scores = _get_all_flexibility_scores(pdb_path, cb_coords, keys)
    
    # Compute weighted scores for each mutation
    return [_compute_weighted_flexibility_score(cb_coords, keys, coords, flex_scores, mut, sigma) 
            for mut in mutations]
