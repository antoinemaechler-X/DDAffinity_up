import re
import numpy as np
from Bio.PDB import PDBParser

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
    # Read flexibility scores to get all available chains
    with open("data/SKEMPI2/SKEMPI2_cache/skempi_output_wildtype-3D-all-predictions.txt", "r") as f:
        content = f.read()
    
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
    
    # Create mapping based on:
    # 1. Residue count matching
    # 2. Score patterns (e.g., high flexibility regions should match)
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
            
            # Calculate a similarity score based on:
            # 1. Count difference (smaller is better)
            # 2. Score pattern similarity (correlation of flexibility scores)
            count_diff = abs(pdb_data['count'] - flex_data['count'])
            count_score = 1.0 / (1.0 + count_diff)
            
            # Use the shorter length for comparison
            min_len = min(len(pdb_data['seq']), len(flex_data['scores']))
            
            # Calculate correlation between flexibility scores
            # This helps match chains with similar flexibility patterns
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
    
    return mapping

def _read_flexibility_scores(pdb_id: str, chain: str) -> list[float]:
    """Read flexibility scores for a specific PDB ID and chain from the predictions file."""
    with open("data/SKEMPI2/SKEMPI2_cache/skempi_output_wildtype-3D-all-predictions.txt", "r") as f:
        content = f.read()
    
    # Find the section for this PDB ID and chain
    pattern = f">{pdb_id}_{chain}\n(.*?)(?=>|$)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise KeyError(f"No flexibility scores found for {pdb_id}_{chain}")
    
    # Parse the comma-separated values
    scores_str = match.group(1).strip()
    scores = [float(x.strip()) for x in scores_str.split(",")]
    return scores

def flexibility_score(pdb_path: str, mutation: str) -> float:
    """
    Get flexibility score for a single mutation.
    The mutation string should be in format like 'HA1A' where:
    - First letter: amino acid
    - Second letter: chain ID
    - Digits: residue number
    - Last letter: 'A' (mutant AA, not used here)
    """
    if len(mutation) < 4:
        raise ValueError(f"Mutation string '{mutation}' too short")
    
    pdb_id = _parse_pdb_id(pdb_path)
    pdb_chain = mutation[1]
    print(f"Debug - mutation: {mutation}, pdb_chain: {pdb_chain}")  # Debug print
    
    try:
        pos = int(mutation[2:-1])
    except ValueError:
        raise ValueError(f"Cannot parse residue number from '{mutation}'")
    
    # Map PDB chain to flexibility score chain
    chain_mapping = _get_chain_mapping(pdb_path)
    print(f"Debug - chain_mapping: {chain_mapping}")  # Debug print
    if pdb_chain not in chain_mapping:
        raise KeyError(f"No flexibility scores mapping found for chain {pdb_chain}")
    flex_chain = chain_mapping[pdb_chain]
    
    scores = _read_flexibility_scores(pdb_id, flex_chain)
    if pos > len(scores):
        raise IndexError(f"Position {pos} out of range for chain {pdb_chain} (length {len(scores)})")
    
    return scores[pos - 1]  # Convert to 0-based index

def flexibility_scores(pdb_path: str, mutations: list[str]) -> list[float]:
    """
    Get flexibility scores for multiple mutations.
    Returns a list of scores in the same order as the mutations list.
    """
    pdb_id = _parse_pdb_id(pdb_path)
    chain_mapping = _get_chain_mapping(pdb_path)
    
    # Group mutations by chain to minimize file reads
    chain_scores = {}
    for mutation in mutations:
        pdb_chain = mutation[1]
        if pdb_chain not in chain_mapping:
            raise KeyError(f"No flexibility scores mapping found for chain {pdb_chain}")
        flex_chain = chain_mapping[pdb_chain]
        if flex_chain not in chain_scores:
            chain_scores[flex_chain] = _read_flexibility_scores(pdb_id, flex_chain)
    
    # Get scores for each mutation
    scores = []
    for mutation in mutations:
        pdb_chain = mutation[1]
        flex_chain = chain_mapping[pdb_chain]
        try:
            pos = int(mutation[2:-1])
        except ValueError:
            raise ValueError(f"Cannot parse residue number from '{mutation}'")
        
        if pos > len(chain_scores[flex_chain]):
            raise IndexError(f"Position {pos} out of range for chain {pdb_chain} (length {len(chain_scores[flex_chain])})")
        
        scores.append(chain_scores[flex_chain][pos - 1])  # Convert to 0-based index
    
    return scores 