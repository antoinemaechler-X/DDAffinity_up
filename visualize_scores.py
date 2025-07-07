from pymol import cmd, stored
from explore_centrality_interface.interface_score import interface_scores
from explore_centrality_interface.burial_score import burial_scores
from explore_centrality_interface.flexibility_scores import flexibility_scores

# --- User input ---
PDB_PATH = "data/SKEMPI2/SKEMPI2_cache/wildtype/1072_1FC2.pdb"
CHAIN_A = "C"
CHAIN_B = "D"

# --- Load structure ---
cmd.load(PDB_PATH, "complex")
cmd.hide("everything", "complex")
cmd.show("sticks", "complex")

# --- Helper: get all residues in a chain ---
def get_residue_ids(chain):
    stored.residues = []
    cmd.iterate(f"chain " + chain, "stored.residues.append((resn, resi))")
    unique = sorted(set(stored.residues), key=lambda x: int(x[1]))
    return [(resn, resi) for resn, resi in unique if resi.isdigit()]

# --- Build mutation list for all residues ---
mutations = []
residue_map = {}

for chain in [CHAIN_A, CHAIN_B]:
    for resn, resi in get_residue_ids(chain):
        mutation_code = f"{resn[0]}{chain}{resi}A"
        mutations.append(mutation_code)
        residue_map[mutation_code] = (chain, resi)

print("Generated mutations:", mutations[:5])  # Print first 5 mutations to see format
print("First few mutation codes:", [m[3] for m in mutations[:5]])  # Print chain IDs

# --- Compute scores ---
interface = interface_scores(PDB_PATH, mutations)
burial = burial_scores(PDB_PATH, mutations)
flexibility = flexibility_scores(PDB_PATH, mutations)

# --- Score normalization ---
def normalize_interface(score):
    return min(max(score, 0.0), 1.0)  # already 0–1

def normalize_burial(score):
    return min(max((score - 0.5) / 0.5, 0.0), 1.0)  # stretch 0.5–1 → 0–1

def normalize_flexibility(score):
    return min(max(score, 0.0), 1.0)  # already 0–1

def color_from_score(norm_score, chain):
    if chain == CHAIN_A:
        return f"[{1 - norm_score}, {1 - norm_score}, 1]"  # blue scale
    elif chain == CHAIN_B:
        return f"[1, {1 - norm_score}, {1 - norm_score}]"  # red scale
    else:
        return f"[{1 - norm_score}, {1 - norm_score}, {1 - norm_score}]"  # gray scale for other chains

# --- Shared view setup ---
cmd.zoom("all")
cmd.turn("y", 85)
cmd.turn("x", 35)

# --- Render interface image ---
for mutation_code, score in zip(mutations, interface):
    chain, resi = residue_map[mutation_code]
    norm = normalize_interface(score)
    color = color_from_score(norm, chain)
    color_name = f"interface_{mutation_code}"
    cmd.set_color(color_name, eval(color))
    cmd.color(color_name, f"chain {chain} and resi {resi}")

cmd.ray(2000, 1500)
cmd.png("interface_scores.png", dpi=300)

# --- Render burial image ---
for mutation_code, score in zip(mutations, burial):
    chain, resi = residue_map[mutation_code]
    norm = normalize_burial(score)
    color = color_from_score(norm, chain)
    color_name = f"burial_{mutation_code}"
    cmd.set_color(color_name, eval(color))
    cmd.color(color_name, f"chain {chain} and resi {resi}")

cmd.ray(2000, 1500)
cmd.png("burial_scores.png", dpi=300)

# --- Render flexibility image ---
for mutation_code, score in zip(mutations, flexibility):
    chain, resi = residue_map[mutation_code]
    norm = normalize_flexibility(score)
    color = color_from_score(norm, chain)
    color_name = f"flexibility_{mutation_code}"
    cmd.set_color(color_name, eval(color))
    cmd.color(color_name, f"chain {chain} and resi {resi}")

cmd.ray(2000, 1500)
cmd.png("flexibility_scores.png", dpi=300)
