#!/usr/bin/env python3
"""
compare_foldx_pyrosetta.py

Compare how FoldX and Pyrosetta evaluate mutation effects by analyzing the structural differences
between their wildtype and mutant structures. The script calculates and visualizes how each method
predicts the impact of mutations on protein structure.

Usage:
  python compare_foldx_pyrosetta.py
"""

import os
from pathlib import Path
from statistics import mean
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_pdb_coords(path):
    coords = {}
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                # Skip hydrogens and alternate conformations
                if line[12:16].strip().startswith('H') or line[16] != ' ':
                    continue
                    
                atom_name = line[12:16].strip()
                chain_id  = line[21].strip()
                res_seq   = line[22:26].strip()
                i_code    = line[26].strip()
                res_name  = line[17:20].strip()
                key = (chain_id, res_seq, i_code, res_name, atom_name)
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                coords[key] = (x, y, z)
    return coords

def is_backbone_atom(atom_name):
    return atom_name in ['N', 'CA', 'C', 'O']

def is_cbeta(atom_name):
    return atom_name == 'CB'

def calculate_mutation_effects(wt_coords, mt_coords):
    """Calculate how much each atom moves due to mutation"""
    effects = []
    
    # Find common atoms
    common_atoms = set(wt_coords.keys()) & set(mt_coords.keys())
    
    for key in common_atoms:
        chain, res, icode, resname, atom = key
        x1,y1,z1 = wt_coords[key]
        x2,y2,z2 = mt_coords[key]
        if (x1!=x2) or (y1!=y2) or (z1!=z2):
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            mag = (dx*dx + dy*dy + dz*dz)**0.5
            effects.append((key, mag))
    
    return effects

def analyze_mutation_effects(wt_coords, mt_coords, method_name):
    """Analyze mutation effects for a given method"""
    effects = calculate_mutation_effects(wt_coords, mt_coords)
    
    # Calculate statistics
    all_effects = [d[1] for d in effects]
    backbone_effects = [d[1] for d in effects if is_backbone_atom(d[0][4])]
    cbeta_effects = [d[1] for d in effects if is_cbeta(d[0][4])]
    
    stats = {
        'all': {
            'mean': mean(all_effects) if all_effects else 0,
            'max': max(all_effects) if all_effects else 0,
            'count': len(all_effects)
        },
        'backbone': {
            'mean': mean(backbone_effects) if backbone_effects else 0,
            'max': max(backbone_effects) if backbone_effects else 0,
            'count': len(backbone_effects)
        },
        'cbeta': {
            'mean': mean(cbeta_effects) if cbeta_effects else 0,
            'max': max(cbeta_effects) if cbeta_effects else 0,
            'count': len(cbeta_effects)
        },
        'effects': effects
    }
    
    # Print summary
    print(f"\n{method_name} Mutation Effects:")
    print(f"Total atoms affected: {stats['all']['count']}")
    print(f"Backbone atoms affected: {stats['backbone']['count']}")
    print(f"Cβ atoms affected: {stats['cbeta']['count']}")
    print(f"\nAverage displacement:")
    print(f"  All atoms: {stats['all']['mean']:.3f} Å")
    print(f"  Backbone: {stats['backbone']['mean']:.3f} Å")
    print(f"  Cβ: {stats['cbeta']['mean']:.3f} Å")
    print(f"\nMaximum displacement:")
    print(f"  All atoms: {stats['all']['max']:.3f} Å")
    print(f"  Backbone: {stats['backbone']['max']:.3f} Å")
    print(f"  Cβ: {stats['cbeta']['max']:.3f} Å")
    
    return stats

def plot_comparison(foldx_stats, pyro_stats):
    """Create plots comparing FoldX and Pyrosetta mutation effects"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create dictionaries for easy lookup
    foldx_effects = {k: v for k, v in foldx_stats['effects']}
    pyro_effects = {k: v for k, v in pyro_stats['effects']}
    
    # Find common affected atoms
    common_atoms = set(foldx_effects.keys()) & set(pyro_effects.keys())
    
    # Calculate averages for common atoms
    common_all = [foldx_effects[k] for k in common_atoms]
    common_backbone = [foldx_effects[k] for k in common_atoms if is_backbone_atom(k[4])]
    common_cbeta = [foldx_effects[k] for k in common_atoms if is_cbeta(k[4])]
    
    pyro_common_all = [pyro_effects[k] for k in common_atoms]
    pyro_common_backbone = [pyro_effects[k] for k in common_atoms if is_backbone_atom(k[4])]
    pyro_common_cbeta = [pyro_effects[k] for k in common_atoms if is_cbeta(k[4])]
    
    # Plot average mutation effects
    categories = ['All Atoms', 'Common Atoms', 'Backbone', 'Cβ Atoms']
    
    # Calculate values for each category
    foldx_values = [
        foldx_stats['all']['mean'],  # All atoms
        mean(common_all) if common_all else 0,  # Common atoms
        foldx_stats['backbone']['mean'],  # Backbone
        foldx_stats['cbeta']['mean']  # Cβ
    ]
    
    pyro_values = [
        pyro_stats['all']['mean'],  # All atoms
        mean(pyro_common_all) if pyro_common_all else 0,  # Common atoms
        pyro_stats['backbone']['mean'],  # Backbone
        pyro_stats['cbeta']['mean']  # Cβ
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, foldx_values, width, label='FoldX', color='skyblue')
    bars2 = ax.bar(x + width/2, pyro_values, width, label='Pyrosetta', color='lightcoral')
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    ax.set_ylabel('Average Displacement (Å)')
    ax.set_title('Average Structural Changes Due to Mutation')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add statistics box on the right
    stats_text = (
        f"Atoms affected:\n"
        f"  FoldX: {foldx_stats['all']['count']}\n"
        f"  Pyrosetta: {pyro_stats['all']['count']}\n"
        f"Common atoms affected:\n"
        f"  Total: {len(common_atoms)}\n"
        f"  Backbone: {len(common_backbone)}\n"
        f"  Cβ: {len(common_cbeta)}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=props, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('mutation_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_dir = Path("data/SKEMPI2/SKEMPI2_cache")
    structure_id = "0_1CSE.pdb"

    # Define paths
    paths = {
        "wildtype": {
            "foldx": base_dir / "wildtype" / structure_id,
            "pyrosetta": base_dir / "wildtype_pyrosetta" / structure_id
        },
        "mutant": {
            "foldx": base_dir / "optimized" / structure_id,
            "pyrosetta": base_dir / "optimized_pyrosetta" / structure_id
        }
    }

    # Load all structures
    print("Loading structures...")
    wt_foldx = parse_pdb_coords(paths["wildtype"]["foldx"])
    wt_pyro = parse_pdb_coords(paths["wildtype"]["pyrosetta"])
    mt_foldx = parse_pdb_coords(paths["mutant"]["foldx"])
    mt_pyro = parse_pdb_coords(paths["mutant"]["pyrosetta"])

    # Analyze mutation effects for each method
    print("\nAnalyzing mutation effects...")
    foldx_stats = analyze_mutation_effects(wt_foldx, mt_foldx, "FoldX")
    pyro_stats = analyze_mutation_effects(wt_pyro, mt_pyro, "Pyrosetta")
    
    # Create comparison plots
    print("\nGenerating plots...")
    plot_comparison(foldx_stats, pyro_stats)
    print("\nPlots have been saved to 'mutation_effects.png'")

if __name__ == '__main__':
    main() 