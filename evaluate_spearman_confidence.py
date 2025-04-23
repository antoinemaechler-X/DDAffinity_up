#!/usr/bin/env python
import os
import copy
import argparse
import warnings
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Suppress Biopython deprecation warnings
from Bio import BiopythonDeprecationWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BiopythonDeprecationWarning)
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB import Selection
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.Polypeptide import index_to_one, one_to_index

# Import custom modules (ensure these are in your PYTHONPATH)
from rde.utils.misc import load_config, seed_all
from rde.utils.train_mpnn import *  # This should include recursive_to and CrossValidation
from rde.utils.data_skempi_mpnn import PaddingCollate
from rde.models.protein_mpnn_network_2 import ProteinMPNN_NET
from rde.utils.skempi_mpnn import SkempiDatasetManager, eval_skempi_three_modes

def evaluate(config, config_model, cv_mgr, data_params, device, num_cvfolds, num_workers):
    """
    Run one full evaluation for a given directory pair.
    This function updates the data directories in the configuration, creates a new dataset manager,
    and runs through each cross-validation fold to collect prediction results.
    """
    # Override the data directories in config and update the model's data configuration
    config['data']['pdb_wt_dir'] = data_params['pdb_wt_dir']
    config['data']['pdb_mt_dir'] = data_params['pdb_mt_dir']
    config_model['data'] = config['data']
    
    # Create a new dataset manager using the updated config
    dataset_mgr = SkempiDatasetManager(config_model, num_cvfolds=num_cvfolds, num_workers=num_workers)

    results = []
    # Loop through each cross-validation fold to get predictions
    for fold in range(num_cvfolds):
        model, _, _ = cv_mgr.get(fold)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataset_mgr.get_val_loader(fold), desc=f'Fold {fold} Validation', leave=False):
                # Move the batch to the device
                batch = recursive_to(batch, device)
                # Forward pass
                _, output_dict = model(batch)
                # Collect results for each sample in the batch
                for comp, mutstr, ddg_true, ddg_pred in zip(
                    batch["wt"]['complex'], 
                    batch["wt"]['mutstr'], 
                    output_dict['ddG_true'], 
                    output_dict['ddG_pred']
                ):
                    results.append({
                        'complex': comp,
                        'mutstr': mutstr,
                        'num_muts': len(mutstr.split(',')),
                        'ddG': ddg_true.item(),
                        'ddG_pred': ddg_pred.item()
                    })

    results_df = pd.DataFrame(results)
    # Add the 'datasets' column, as expected by eval_skempi_three_modes
    results_df['datasets'] = 'SKEMPI2'
    # Compute evaluation metrics; assumed to return a DataFrame with columns overall_pearson and overall_spearman
    df_metrics = eval_skempi_three_modes(results_df)
    overall_pearson = df_metrics['overall_pearson'].iloc[0]
    overall_spearman = df_metrics['overall_spearman'].iloc[0]
    return overall_pearson, overall_spearman

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MP-Protein model on SKEMPI2 with multiple directory configurations"
    )
    parser.add_argument('config', type=str, help="Path to configuration YAML file")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (e.g., cuda:0 or cpu)")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of data loader workers")
    parser.add_argument('--num_iterations', type=int, default=10, help="Number of iterations per directory pair")
    args = parser.parse_args()
    
    # Load configuration and model checkpoint
    config, _ = load_config(args.config)
    ckpt = torch.load(config.checkpoint, map_location=args.device, weights_only=False)
    config_model = ckpt['config']
    num_cvfolds = len(ckpt['model']['models'])
    
    # Setup cross-validation manager and load model state
    cv_mgr = CrossValidation(
        model_factory=ProteinMPNN_NET,
        config=config_model,
        early_stoppingdir=config.early_stoppingdir,
        num_cvfolds=num_cvfolds
    ).to(args.device)
    cv_mgr.load_state_dict(ckpt['model'])
    
    # Define the four directory pairs for evaluation
    dir_pairs = {
        "WT/WT": {
            "pdb_wt_dir": "./data/SKEMPI2/M1340_cache/wildtype",
            "pdb_mt_dir": "./data/SKEMPI2/M1340_cache/wildtype"
        },
        "WT/Opt": {
            "pdb_wt_dir": "./data/SKEMPI2/M1340_cache/wildtype",
            "pdb_mt_dir": "./data/SKEMPI2/M1340_cache/optimized"
        },
        "WT_Evo/WT_Evo": {
            "pdb_wt_dir": "./data/SKEMPI2/M1340_cache/wildtype_evoef1",
            "pdb_mt_dir": "./data/SKEMPI2/M1340_cache/wildtype_evoef1"
        },
        "WT_Evo/Opt_Evo": {
            "pdb_wt_dir": "./data/SKEMPI2/M1340_cache/wildtype_evoef1",
            "pdb_mt_dir": "./data/SKEMPI2/M1340_cache/optimized_evoef1"
        }
    }
    
    # Containers to store metrics for each configuration
    results_overall_pearson = { name: [] for name in dir_pairs }
    results_overall_spearman = { name: [] for name in dir_pairs }
    
    # Iterate through each directory pair configuration
    for pair_name, dirs in dir_pairs.items():
        print(f"\nEvaluating configuration: {pair_name}")
        # Run the evaluation for the defined number of iterations
        for it in range(args.num_iterations):
            print(f"  Iteration {it+1}/{args.num_iterations}")
            overall_pearson, overall_spearman = evaluate(
                config, config_model, cv_mgr, dirs, args.device, num_cvfolds, args.num_workers
            )
            results_overall_pearson[pair_name].append(overall_pearson)
            results_overall_spearman[pair_name].append(overall_spearman)
    
    # Prepare data for plotting box plots
    labels = list(dir_pairs.keys())
    pearson_data = [results_overall_pearson[label] for label in labels]
    spearman_data = [results_overall_spearman[label] for label in labels]
    
    # Generate and save box plot for Overall Pearson Scores
    plt.figure(figsize=(10, 6))
    plt.boxplot(pearson_data, labels=labels, patch_artist=True)
    plt.title("Overall Pearson Scores")
    plt.xlabel("Directory Pair Configuration")
    plt.ylabel("Pearson Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("boxplot_overall_pearson.png")
    plt.close()
    
    # Generate and save box plot for Overall Spearman Scores
    plt.figure(figsize=(10, 6))
    plt.boxplot(spearman_data, labels=labels, patch_artist=True)
    plt.title("Overall Spearman Scores")
    plt.xlabel("Directory Pair Configuration")
    plt.ylabel("Spearman Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("boxplot_overall_spearman.png")
    plt.close()
    
    print("\nEvaluation completed. Box plots saved as:")
    print("  boxplot_overall_pearson.png")
    print("  boxplot_overall_spearman.png")

if __name__ == '__main__':
    main()
