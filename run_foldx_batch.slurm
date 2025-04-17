#!/bin/bash
#SBATCH --job-name=foldx_skemp
#SBATCH --output=foldx_%j.out
#SBATCH --error=foldx_%j.err
#SBATCH --time=24:00:00             # Max time (up to you)
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=shared,sapphire,shakhnovich
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@harvard.edu  # << set this!
#SBATCH --nice=10000                # Lower priority, no rush

# Load your environment
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate RDE

# Go to your project directory
cd ~/DDAffinity/rde/datasets

# Run your script
python PDB_generate_new_version.py
