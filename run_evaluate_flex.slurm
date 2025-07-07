#!/bin/bash
#SBATCH --job-name=eval_flex
#SBATCH --output=eval_flex%j.out
#SBATCH --error=eval_flex%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=shared,sapphire,shakhnovich
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antoinemaechler@fas.harvard.edu
#SBATCH --nice=10000

# Load your environment
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate RDE

# Go to your project directory
cd /n/netscratch/shakhnovich_lab/Lab/amaechler/DDAffinity_up

# Run your script
python explore_centrality_interface/flexibility_analysis.py -n -1 -v