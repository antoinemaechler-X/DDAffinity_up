#!/bin/bash
#SBATCH --job-name=train_M1340    # a short name for your job
#SBATCH --output=train_%j.out      # STDOUT → evoef1_<jobid>.out
#SBATCH --error=train_%j.err       # STDERR → evoef1_<jobid>.err
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -t 12:00:00
#SBATCH --nice=10000                # lower priority
#SBATCH --mail-type=END,FAIL        # send email on job end/fail
#SBATCH --mail-user=antoinemaechler_new@fas.harvard.edu

# === load your environment ===
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate RDE

# === move into your working folder ===
cd /n/netscratch/shakhnovich_lab/Lab/amaechler/DDAffinity_up

# === run the training script ===
python train_DDAffinity.py ./configs/train/mpnn_ddg.yml --device cuda:0 --num_cvfolds 10

# === end of script ===
