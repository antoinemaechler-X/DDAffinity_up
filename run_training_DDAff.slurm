#!/bin/bash
#SBATCH --job-name=train_DDAff_simple    # a short name for your job
#SBATCH --output=train_simple_%j.out      # STDOUT → train_simple_<jobid>.out
#SBATCH --error=train_simple_%j.err       # STDERR → train_simple_<jobid>.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --nice=10000                # lower priority
#SBATCH --mail-type=END,FAIL        # send email on job end/fail
#SBATCH --mail-user=antoinemaechler_new@fas.harvard.edu
#SBATCH --export=CUDA_LAUNCH_BLOCKING=1

# === load your environment ===
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate RDE

# === move into your working folder ===
cd /n/netscratch/shakhnovich_lab/Lab/amaechler/DDAffinity_up

# === temporarily downgrade numpy to 1.x version ===
#pip install numpy==1.24.3 --no-deps

# === run the training script ===
python train_DDAffinity_grouped.py ./configs/train/mpnn_ddg.yml --num_cvfolds 10 --device cuda:0 --split data/complex_sequences_grouped_80.csv
#python scripts/train_spr.py   configs/train/mpnn_ddg_spr.yml   --num_cvfolds 10   --logdir ./logs_skempi_spr   --early_stoppingdir ./early_stopping_spr   --device cuda:0
#python train_DDAffinity.py ./configs/train/mpnn_ddg.yml --num_cvfolds 10 --logdir ./logs_skempi --early_stoppingdir ./early_stopping --device cuda:0
# === end of script ===
