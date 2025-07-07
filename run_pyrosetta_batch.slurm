#!/bin/bash
#SBATCH --job-name=ptm_pyrosetta
#SBATCH --output=ptm_pyrosetta_%j.out
#SBATCH --error=ptm_pyrosetta_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=shared,sapphire,shakhnovich
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antoinemaechler@fas.harvard.edu

source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate RDE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /n/netscratch/shakhnovich_lab/Lab/amaechler/DDAffinity_up

# for optimized (mutation + minimize)
# python opti_pyrosetta.py \
#   --csv    data/SKEMPI2/SKEMPI2.csv \
#   --pdbs   data/SKEMPI2/PDBs \
#   --outdir data/SKEMPI2/SKEMPI2_cache/wildtype_pyrosetta \
#   --mode   wildtype

#for wildtype (only minimize)
python opti_pyrosetta.py \
  --csv    data/SKEMPI2/SKEMPI2.csv \
  --pdbs   data/SKEMPI2/PDBs \
  --outdir data/SKEMPI2/SKEMPI2_cache/wildtype_pyrosetta \
  --mode   wildtype
