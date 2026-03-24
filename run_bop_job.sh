#!/bin/bash
#SBATCH -J bop
#SBATCH --array=0-99          # Set to number_of_panels - 1 (e.g. 0-99 for 100 panels)
#SBATCH --cpus-per-task=16    # Matches MODEL_N_JOBS max (16 workers)
#SBATCH --mem=48GB
#SBATCH -t 0-4:00             # 4-hour wall (htc partition max)
#SBATCH -p htc
#SBATCH -o outslurm/bop.%A.%a.out
#SBATCH -e outslurm/bop.%A.%a.err

# ---------------------------------------------------------------------------
# Configuration — edit these before submitting
# ---------------------------------------------------------------------------

MODEL=bgn                          # bgn | kp14 | gs21
CONDA_ENV=bop             # your conda environment name
SCRATCH=/scratch/sjpruitt/bop      # all output files land here

# Optional: restrict to a subset of factors via --chars.
# Specify factor names (hml, cma, rmw, umd; mkt_lev for gs21 only).
# "size" is always included as a characteristic.
# SMB and market are always present as factors in the FF and FM models.
# Leave empty to use the full default set for the chosen model.
CHARS_FLAG=
# CHARS_FLAG="--chars cma,umd"     # → FF/FM factors: smb + cma + umd + market

# ---------------------------------------------------------------------------

module load mamba/latest
source activate $CONDA_ENV

export BOP_SCRATCH_DIR=$SCRATCH

mkdir -p $SCRATCH
mkdir -p outslurm

echo "Running $MODEL panel $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

python main.py $MODEL $SLURM_ARRAY_TASK_ID $((SLURM_ARRAY_TASK_ID + 1)) $CHARS_FLAG
