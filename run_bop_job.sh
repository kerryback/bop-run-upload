#!/bin/bash
#SBATCH -J bop
#SBATCH --array=1,4,6,8          # Set to number_of_panels - 1 (e.g. 0-99 for 100 panels)
#SBATCH --cpus-per-task=16    # coordinate with N_JOBS max (16 workers)
#SBATCH --mem=48GB
#SBATCH -t 0-10:00             # 
#SBATCH -p public
#SBATCH -o outslurm/bop.bgn.%a.out 
#SBATCH -e outslurm/bop.bgn.%a.err

# ---------------------------------------------------------------------------
# Configuration — edit these before submitting
# -------------------------------------------------------------------------
MODEL=bgn                               # bgn | kp14 | gs21
CONDA_ENV=bop                           # your conda environment name
SCRATCH=/scratch/sjpruitt/bop           # permanent output files (pkl) land here
TEMP=/scratch/sjpruitt/bop_temp         # intermediate _arr/ directories land here
                                        # (kept separate so pkl output stays clean;
                                        #  stale _arr/ from crashes never pollutes SCRATCH)

# Optional: restrict to a subset of factors via --chars.
# Specify factor names (hml, cma, rmw, umd; mkt_lev for gs21 only).
# "size" is always included as a characteristic.
# SMB and market are always present as factors in the FF and FM models.
# Leave empty to use the full default set for the chosen model.
CHARS_FLAG="--chars umd"
# CHARS_FLAG="--chars cma,umd"     # → FF/FM factors: smb + cma + umd + market

# ---------------------------------------------------------------------------

module load mamba/latest
source activate $CONDA_ENV

export BOP_SCRATCH_DIR=$SCRATCH
export BOP_TEMP_DIR=$TEMP

mkdir -p $SCRATCH
mkdir -p $TEMP
mkdir -p outslurm

echo "Running $MODEL panel $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

python main.py $MODEL $SLURM_ARRAY_TASK_ID $((SLURM_ARRAY_TASK_ID + 1)) $CHARS_FLAG
