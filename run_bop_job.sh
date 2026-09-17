#!/bin/bash
#SBATCH -J bop_kp14
#SBATCH --array=0-10         # Set to number_of_panels - 1 (e.g. 0-99 for 100 panels)
#SBATCH --cpus-per-task=16    # coordinate with N_JOBS max (16 workers)
#SBATCH --mem=90GB
#SBATCH -t 0-16:00             #
#SBATCH -p public
#SBATCH -o outslurm/phx.bop.kp14.%a.log   # <cluster>. prefix per docs/RUNS.md; must match BOP_LOG_FILE below

# ---------------------------------------------------------------------------
# Configuration — edit these before submitting
# -------------------------------------------------------------------------
MODEL=kp14                               # bgn | kp14 | gs21
CONDA_ENV=bop                           # your conda environment name
RUN_TAG=20260917                        # unique per run (date or label); must not be reused
                                        # (same dirs = silent [SKIP] or overwrite, docs/RUNS.md)
CLUSTER=phx                             # sol | phx; must match the -o line below
SCRATCH=/data/sjpruitt/bop_${MODEL}_${RUN_TAG}      # pkl output; /data is shared and durable
                                                    # (/scratch is purged -- docs/RUNS.md)
TEMP=/scratch/sjpruitt/bop_temp_${MODEL}_${RUN_TAG} # intermediate _arr/ dirs; disposable, so
                                                    # per-cluster /scratch is fine here
                                        # (kept separate so pkl output stays clean;
                                        #  stale _arr/ from crashes never pollutes SCRATCH)

# Optional: restrict to a subset of factors via --chars.
# Specify factor names (hml, cma, rmw, umd; mkt_lev for gs21 only).
# "size" is always included as a characteristic.
# SMB and market are always present as factors in the FF and FM models.
# Leave empty to use the full default set for the chosen model.
# CHARS_FLAG="--chars umd"
# CHARS_FLAG="--chars cma,umd"     # → FF/FM factors: smb + cma + umd + market

# ---------------------------------------------------------------------------

[ -n "$RUN_TAG" ] || { echo "ERROR: set RUN_TAG to a unique value for this run"; exit 1; }

module load mamba/latest
source activate $CONDA_ENV
# `source activate` sets CONDA_PREFIX but does NOT prepend the env's bin/ to PATH
# in a non-interactive shell (which is what sbatch gives you). Without this,
# `python` resolves to the mamba BASE interpreter and every task dies instantly
# with `ModuleNotFoundError: No module named 'numpy'` -- the traceback names
# /etc/python/sitecustomize.py, which is the tell. Hit on Phoenix 2026-08-31,
# all 11 array tasks, ~1 s each.
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "python: $(which python)"          # must be under .../envs/$CONDA_ENV/bin
# ~/.conda is per-cluster: catch a missing env in ~2 s per task instead of
# 11 tasks dying on the numpy import (the Phoenix 2026-08-31 failure mode)
python -c "import numpy" 2>&1 || { echo "ERROR: env '$CONDA_ENV' not usable on this cluster (python: $(command -v python))"; exit 1; }

export BOP_SCRATCH_DIR=$SCRATCH
export BOP_TEMP_DIR=$TEMP

# Tell main.py that sbatch's `-o` is already capturing stdout+stderr, so it
# should NOT also tee to logs/. Keep this path in sync with the #SBATCH -o line
# at the top -- SBATCH directives cannot use shell variables, so the model name
# appears in both places and must match.
export BOP_LOG_FILE="outslurm/${CLUSTER}.bop.${MODEL}.${SLURM_ARRAY_TASK_ID}.log"
[ -f "$BOP_LOG_FILE" ] || { echo "ERROR: $BOP_LOG_FILE missing -- #SBATCH -o line has drifted from BOP_LOG_FILE"; exit 1; }

mkdir -p $SCRATCH
mkdir -p $TEMP
mkdir -p outslurm

echo "Running $MODEL panel $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

python main.py $MODEL $SLURM_ARRAY_TASK_ID $((SLURM_ARRAY_TASK_ID + 1)) $CHARS_FLAG
