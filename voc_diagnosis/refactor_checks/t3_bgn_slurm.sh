#!/bin/bash
#SBATCH -J bop_t3_bgn
#SBATCH --cpus-per-task=16
#SBATCH --mem=90GB
#SBATCH -t 0-06:00
#SBATCH -p public
#SBATCH -o outslurm/t3_bgn.%j.log
# RECOVERED 2026-09-08 from an UNTRACKED file at the repo root on Sol (dated 2026-08-25).
# It is the SLURM runner for the tracked acceptance test below, and the only record of
# how the _term4_gram refactor (c96f8d8, "halve peak memory in _term4_gram with an
# in-place exp") was validated at production scale. Every other cluster runner in this
# repo is committed beside what it drives; this one was left behind and would have died
# with /tmp. Moved here, unchanged below this header. Submit from the REPO ROOT, as the
# relative paths assume.

# T3: production-scale PAIRED acceptance test for the sdf_compute_bgn refactor.
# Generates one BGN panel at N=1000, then runs the moments step through both the
# pre-refactor and post-refactor code on the SAME arrays. See
# voc_diagnosis/refactor_checks/t3_production.py for why it is paired.

set -euo pipefail
module load mamba/latest
source activate bop

SCRATCH=/scratch/sjpruitt/bop_t3_bgn
TEMP=/scratch/sjpruitt/bop_temp_t3_bgn
export BOP_SCRATCH_DIR=$SCRATCH
export BOP_TEMP_DIR=$TEMP
mkdir -p "$SCRATCH" "$TEMP" outslurm

echo "=== T3 on $(hostname) at $(date) ==="
git -C . log --oneline -1
echo "--- confirming the refactor is present ---"
grep -c "_term4_gram" utils_bgn/sdf_compute_bgn.py

echo; echo "=== STEP 1: generate one production BGN panel (N=1000) ==="
/usr/bin/time -v python utils/generate_panel.py bgn 0 2>&1 | \
  grep -vE "^\s+(Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Exit status|Average|Minor|Major)" || true

echo; echo "=== STEP 2: paired old-vs-new moments comparison ==="
python voc_diagnosis/refactor_checks/t3_production.py bgn_0 360
RC=$?

echo; echo "=== done at $(date), exit $RC ==="
exit $RC
