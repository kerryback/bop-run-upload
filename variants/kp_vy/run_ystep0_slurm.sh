#!/bin/bash
#SBATCH -J ystep0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 0-03:30
#SBATCH -p htc
#SBATCH -q public
#SBATCH -A grp_sjpruitt
#SBATCH -o /data/sjpruitt/ystep0/ystep0.%j.log
#
# The KP14 y risk-adjustment gate, run 2026-09-17 (Sol 63535705). It fired: the generator was
# corrected in merge 91095fb and the corrected campaign withdrew the vyx headline. Kept as the
# reproducible harness for the check, not as pending work -- see docs/NEXTUP.md for the queue.
# Runs check_y_common_slope.py on the PROTOCOL panel (N 500, T 500, burn-in 400) for the
# two affected economies and the control, which is why it is here and not on a laptop:
# create_arrays peaks near 31 GiB at T+1 = 901.
#
# kpbase is the zero point. It has type_bv = [0.0] and gamma_v = 0, so no firm loads on y
# and its residual is the model's ordinary discretisation error with the y channel absent.
# On the short smoke panel it came out at +0.000018/month; if it is not similarly small
# here, nothing in the vyx and vyg25 rows can be read as a y result.
#
# Reads only committed tables and writes only to OUT. It does not touch the repository's
# results/ directory, so it cannot collide with variants/cluster_pull.sh.

set -u
OUT=/data/sjpruitt/ystep0
REPO=$HOME/GitHub/bop-run-upload
mkdir -p "$OUT"

module load mamba/latest
source activate bop
# `source activate` sets CONDA_PREFIX but does not prepend the env's bin/ in a
# non-interactive shell -- see run_bop_job.sh for the failure this avoids.
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "python: $(which python)"
python -c "import numpy, pandas, scipy" || { echo "ERROR: env 'bop' not usable here"; exit 1; }

cd "$REPO/variants" || exit 1
GRID=0,0.5,1,1.5,1.8,2.2,2.5,3,3.5,4,5,6,8,12,20

run () {   # run <tag> <prefix> <overrides>
  echo "=============== $1 ==============="
  KP_PARAM_OVERRIDES="$3" KP_VY_PREFIX="$2" \
    python kp_vy/check_y_common_slope.py --tag "$1" --N 500 --T 500 --seed 0 \
      --gamma_grid "$GRID" --out "$OUT/ystep0_$1_s000.json" || echo "FAILED: $1"
}

run kpbase kpbase '{"type_share":[1.0],"type_bv":[0.0],"gamma_v":0.0,"bv_comp":0.0}'
run vyx    vyx    '{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":1.8,"bv_comp":1.2}'
run vyg25  vyg25  '{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":2.5,"bv_comp":1.2}'

echo "ALL DONE $(date)"
