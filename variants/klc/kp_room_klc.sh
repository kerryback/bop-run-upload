#!/bin/bash
#SBATCH -J bop_kproom
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 0-08:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/kproom.%A.%a.log
#
# KP14 oracle-only ROOM screen (docs/REDUCED_FORM.md, 2026-10-07). One array task = one point of
# variants/klc/kp_room_points.json: build that point's G and integral tables under its own prefix
# (build_vy_tables.py, ~20 min on Quest), then run the ORACLE stage only (run_oracle.py, no
# estimators, no panel) for seeds 0..NSEEDS-1, writing to BOP_RESULTS_DIR=/kellogg/proj/axa3203/
# bop_runs/kpscreen -- never variants/results. Read with variants/klc/kp_room_table.py.
#
#   sbatch --export=ALL --array=0-5%3 variants/klc/kp_room_klc.sh
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -f "$REPO/variants/run_seeds_slurm.sh" ] || { echo "submit from the repo root" >&2; exit 2; }
source "$REPO/variants/klc/klc_env.sh"
export BOP_RESULTS_DIR=${BOP_RESULTS_DIR:-$BOP_RUNS/kpscreen}
NSEEDS=${NSEEDS:-2}
POINTS="$REPO/variants/klc/kp_room_points.json"
i=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
read -r PREFIX KOV <<< "$(python - "$POINTS" "$i" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); i = int(sys.argv[2])
pts = d["points"]
if i >= len(pts): sys.exit(f"array index {i} has no point (there are {len(pts)})")
p = dict(d["base"]); p.update(pts[i]["override"])
print(pts[i]["prefix"], json.dumps(p, separators=(",", ":")))
PY
)"
echo "=== $(hostname) $(date '+%F %T') point $i prefix=$PREFIX overrides=$KOV results=$BOP_RESULTS_DIR ==="
mkdir -p "$BOP_RESULTS_DIR"
cd "$REPO"
# 1. tables. The integ stage fans out over 6 worker processes; BLAS single-threaded inside them.
( cd variants/kp_vy && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    KP_PARAM_OVERRIDES="$KOV" python -W ignore build_vy_tables.py "$PREFIX" ) 2>&1 | tee "$BOP_RUNS/logs/kproom.${SLURM_ARRAY_JOB_ID}.${PREFIX}.build.log"
# 2. oracle only, the protocol's panel (N 500, T 500, eval window 360, --levels, kp_vy's full rf set).
cd variants
for s in $(seq 0 $((NSEEDS - 1))); do
  echo "--- oracle $PREFIX seed $s  $(date '+%F %T')"
  KP_PARAM_OVERRIDES="$KOV" KP_VY_PREFIX="$PREFIX" python -W ignore run_oracle.py --model kp_vy --N 500 --T 500 \
      --seed "$s" --tag "$PREFIX" --eval_window 360 --levels
done
echo "=== END $(date '+%F %T') ==="
