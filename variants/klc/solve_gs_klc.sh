#!/bin/bash
#SBATCH -J bop_solve_gs
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 0-20:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/solve_gs.%A.%a.log
#
# The five solves of var-gs_bx-gx7w-v1, one exposure type per task, parameters READ from the spec's
# stage so there is no second copy. Each task checks the id the solver prints against the spec's.
# Measured envelope elsewhere (run_gx7_slurm.sh): ~6 h, ~2.7 GiB, tolerance exit near sweep 2200-3000
# of a 5600 cap. Submit from the repo root:  sbatch variants/klc/solve_gs_klc.sh
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -f "$REPO/variants/run_seeds_slurm.sh" ] || { echo "submit from the repo root" >&2; exit 2; }
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
S=${GSBX_SPEC:-var-gs_bx-gx7w-v1}
i=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
st() { python -c "import json,sys; s=json.load(open('experiments/specs/$S.json')); g=s['solve']['stages'][$i]; print($1)"; }
OUTDIR=$(st "g['name']"); OV=$(st "json.dumps(g['params'])"); XNUM=$(st "g['args']['xnum']"); TOL=$(st "g['args']['tol']")
EXPECT=$(st "(s.get('expected_solves') or {}).get(g['name']) or ''")
LOG="$BOP_RUNS/logs/solve_gs.detail.$OUTDIR.log"
echo "=== $OUTDIR on $(hostname) $(date '+%F %T') threads=$NT overrides=$OV expecting $EXPECT ===" | tee "$LOG"
cd variants/gs_bx
T0=$(date +%s)
GS_PARAM_OVERRIDES="$OV" python -W ignore gs_solve_gam.py "$XNUM" "$TOL" "$OUTDIR" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== $OUTDIR END $(date '+%F %T') WALL=$(( $(date +%s)-T0 ))s rc=$RC ===" | tee -a "$LOG"
# The solver prints "converged" on the sweep-capped exit too (WORKING.md §43); trust the marker.
if grep -q "cycle-averaged; stopping" "$LOG"; then
  echo "FAIL: $OUTDIR hit the sweep cap (cycle-averaged) and did NOT meet tolerance" | tee -a "$LOG"; RC=4
fi
if [ -n "$EXPECT" ]; then
  grep -q "solve_id=$EXPECT" "$LOG" || { echo "FAIL: $OUTDIR did not print the spec's id $EXPECT" | tee -a "$LOG"; RC=3; }
fi
exit $RC
