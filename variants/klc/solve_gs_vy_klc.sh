#!/bin/bash
#SBATCH -J bop_solve_gvy
#SBATCH -A kellogg
#SBATCH -p kellogg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 1-12:00
#SBATCH -o /kellogg/proj/axa3203/bop_runs/logs/solve_gvy.%A.%a.log
#
# The gs_vy solves of one spec, one exposure type per array task, parameters READ from the spec's stage.
# Each task checks the id the solver prints against the spec's (or fills it in when the spec pins none).
#   export GSPEC=var-gs_vy-gvy25-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh
# Measured: NY=9 at xnum 161 ~ (see the build log) s/sweep; tolerance exit typically 2200-3000 sweeps of a 5600 cap.
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
[ -f "$REPO/variants/run_seeds_slurm.sh" ] || { echo "submit from the repo root" >&2; exit 2; }
: "${GSPEC:?export GSPEC=<gs_vy spec id>}"
source "$REPO/variants/klc/klc_env.sh"
cd "$REPO"
i=${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array job}
st() { python -c "import json,sys; s=json.load(open('experiments/specs/$GSPEC.json')); g=s['solve']['stages'][$i]; print($1)"; }
OUTDIR=$(st "g['name']"); OV=$(st "json.dumps(g['params'])"); XNUM=$(st "g['args']['xnum']"); TOL=$(st "g['args']['tol']")
EXPECT=$(st "(s.get('expected_solves') or {}).get(g['name'], '')")
LOG="$BOP_RUNS/logs/solve_gvy.detail.$OUTDIR.log"
echo "=== $OUTDIR on $(hostname) $(date '+%F %T') threads=$NT overrides=$OV expecting '$EXPECT' ===" | tee "$LOG"
cd variants/gs_vy
T0=$(date +%s)
GS_PARAM_OVERRIDES="$OV" python -W ignore gs_solve_vy.py "$XNUM" "$TOL" "$OUTDIR" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== $OUTDIR END $(date '+%F %T') WALL=$(( $(date +%s)-T0 ))s rc=$RC ===" | tee -a "$LOG"
if grep -q "cycle-averaged; stopping" "$LOG"; then
  echo "FAIL: $OUTDIR hit the sweep cap (cycle-averaged) and did NOT meet tolerance" | tee -a "$LOG"; RC=4
fi
if [ -n "$EXPECT" ]; then
  grep -q "solve_id=$EXPECT" "$LOG" || { echo "FAIL: $OUTDIR did not print the spec's id $EXPECT" | tee -a "$LOG"; RC=3; }
fi
exit $RC
