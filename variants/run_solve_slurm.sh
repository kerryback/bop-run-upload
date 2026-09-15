#!/bin/bash
#SBATCH -J bop_solve
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 0-06:00
#SBATCH -p public
#SBATCH -o outslurm/solve.%j.log
#
# Build ONE spec's solve FROM THE SPEC, and fail unless every id it produces is the one the
# spec precommitted. Chain the spec's seed array on it with --dependency=afterok:<this job>,
# so the array can only start on the economy its spec names.
#
#     sbatch --export=ALL,SOLVE_SPEC=var-kp_vy-vyg25-v2   variants/run_solve_slurm.sh
#     sbatch --export=ALL,SOLVE_SPEC=var-bgn_gam-g0235d-v2 variants/run_solve_slurm.sh
#
# The parameters are READ from experiments/specs/<SOLVE_SPEC>.json, never restated here: a
# second copy of an economy's parameters is what drifted before (tests/test_specs_match_shell.py
# pins the seed array's copy for exactly that reason). Covers the two single-namespace producers,
# kp_vy (build_vy_tables.py: stages G and integ) and bgn_gam (rebuild_jstar_gam.py: stage jstar).
# GS solves have their own per-type arrays.
#
# Submit from the REPO ROOT. Both producers record their manifest in experiments/registry and
# their tables beside themselves, which is where the seed array's precondition looks.
set -euo pipefail
: "${SOLVE_SPEC:?set SOLVE_SPEC=<spec id>, e.g. sbatch --export=ALL,SOLVE_SPEC=var-kp_vy-vyg25-v2 ...}"

CONDA_ENV=${CONDA_ENV:-bop}
module load mamba/latest
source activate "$CONDA_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"      # see run_seeds_slurm.sh: source activate alone leaves base python first
# build_vy_tables.py fans the integ stage out over 6 worker processes; keep BLAS single-threaded
# inside them so they do not oversubscribe the allocation.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
SPECF="experiments/specs/$SOLVE_SPEC.json"
[ -f "$SPECF" ] || { echo "ABORT: no $SPECF (submit from the repo root)" >&2; exit 2; }
spec() { python -c "import json,sys; s=json.load(open(sys.argv[1])); print($1)" "$SPECF"; }
MODEL=$(spec "s['model']")
OV=$(spec "json.dumps(s['params'])")
mkdir -p outslurm
LOG="$REPO/outslurm/solve.${SOLVE_SPEC}.${SLURM_JOB_ID:-local}.txt"
echo "=== solve $SOLVE_SPEC ($MODEL) on $(hostname) $(date '+%F %T') commit $(git rev-parse --short HEAD) ===" | tee "$LOG"
echo "expected: $(spec "s['expected_solves']")" | tee -a "$LOG"

case "$MODEL" in
  kp_vy)
    PREFIX=$(spec "s['env']['KP_VY_PREFIX']")
    ( cd variants/kp_vy && KP_PARAM_OVERRIDES="$OV" python -W ignore build_vy_tables.py "$PREFIX" ) 2>&1 | tee -a "$LOG" ;;
  bgn_gam)
    ( cd variants/bgn_gam && BGN_PARAM_OVERRIDES="$OV" python -W ignore rebuild_jstar_gam.py ) 2>&1 | tee -a "$LOG" ;;
  *)
    echo "ABORT: run_solve_slurm.sh does not build model $MODEL" | tee -a "$LOG"; exit 2 ;;
esac

# Every precommitted id must be what this run printed AND must now have a manifest. Anything else
# exits non-zero, so an afterok seed array never starts on an economy its spec does not name.
python - "$SPECF" "$LOG" <<'EOF' | tee -a "$LOG"
import json, os, re, sys
spec = json.load(open(sys.argv[1])); log = open(sys.argv[2]).read()
printed = set(re.findall(r"solve_id[= ]([0-9a-f]{16})", log))
bad = []
for stage, sid in spec["expected_solves"].items():
    if sid not in printed:
        bad.append(f"{stage}: spec precommitted {sid}; this run printed {sorted(printed)}")
    if not os.path.exists(os.path.join("experiments", "registry", sid + ".json")):
        bad.append(f"{stage}: no manifest experiments/registry/{sid}.json after the build")
if bad:
    print("SOLVE MISMATCH -- the dependent seed array must not start:\n  " + "\n  ".join(bad))
    sys.exit(3)
print("SOLVE OK: " + ", ".join(f"{k}={v}" for k, v in spec["expected_solves"].items()))
EOF
