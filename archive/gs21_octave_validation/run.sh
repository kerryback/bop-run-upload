#!/bin/bash
# Run the GS21 port validation suite. Needs octave on PATH (brew install octave).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
fail=0

echo "=== 1. operator: one debt pass + one price pass vs GS21.m, dense form ==="
run_case () {
    python py_ref.py > /dev/null 2>&1 || { echo "  py_ref.py FAILED"; return 1; }
    octave --no-gui --quiet oct_ref.m > /dev/null 2>&1 || { echo "  oct_ref.m FAILED"; return 1; }
    python cmp.py "$1"
}
T_BNUM=4 T_XNUM=5 T_ZNUM=6  T_SEED=20260826 T_PSCALE=8  T_PMEAN=0    T_SIGMA_M=5   run_case "baseline 4x5x6"      || fail=1
T_BNUM=4 T_XNUM=5 T_ZNUM=6  T_SEED=7        T_PSCALE=0.5 T_PMEAN=-0.3 T_SIGMA_M=0.4 run_case "small shocks, P<0"   || fail=1
T_BNUM=4 T_XNUM=5 T_ZNUM=6  T_SEED=99       T_PSCALE=30 T_PMEAN=-25  T_SIGMA_M=5   run_case "big neg P, big shock" || fail=1
T_BNUM=3 T_XNUM=7 T_ZNUM=4  T_SEED=1234     T_PSCALE=2  T_PMEAN=-1   T_SIGMA_M=1.5 run_case "asym dims 3x7x4"     || fail=1

echo
echo "=== 2. update_cutoffs fuzz, every branch ==="
python fuzz_cutoffs.py || fail=1

echo
echo "=== 3. tauchen + Gauss-Hermite ==="
python check_tauchen_and_gh.py || fail=1

echo
if [ $fail -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "*** SOMETHING FAILED ***"; fi
exit $fail
