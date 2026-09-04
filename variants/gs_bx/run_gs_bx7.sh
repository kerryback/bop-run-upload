#!/bin/zsh
cd "$(dirname "$0")"
mkdir -p ../results/logs
PY=${PYTHON:-python3}   # any env with the repo requirements (numpy/pandas/scipy/joblib/statsmodels/sklearn/pyarrow)
# 2026-09-04: the old guard here was `[ -f sol_reg/solution.npz ] || ...` -- a bare
# existence check, blind to parameters, so sol_reg was silently reused across
# different economies. gs_solve_reg.py now caches on its own content-addressed
# solve_id (variants/common/solstamp.py) and exits early on a genuine hit, so the
# guard is both unnecessary and unsafe. Set GS_SOLVE_FORCE=1 to re-solve anyway.
GS_PARAM_OVERRIDES='{"gmreg":[0.6,3.0]}' $PY -W ignore gs_solve_reg.py 161 1e-6 sol_reg > ../results/logs/log_gs_reg_solve.txt 2>&1
echo "SOL_REG DONE"
GS_PARAM_OVERRIDES='{"gmreg":[0.6,3.0],"gs_bx":2.5,"gs_ashift":0.225}' $PY -W ignore gs_solve_reg.py 161 1e-6 sol_b25c > ../results/logs/log_gs_b25c.txt 2>&1
echo "B25 DONE"
GS_PARAM_OVERRIDES='{"gmreg":[0.6,3.0],"gs_bx":4.0,"gs_ashift":0.450}' $PY -W ignore gs_solve_reg.py 161 1e-6 sol_b40c > ../results/logs/log_gs_b40c.txt 2>&1
echo "B40 DONE"
GS_PARAM_OVERRIDES='{"gmreg":[0.6,3.0],"gs_bx":5.5,"gs_ashift":0.675}' $PY -W ignore gs_solve_reg.py 161 1e-6 sol_b55c > ../results/logs/log_gs_b55c.txt 2>&1
echo "B55 DONE"
GS_PARAM_OVERRIDES='{"gmreg":[0.6,3.0],"gs_bx":7.0,"gs_ashift":0.900}' $PY -W ignore gs_solve_reg.py 161 1e-6 sol_b70c > ../results/logs/log_gs_b70c.txt 2>&1
echo "GS BX7 SOLVES DONE"
cd ..
export GS_BX_SOLDIRS="sol_reg,sol_b25c,sol_b40c,sol_b55c,sol_b70c"
export GS_BX_BETAS="1.0,2.5,4.0,5.5,7.0"
export GS_BX_SHARES="0.2,0.2,0.2,0.2,0.2"
$PY run_oracle.py --model gs_bx --N 500 --T 500 --tag bx7 --levels --save_panel > results/logs/log_gs_bx7_oracle.txt 2>&1
echo "BX7 ORACLE DONE"
$PY run_estimators.py --model gs_bx --tag bx7 --window 360 --levels --include_mkt --kappas 0.001,0.01,0.03,0.1,0.3,1,3,10 > results/logs/log_gs_bx7_est.txt 2>&1
echo "BX7 ALL DONE"
