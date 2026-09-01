#!/bin/zsh
cd "$(dirname "$0")"
mkdir -p ../results/logs
PY=${PYTHON:-python3}   # any env with the repo requirements (numpy/pandas/scipy/joblib/statsmodels/sklearn/pyarrow)
export BGN_PARAM_OVERRIDES='{"gmult":[0.5,2.0],"jstar_gam_file":"Jstar_g0520.csv"}'
$PY rebuild_jstar_gam.py > ../results/logs/log_bgn_gam_jstar0520.txt 2>&1
echo "JSTAR 0520 DONE"
$PY validate_bgn_gam.py > ../results/logs/log_bgn_gam_val.txt 2>&1
echo "VALIDATION DONE"
cd ..
$PY run_oracle.py --model bgn_gam --N 500 --T 500 --tag g0520 --levels --save_panel > results/logs/log_bgn_gam_oracle.txt 2>&1
echo "BGNGAM ORACLE DONE"
$PY run_estimators.py --model bgn_gam --tag g0520 --window 360 --levels --include_mkt > results/logs/log_bgn_gam_est.txt 2>&1
echo "BGNGAM ALL DONE"
