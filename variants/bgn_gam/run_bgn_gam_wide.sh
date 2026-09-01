#!/bin/zsh
cd "$(dirname "$0")/.."
mkdir -p results/logs
PY=${PYTHON:-python3}   # any env with the repo requirements (numpy/pandas/scipy/joblib/statsmodels/sklearn/pyarrow)
export BGN_PARAM_OVERRIDES='{"gmult":[0.3,3.0],"jstar_gam_file":"Jstar_g0330.csv"}'
(cd bgn_gam && $PY rebuild_jstar_gam.py) > results/logs/log_bgn_gam_jstar0330.txt 2>&1
(cd bgn_gam && $PY validate_bgn_gam.py) > results/logs/log_bgn_gam_val0330.txt 2>&1
$PY run_oracle.py --model bgn_gam --N 500 --T 500 --tag g0330 --levels --save_panel > results/logs/log_bgn_gam_oracle0330.txt 2>&1
echo "BGNGAM WIDE ORACLE DONE"
$PY run_estimators.py --model bgn_gam --tag g0330 --window 360 --levels --include_mkt > results/logs/log_bgn_gam_est0330.txt 2>&1
echo "BGNGAM WIDE ALL DONE"
