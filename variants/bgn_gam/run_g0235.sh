#!/bin/zsh
# Extreme BGN regime economy: gmult=[0.2,3.5] (closed-form frontier is gmult<3.65).
cd "$(dirname "$0")/.."
mkdir -p results/logs
PY=${PYTHON:-python3}   # any env with the repo requirements (numpy/pandas/scipy/joblib/statsmodels/sklearn/pyarrow)
export BGN_PARAM_OVERRIDES='{"gmult":[0.2,3.5],"jstar_gam_file":"Jstar_g0235.csv"}'
(cd bgn_gam && $PY -W ignore rebuild_jstar_gam.py) > results/logs/log_g0235_jstar.txt 2>&1
echo "G0235 JSTAR DONE"
(cd bgn_gam && $PY validate_bgn_gam.py) > results/logs/log_g0235_val.txt 2>&1
echo "G0235 VALIDATION DONE"
$PY run_oracle.py --model bgn_gam --N 500 --T 500 --tag g0235 --levels --save_panel > results/logs/log_g0235_oracle.txt 2>&1
echo "G0235 ORACLE DONE"
$PY run_estimators.py --model bgn_gam --tag g0235 --window 360 --levels --include_mkt --kappas 0.001,0.01,0.1,1 > results/logs/log_g0235_est.txt 2>&1
echo "G0235 ALL DONE"
