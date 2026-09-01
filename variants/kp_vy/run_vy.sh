#!/bin/zsh
cd "$(dirname "$0")/.."
mkdir -p results/logs
PY=${PYTHON:-python3}   # any env with the repo requirements (numpy/pandas/scipy/joblib/statsmodels/sklearn/pyarrow)
export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.06,0.12],"gamma_v":1.2,"bv_comp":1.2}'
(cd kp_vy && $PY build_vy_tables.py vys) > results/logs/log_vy_tables.txt 2>&1   # cached if meta_vys.json matches
echo "VY TABLES DONE"
(cd kp_vy && $PY validate_vy.py) > results/logs/log_vy_val.txt 2>&1
echo "VY VALIDATION PHASE DONE"
$PY run_oracle.py --model kp_vy --N 500 --T 500 --tag vys --levels --save_panel > results/logs/log_vy_oracle.txt 2>&1
echo "VY ORACLE DONE"
$PY run_estimators.py --model kp_vy --tag vys --window 360 --levels --include_mkt --kappas 0.001,0.01,0.03,0.1,0.3,1,3,10 > results/logs/log_vy_est.txt 2>&1
echo "VY ALL DONE"
