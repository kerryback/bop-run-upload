#!/bin/zsh
# NOT A REPORTABLE RUN. One unseeded replication, so its output files carry no _sNNN and
# aggregate_seeds.py will not read them (tests/test_aggregate_seeds.py pins that). This is the
# local path for building the solve and eyeballing one panel; every reported number comes from
# ten seeds via variants/run_seeds_slurm.sh. The sample and the ridge grid below are the
# measurement protocol's (variants/common/protocol.py) and are pinned to the spec by
# tests/test_specs_match_shell.py -- this script is a THIRD copy of them, which is why it is
# checked rather than trusted.
# Extreme kp_vy: gamma_v=1.8, type_bv=[0.02,0.07,0.14] (premia ~4.6/12/22%/yr).
# Feasibility pre-checked: all A-coefficients finite, rho_ty > 0.
cd "$(dirname "$0")/.."
mkdir -p results/logs
PY=${PYTHON:-python3}   # any env with the repo requirements (numpy/pandas/scipy/joblib/statsmodels/sklearn/pyarrow)
export KP_PARAM_OVERRIDES='{"type_share":[0.34,0.33,0.33],"type_bv":[0.02,0.07,0.14],"gamma_v":1.8,"bv_comp":1.2}'
export KP_VY_PREFIX=vyx
(cd kp_vy && $PY build_vy_tables.py vyx) > results/logs/log_vyx_tables.txt 2>&1   # per-stage content-addressed cache; see variants/README.md
echo "VYX TABLES DONE"
(cd kp_vy && $PY validate_vy.py) > results/logs/log_vyx_val.txt 2>&1
echo "VYX VALIDATION DONE"
$PY run_oracle.py --model kp_vy --N 500 --T 500 --tag vyx --levels --save_panel > results/logs/log_vyx_oracle.txt 2>&1
echo "VYX ORACLE DONE"
$PY run_estimators.py --model kp_vy --tag vyx --window 360 --levels --include_mkt --kappas 1e-07,1e-06,1e-05,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0 --fair_linear > results/logs/log_vyx_est.txt 2>&1
echo "VYX ALL DONE"
