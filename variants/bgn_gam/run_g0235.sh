#!/bin/zsh
# NOT A REPORTABLE RUN. One unseeded replication, so its output files carry no _sNNN and
# aggregate_seeds.py will not read them (tests/test_aggregate_seeds.py pins that). This is the
# local path for building the solve and eyeballing one panel; every reported number comes from
# ten seeds via variants/run_seeds_slurm.sh. The sample and the ridge grid below are the
# measurement protocol's (variants/common/protocol.py) and are pinned to the spec by
# tests/test_specs_match_shell.py -- this script is a THIRD copy of them, which is why it is
# checked rather than trusted.
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
$PY run_estimators.py --model bgn_gam --tag g0235 --window 360 --levels --include_mkt --kappas 1e-05,0.0001,0.001,0.01,0.1,1.0,10.0 --fair_linear > results/logs/log_g0235_est.txt 2>&1
echo "G0235 ALL DONE"
