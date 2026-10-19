# Sourced, not executed. Quest has no `mamba/latest` module and no shared `bop` env, so the env
# lives under the project directory and run_seeds_slurm.sh is told where via BOP_ENV_PREFIX.
export BOP_ENV_PREFIX="${BOP_ENV_PREFIX:-/kellogg/proj/axa3203/envs/bop}"
export PATH="$BOP_ENV_PREFIX/bin:$PATH"
export BOP_RUNS="${BOP_RUNS:-/kellogg/proj/axa3203/bop_runs}"
export BOP_RESULTS_DIR="${BOP_RESULTS_DIR:-$BOP_RUNS/results}"
NT=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT
export PYTHONUNBUFFERED=1
