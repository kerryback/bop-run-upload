# gs_vy — GS21 with a priced y chain and firm exposure types

The `kp_vy` construction in GS21 (2026-10-07). The unpriced two-state regime of `gs_bx/gs_solve_reg.py` is
replaced by an NY-node Tauchen chain for y whose transitions are PRICED: `My = exp(-lam_y eta - lam_y^2/2)`,
renormalised on the grid, `lam_y = gamma_v sqrt(1/12)`. A type's production loads as
`exp(gs_bx x + z + gs_bv y + gs_ashift)`, with `gs_ashift = -bv_comp log W(gs_bv)` the y=0 level
compensation (W = the y-premium discount of a level perpetuity at r). Unlike gs_bx's old ladder the
simulator applies `gs_ashift` and refuses a mismatch. Each type is one solve (hours).

| file | what |
|---|---|
| `gs_solve_vy.py` | the solver: `python gs_solve_vy.py <xnum> <tol> <outdir>` with GS_PARAM_OVERRIDES (gs_bv, gamma_v, kappa_y, NY, y_mult, bv_comp) |
| `gs_sim_vy.py` | simulator + exact conditional moments; GS_VY_SOLDIRS / GS_VY_BV / GS_VY_SHARES; exports `rf_stand` (x) and `y_stand` |
| `validate_gs_vy.py` | E[M R] = 1 on the typed economy, realised vs expected by type, SR by y-tercile |
| `check_parity.py` | NY = 1 against gs_solve_reg at gmreg [1, 1], low resolution |

Solve on Quest: `export GSPEC=var-gs_vy-gvy25-v1; sbatch --export=ALL --array=0-2 variants/klc/solve_gs_vy_klc.sh`,
then `variants/klc/verify_gs_vy_klc.sh` (afterok; fills the ids and runs the gate), then `SPECS=gvy25` seeds.
