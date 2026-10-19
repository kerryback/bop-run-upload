# bgn_vy — BGN with a priced AR(1) state y and firm exposure types

The `kp_vy` construction one model over (2026-10-07). `y_{t+1} = rho_y y_t + s_y eta`, unit stationary
sd, `rho_y = exp(-kappa_y/12)`; the SDF gains `exp(-lam_y eta - lam_y^2/2)` with `lam_y = gamma_v sqrt(1/12)`
(so `gamma_v` is KP14's price per unit Brownian per sqrt(year)); a type-f firm's project cash flows are
multiplied by `e^{b_f y}`, and `type_theta[f] = (D_0/D_f at (rbar, 0))^bv_comp` compensates the level so that
at y = 0 value does not reveal the type (`bv_comp` as in kp_vy). `gmult = [1, 1]` keeps bgn_gam's regime
inert; the module nests bgn_gam and reduces to it at `type_bv = [0]` (`check_reduction.py`).

| file | what |
|---|---|
| `parameters.py` | bgn_gam's calibration + the y block; overrides that have no effect raise (as parameters_kp14.py) |
| `vasicek_vy.py` | bond pricing and the beta distribution (verbatim), the bases `DA_f(r, y, s)`, `DB_f` with the closed-form y factor, the option value, and `J*_f(r, y, s)` as a dynamic programme on a 321 x (dy/4) grid |
| `rebuild_jstar_vy.py` | the producer: writes `jstar_vy_file` (npz), content-addressed via solstamp; JSTAR_TOL hashed |
| `sdf_compute.py` | true conditional moments: every table is (type, regime)-indexed; y'-expectations by quadrature first, r'-expectations second, so every table is 2-D in (E_t r', y) |
| `panel_functions.py` | the panel; firm shocks from the global stream in bgn_gam's order, y from `default_rng([gam_seed, y_seed_tag])`, types from `bx_seed`; exports `y_stand` |
| `validate_bgn_vy.py` | the project-value Euler equation WITH y, the J* fixed-point residual, realised vs expected by regime and type |
| `check_reduction.py` | bgn_vy at zero exposure against bgn_gam on one seed |

Solve on Quest: `export BSPECS=var-bgn_vy-bvy25-v1; sbatch --export=ALL variants/klc/solve_bgn_vy_klc.sh`
(build + id fill + gate), then `SPECS=bvy25` through `variants/klc/seeds_klc.sh`.
