# Dead end: extra aggregate factors at the cash-flow shock (BGN)

Abandoned 2026-08-25. **These scripts will not run against the current repo** — they
import `config.BGN_N_EXTRA_FACTORS`, `BGN_EXTRA_OMEGA`, `BGN_EXTRA_DECAY` and
`BGN_SIGMAJ_BAND`, all of which were rolled back.

To reproduce, apply the preserved patch from the repo root:

```bash
git apply voc_diagnosis/dead_end_K/K_factors.patch
python voc_diagnosis/dead_end_K/check_h_srlin.py     # the decisive table
```

`K_factors.patch` is the full working implementation: the four config knobs, the
Fourier-mode loading construction in `panel_functions_bgn.create_arrays` (append-only
draws, `gload` appended to `arr_tuple`), and the `extra=` path in `_term4_gram`. It passed
every correctness test — bit-identical panel at K=0, `rp` invariant to 0.000e+00 at K=40.
It was abandoned because it does not move the objective, not because it is wrong.

The numbers and the reason are in `../README.md` ("Dead ends") and
`../refactor_checks/RESULTS.md` Part 2.

The speed-up discovered alongside it was retained: see `../refactor_checks/`.
