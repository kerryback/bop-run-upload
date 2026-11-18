# KP14 oracle probes with no spec (Quest, 2026-10-07)

Two-seed `run_oracle.py` probes, tags sal75, sg30, sgx95, sgz15 and ssl30: vyg25-type three-type
ladders at other prices and loadings (e.g. sg30 is gamma_v 3.0). They are on the protocol's sample
(N = 500, T = 500, evaluation window 360) but were never estimated and no spec names them. Each
`.prov.json` records the exact overrides.

They were moved here from `variants/results/` on 2026-11-18. The canonical directory's rows must each
name their spec and solves (`tests/test_aggregate_seeds.py`), and an unspecified probe cannot. To read
them, point `BOP_RESULTS_DIR` (or `aggregate_seeds.py --results`) at this directory.
