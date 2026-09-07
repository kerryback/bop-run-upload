# TASK for ASU — verifying the five gs_bx solves

**From:** Personal (sethjpruitt@gmail.com)
**Written:** 2026-09-07 ~14:00 MST
> **STATUS 2026-09-07 ~14:30 — this handoff could not be delivered, so Personal took
> over the parts that do not disturb the running jobs.**
>
> - §2 (pull Sol) **NOT DONE, deliberately.** No benefit while the jobs run, and it is
>   the one action that could disturb them. Sol stays at `227f5e9`. Still lossless
>   whenever someone does it.
> - §3 (verification) **IN PROGRESS, read-only.** `_scratch/watch_gs_bx.sh` polls
>   `squeue` and reads `experiments/solfiles/`; it writes nothing to Sol. Two of five
>   have landed and **both match**: `sol_b55c 645262a8e72d944c`, `sol_b70c
>   0818d7153d5708cc`. Result lands in `_scratch/GS-BX-VERIFY.md`.
> - §5 (`--mem`) **DONE and superseded — read §5 with care, its verdict was wrong.**
>   The N=500 point landed after this was written and reversed it: the flagship projects
>   to 23754 MiB, *under* the 24576 MiB cap, not over it. `--mem` was still raised to
>   64G in `3c48014` for better reasons. See WORKING.md §33.
> - §6 (pyarrow env, `method` keys, stray files) **untouched, still ASU's or Seth's.**

**Sol state when written:** all five tasks RUNNING, 3:52–4:15 elapsed against a 1-day
limit. Sol HEAD `227f5e9`; origin/main is `9497089` (four ahead).

---

## 1. Read this before you use `--spec` again

`9497089` changes how specs are enforced, and it will change what your commands do.

`runstamp.verify_against_spec` now **refuses a superseded spec** instead of appending a
NOTE. `var-gs_bx-bx7-v2` is superseded by v3, so:

```
--spec var-gs_bx-bx7-v2      # now ABORTS
--spec var-gs_bx-bx7-v3      # use this
```

That is intended — v2 is the `gs_ashift = 0.15*(gs_bx-1)` economy the running array
replaced — but it will surprise you if you have it in a script.

Second change: `run_oracle.py`'s guard went from `if _ok is False` to `if _ok is not
True`. `verify_against_spec` returns `None` for a spec it cannot check at all, and
`None is False` is False — so an **unverifiable** spec used to pass as though verified
and stamp the summary with a `spec_id` the run had not earned. Every live spec declares
`expected_solves`, so this should not bite you; it will only fire on a v1.

Why it mattered: `--spec var-kp_vy-vyx-v1` builds the *v2* economy today, because the
lambda regime-label fix was made by editing `parameters_kp14.py` in place rather than
behind a `method` switch. It would have written `var-kp_vy-vyx-v1` onto a v2 result.

---

## 2. Pulling is lossless — no stash dance needed

I checked this rather than assuming it. Sol's working copy of
`variants/gs_bx/run_gs_bx7_slurm.sh` hashes

```
12be355b2de10d3a2f8c2790b56fe19ba64c4a46818470f7074985539a8da0f9
```

which is **byte-identical** to the version committed in `a1cf71e`. Your local
modification is already upstream, so discarding it loses nothing:

```bash
cd ~/GitHub/bop-run-upload
git checkout -- variants/gs_bx/run_gs_bx7_slurm.sh
git pull                      # 227f5e9 -> 9497089
```

Safe mid-run. The four commits are docs, dependency declarations, an unhashed
`environment()` record, and the spec guard above — none changes a `solve_id`, and the
five running processes already hold their code in memory.

There is also no *need* to pull before the solves land. If you'd rather touch nothing
until they finish, that is fine.

---

## 3. When the five land — the verification

This is the cross-machine reproducibility test the precommitment was built for. The
five ids below were computed on the laptop **without solving**, from parameters and
source alone.

| stage | expected `solve_id` | gs_bx |
|---|---|---|
| `sol_reg`  | `63fa7ebbc2db49ea` | 1.0 |
| `sol_b25c` | `649bb384300faadf` | 2.5 |
| `sol_b40c` | `a3bb50b66287307c` | 4.0 |
| `sol_b55c` | `645262a8e72d944c` | 5.5 |
| `sol_b70c` | `0818d7153d5708cc` | 7.0 |

All five carry `gs_ashift: 0.0`, `gmreg: [0.6, 3.0]`, `xnum: 161`, `tol: 1e-06`.

```bash
python variants/solfiles.py list      # five new entries?
python variants/solfiles.py check     # artifacts present and intact?
```

Then compare each recorded id against the table.

- **All five match** → clear `"solves_pending": true` from
  `experiments/specs/var-gs_bx-bx7-v3.json`, commit, push.
- **Any mismatch** → **stop, and do not clear it.** A mismatch means Sol and the laptop
  disagree on a `solve_id`, which breaks content-addressing itself. Chase that before
  any number from this economy is used. Do not run a panel on a mismatching solve.

---

## 4. Already checked — please don't redo

**Task 0 is running pre-`227f5e9` code, and that is fine.** Array `62740640` was
submitted 09:45:30; the `gs_ashift = 0` commit landed 10:03:22. You ended tasks 1–4 at
10:07 and resubmitted them as `62740930` at 10:08:22, leaving task 0 running. That was
the right call, on two independent grounds:

1. `gs_bx = 1.0` makes `0.15*(gs_bx-1)` equal to 0, so `sol_reg`'s ashift is zero under
   the *old* formula too. The solve is identical either way.
2. `227f5e9` never touched `gs_solve_reg.py` — it changed only WORKING.md, the two
   specs, and `gs_sim_bx.py` (the simulator). So the solver's source digest is
   unchanged and there is no provenance hazard from the mid-run edit.

I went looking for a hazard here and there isn't one. `sol_reg 63fa7ebbc2db49ea` is the
independent assertion; if it matches, both grounds are confirmed empirically.

**The array is running exactly the committed parameters** — see the hash in §2.

---

## 5. What I'm doing, so we don't collide

Local scaling ladder on `kp_vy`, to size `--mem` and walltime for
`variants/run_seeds_slurm.sh`. Nothing here touches gs_bx, and **I am not submitting
anything to Sol.**

| N | T | wall | max RSS |
|---|---|---|---|
| 100 | 200 | 281 s | 4772 MiB |
| 200 | 200 | 413 s | 5268 MiB |
| 300 | 200 | 552 s | 7559 MiB |
| 100 | 500 | 678 s | 6323 MiB |
| 200 | 500 | 1036 s | 10969 MiB |
| 500 | 200 | *running* | |

Wall is affine in N at fixed T (`145 + 1.351*N`, fitting the middle point to 0.5%) and
T-linear at N=100 (2.41x for a 2.5x T). The bilinear model predicted the N=200/T=500
point at 1038 s against 1036 s measured.

### `--mem=24G` IS NOT ENOUGH. Raise it before you submit the seed array.

The two T=500 points give the flagship directly, with **no cross-dimension
extrapolation** — that was the point of measuring T=500 rather than scaling T=200 up:

    T=500:  wall = 320 + 3.578*N s      maxRSS = 1677 + 46.46*N MiB
    N=500:  wall = 2110 s (35 min)      maxRSS = 24907 MiB = 24.32 GiB

`--mem=24G` is 24576 MiB. The projection **exceeds it by 331 MiB (1.3%)**.

A 1.3% overrun is a dead heat, and a dead heat loses here, for two reasons:

1. **It is a lower bound.** It assumes memory stays linear in N up to 500, which is
   exactly the assumption the wall-clock data warns against above N=300.
2. **The T-scaling is already superlinear.** The memory N-slope went 13.94 -> 46.46
   MiB/N for a 2.5x T — a factor of **3.33**, not 2.5. Whatever drives that is not
   captured by either fit.

And the asymmetry in `run_seeds_slurm.sh:74` applies with full force: an OOM at hour 3
writes nothing, the oracle has no mid-run checkpoint, and the seed is simply lost.
Over-requesting on an empty queue costs approximately nothing.

**Recommendation: `--mem=64G`.** Tighten later from an actual `MaxRSS`, not from this.

Walltime is fine as it stands: 35 min projected against `-t 2-00:00`. Deliberately
over-provisioned, and that was the right call — leave it until a real `Elapsed` exists.

**This does not refute the `~T*N^2` comment in `run_seeds_slurm.sh:65`, and I said
otherwise earlier today — that was wrong.** The measurement behind N^2
(WORKING.md:1284, bgn_gam, `--nmat 2`) had per-month cost 0.70 / 1.53 / 18.02 s at
N = 100 / 200 / 500: linear from 100 to 200, then 12x over the next 2.5x of N. The
blowup lives *above* N=300, entirely outside my ladder's range. The queued N=500 run is
what actually tests it. Until it lands, **do not tighten `-t` or `--mem` on my numbers.**

---

## 6. Open, and yours if you want them

- **pyarrow env.** Recommend cloning rather than mutating `bop` while an array is
  running: `mamba create -n bop-arrow --clone bop && mamba install -n bop-arrow pyarrow
  openpyxl`. `run_seeds_slurm.sh` reads `CONDA_ENV=${CONDA_ENV:-bop}`, so the clone is
  selected by env var with no code change.
- **The `method` keys are inert.** Seven are declared across the specs
  (`kp_interpolation`, `kp_cir_quadrature`, `kp_regime_labels`,
  `zero_book_in_sdf_solve`, `gs21_calibration`, `gs21_discretization`,
  `gs_ashift_ladder`) and **zero** are read by any python file. `method` is also not
  hashed into `solve_id`, so two specs differing only in `method` produce identical ids
  and identical results while claiming to be different experiments. `9497089` stops the
  silent mislabelling; making the switches executable is Seth's call and is not started.
- **Stray files on Sol:** `/data/sjpruitt/GitHub/bop-run-upload/0` (2 bytes, Aug 31 —
  looks like a `2>0` redirect typo) and `t3_bgn.sh` (Aug 25). Both predate today's work.
  Delete or track them at your discretion.

---

*— Personal*
