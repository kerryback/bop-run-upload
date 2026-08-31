# GS21 port validation against GS21.m, via Octave

`utils_gs21/gs21_solve.py` replaced `GS21.m` as GS21's producer (`af67c5b`). This is the
harness that validated it. It is **archival**: nothing in the pipeline imports it, and the
pipeline does not need Octave.

Re-run it if you touch `gs21_solve.py`'s operator — `expectation`, `by_bprime`,
`shock_survive`/`shock_max`, `update_cutoffs`, any reshape, or the spline calls. It is the only
thing that can catch a shape-preserving algebra bug, and two such bugs were found this way
(a transposed `einsum` on the SDF kernel; `Q_I` vs `Q_I_no` at `GS21.m:257`) — neither changed
any array dimension, and the solve converged happily on the wrong answer.

## Run it

```bash
brew install octave          # once
./run.sh                     # ~2 min, honours OMP_NUM_THREADS (default 2)
```

Expected tail: `ALL CHECKS PASSED`.

## Why validate against GS21.m and not against its output

`GS21.m`'s committed solfiles are **not** a converged solution — its nested alternation sits
on an exact period-2 orbit and its outer loop has no break, so the files are an interrupted
run at an unknown iteration. Comparing fixed points would confound a port bug with how far
that run got. So the harness runs `GS21.m`'s *expressions*, verbatim, on identical inputs.

Two design choices make it much stronger than a solfile comparison:

- **Random `P` of both signs.** In the committed files `P > 0` everywhere, so `z_cut` is pinned
  to `zgrid[0]`, no default ever occurs, and `cond0` in `update_cutoffs` is unreachable. The
  2.8e-15 `z_cut` agreement against those files is **vacuous**. Random inputs exercise the
  branches.
- **A small grid.** At the shipped size `Mmat` and `pr_mat_re` are 2.56 GB *each*, so
  `GS21.m`'s dense form is not runnable at all. At 4x5x6 they are ~30 KB. The algebra under
  test is size-independent; asymmetric dims (3x7x4) are included because that is what catches
  transposes and `kron`-ordering errors.

`Setup(..., quad='gh')` is used throughout, because the point is to reproduce `GS21.m`'s
360-node Gauss-Hermite rule. **The shipped default is `quad='exact'`** — closed-form
truncated-normal expectations, which are smooth and exact where the GH rule is a step function
wrong by up to 4.3e-3. That difference is deliberate and is what makes the solve converge; see
`gs21_solve.py`'s docstring.

## What each file does

| file | checks |
|---|---|
| `py_ref.py` + `oct_ref.m` + `cmp.py` | 26 intermediates of one debt pass and one price pass |
| `fuzz_cutoffs.py` + `fz_run.m` | `update_cutoffs` over 60 random cases, every branch |
| `check_tauchen_and_gh.py` + `tch_run.m`, `gh_run.m` | `tauchen.py` vs `tauchen.m`; scipy vs Golub-Welsch Gauss-Hermite |
| `oct_loop.m` | **not** a port comparison — reproduces the period-2 orbit in MATLAB itself |
| `update_cutoffs.m` | `GS21.m:395-408` verbatim, shared by both harnesses |
| `run.sh` | driver for the first three |

Results as of `af67c5b`:

```
operator, 4 input regimes            worst 9.6e-15 relative
update_cutoffs, 60 cases             BIT-EXACT (0.0)
  cond0 202x, cond2 134x, interior 369x, non-finite guard 486x, 1255 exact ties
tauchen, 5 cases incl. production    <= 4.2e-15
Gauss-Hermite n=20000                same 360 nodes, weights 1.05e-15
```

## `oct_loop.m` — the orbit, in MATLAB

`gs21_solve.py` no longer uses `GS21.m`'s nested two-loop scheme, so this is not a comparison
harness. It exists so the central negative result is reproducible without trusting the Python:
run `GS21.m`'s own loop and watch `|P_k - P_k-1|` pin to a constant while `|P_k - P_k-2|`
collapses to ~1e-13. That is an exact period-2 orbit, and it is why the committed solfiles were
an interrupted run.

It needs `loop_args.m` (`maxouter`, `maxit`) and the inputs `py_ref.py` dumps:

```bash
T_BNUM=20 T_XNUM=20 T_ZNUM=6 python py_ref.py
echo "maxouter=20; maxit=1000;" > loop_args.m
octave --no-gui --quiet oct_loop.m
```

**`xnum` must be 20.** The operator contracts only while `i_cut << imax`; a coarse x grid lets
`i_cut` saturate, flipping the gain from `E[M] = 0.992` to `g*E[M] = 1.131`, and the solve
diverges outright — in both implementations, identically. `znum` is only cost and can be cut.

## Gotcha if you extend this

The harnesses load `gs21_solve.py` **by path**, not as `utils_gs21.gs21_solve`. Importing the
package runs `utils_gs21/__init__.py`, which imports the consumers, which
`verify(mode='error')` at import time — and `py_ref.py` deliberately shrinks the grid dims in
`config`, so that check correctly fires and aborts the harness. Keep the bypass.

`*.csv` and the generated `.m` argument files are gitignored; only the harness sources are
tracked.
