# Quickstart — how this project runs experiments

**Read this first.** It orients you in the three documents that carry the research, states the
rules that are not negotiable, and walks the loop we follow every time we try a new economy. It
does not duplicate the detail in those documents; it tells you which one to open and when.

The question the project is trying to answer: **in a simulated economy where we know the true
stochastic discount factor, does a high-complexity method (DKKM's random Fourier features plus
ridge) beat a fair linear benchmark (Fama-French, Fama-MacBeth, linear-in-ranks, and the market)?**
So far, in sixteen engineered economies, two say yes by a small margin and fourteen do not.

---

## 1. The three documents

| file | what it holds | when you open it |
|---|---|---|
| [`RESULTS.md`](RESULTS.md) | Every economy's numbers, what each was built to test, what it decided, and the numbered cross-cutting **findings**. Also the measurement protocol, and "Adding an experiment", which is the authoritative procedure | whenever you want a number, or want to know whether a question is already answered |
| [`NEXTUP.md`](NEXTUP.md) | The live queue, and below it every experiment already run, dropped or withdrawn, each with its **pre-registration and its grade** | when deciding what to run next, and before proposing anything — check it is not already struck |
| [`RUNS.md`](RUNS.md) | Cluster procedure, the hazards that have cost real money, and the **measured cost** of every campaign | before submitting anything, and again when a result's provenance is in question |

`RESULTS.md` cannot fall behind the data: `tests/test_results_md_matches_table.py` checks every
table in it cell by cell against `variants/results/economy_table.csv`, and
`tests/test_live_docs_cite_live_economies.py` refuses prose that cites an economy or a route that
no longer exists. If you edit numbers by hand, the suite tells you.

The operational detail of the pipeline itself — the estimators, the feature bases, the solve
registry — is in [`../variants/README.md`](../variants/README.md).

---

## 2. The measurement protocol, which is not yours to change

One protocol governs every economy, so that the only thing separating two rows of a table is the
economy. It lives in `variants/common/protocol.py`:

```
N = 500 firms        T = 500 months retained      burn-in = 400 months
window = 360         eval months = 125            10 seeds
ridge grid = 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000
```

**A new economy is a parameter override and nothing else.** If what you want to change is the
sample, the window or the ridge grid, you are proposing a change to the protocol, which moves every
row in `RESULTS.md` and is not an experiment. `tests/test_protocol_is_uniform.py` refuses any spec
or runner case that departs from these values, and `variants/run_seeds_slurm.sh` refuses to write
an off-protocol run into `variants/results` at all unless it names its own `BOP_RESULTS_DIR` — the
mechanism the N=1000 probe used (finding 17).

---

## 3. The loop

The authoritative version is **"Adding an experiment" at the end of [`RESULTS.md`](RESULTS.md)**.
Follow that. This is the shape of it, and why each step is where it is.

**1 — Propose, with a prediction AND a falsifier.** Write
`experiments/specs/var-<model>-<tag>-v1.json`: the parameter override, what differs from the
baseline *in economic terms*, a registered prediction, and the condition that would make you say
the idea is wrong. The falsifier is the part that earns the compute. Four of our last four gates
came back against the proposal, and they were worth running because each one had a stated
falsifier and so each one closed something.

**2 — Precommit the solve ids WITHOUT solving.** Compute the ids the spec expects, commit the spec,
*then* build. This is what makes a result a prediction rather than a description, and
`tests/test_precommitment_is_real.py` checks the order by git dates — an id read off an existing
manifest and pasted in proves nothing.

```bash
python variants/precommit_spec.py experiments/specs/var-kp_vy-foo-v1.json
python variants/precommit_spec.py <spec> --build-chained --write    # writes expected_solves
```

It runs each producer inside a throwaway git worktree, so the registry and tables it creates are
discarded and never touch your tree. Most ids are free: a `solve_id` is
`sha256(parameters + source digests)` and every producer prints it with `flush=True` *before* it
starts solving, so the tool reads it and stops. That is how all five GS21 ids come back in seconds
instead of five hours each.

**One stage genuinely cannot be free.** KP14's `integ` id hashes the G tables' **raw bytes**
(`build_vy_tables.py`, `inputs=solstamp.artifact_digests(G_ARTIFACTS)`) — that is what makes the
chain tamper-evident — so it is unknowable until G exists. `--build-chained` pays for that build
inside the throwaway tree. The practical consequence: **whoever builds G must be the one who ships
it.** Derive the integ id from your own G bytes, commit those exact tables, and everyone downstream
verifies against them. If someone later rebuilds G elsewhere, that integ id moves, through no error.

The tool refuses to run if `variants/` has uncommitted changes to tracked files — an id computed
from code that is not what lands is not a precommitment.

**3 — Build, and verify every printed id against its pin.** If an id does not match, stop: either
the parameters are not what the spec says or a producer changed. Clear `solves_pending`, commit the
tables and the manifests.

**4 — Wire the runner.** Add a `SEED_SPEC` case to `variants/run_seeds_slurm.sh` carrying the
economy's parameters and nothing else, plus a row in `variants/submit_campaign.sh` with its memory
and the reason. `tests/test_specs_match_shell.py` pins the case to the spec.

**5 — Ten seeds, and all ten before any of them is aggregated.** Result filenames carry no spec
version, so a partial re-run silently averages two spec versions into one `MIXED:` row at
`n_seeds = 10` and no test catches it. If the economy's peak memory has never been measured, run
**one seed and read `sacct`** before sizing the array.

**6 — `python variants/penalty_gate.py` BEFORE reading any number.** It asks whether the winning
ridge penalty sat at the edge of the grid, i.e. whether the reported DKKM Sharpe is set by where
the grid stops rather than by the economy. Then `python variants/aggregate_seeds.py`.

**7 — Grade against the pre-registration, then write it up.** Say which clauses held and which
failed, in `NEXTUP.md` where the prediction is, and add the finding to `RESULTS.md`. A clause that
fails informatively is the most valuable thing a run produces — finding 16 and finding 17 are both
failed predictions.

### Specs are precommitment records

**Supersede; never retire, and never edit a spec's hashed fields.** A superseded spec gets
`lineage.superseded_by`, the new one gets `lineage.supersedes`. `spec_hash` deliberately excludes
`title`, `question`, `notes`, `lineage`, `provenance`, `solves_pending`, `precommitted` and
`reused_solves` — those are status and commentary, and may be corrected. Everything else defines
the economy and may not.

---

## 4. Before you edit a producer

`solve_id` digests a producer's **whole source file**, so a comment or a docstring edit silently
invalidates every cached solve built from it. That has happened: a rename's `sed` rewrote one
string inside a `print()` and a precommitted id moved, at a cost of five solves of ~5 h each.

The seven digest-bearing files are:

```
variants/bgn_gam/vasicek.py           variants/bgn_gam/parameters.py
variants/kp_vy/kp14_fd_vy.py          variants/kp_vy/parameters_kp14.py
variants/kp_vy/integ_kp14.py
variants/gs_bx/gs_solve_reg.py        variants/gs_bx/gs_solve_gam.py
```

```bash
git config core.hooksPath hooks       # once per clone; the hook is advisory, never blocks
python variants/solve_impact.py       # which solves does this change invalidate, and was it functional?
```

**Do not run bulk `sed` over solver sources.**

---

## 5. Getting the solves without paying for them

**Solve artifacts live in two places, and one command covers both.**

| | | |
|---|---|---|
| **in git** | 17 solves, 81 MB | arrives with the clone: every BGN and KP14 table |
| **published** | 11 solves, 1,090 MB | the GS21 `solution.npz` files, 92-105 MB each |

The rule is a size threshold — `SMALL_ARTIFACT_BYTES` in `variants/common/solstamp.py`,
currently 64 MB — and each manifest records which side its solve fell on in
`committable`. Nothing in the repository sits near the line: the largest committed solve
is 38.1 MB and the smallest published one is 92.1 MB.

**Why the split rather than one location.** A table committed beside the spec that pins
it and the manifest that describes it means `git checkout <commit>` gives a tree where
parameters, solve id, bytes and results all agree — that is what makes an old result
reproducible rather than merely remembered. Dropbox has no commits and no integrity
check, so anything kept only there cannot be checked out with an old revision. The GS21
solutions are published only because at ~100 MB each git would keep every re-solve of
them forever.

**You do not need to know which is which.** `fetch_solves.py` reads each spec's
`expected_solves`, resolves them through the manifests, copies whatever is missing and
verifies the sha256 of everything it copies.

### Telling it where the shared folder is

The published artifacts are in a Dropbox folder shared with you, laid out as
`<folder>/<solve_id>/<basename>`. **That folder sits at a different absolute path on
every machine that syncs it**, so the location is configuration, not code. Set it once,
in `~/.zshrc` or `~/.bashrc`:

```bash
export BOP_SOLVES_DIR="$HOME/<Your Org> Dropbox/<Your Name>/BGN and Kelly Malamud/solves"
```

To find yours: the folder is shared as **BGN and Kelly Malamud**, and Dropbox puts it
under your own account root — on macOS usually `~/<Org> Dropbox/<Your Name>/`, on Linux
or a personal account usually `~/Dropbox/`. Confirm with:

```bash
find ~ -maxdepth 4 -type d -name solves -path "*BGN and Kelly Malamud*" 2>/dev/null
```

Then:

```bash
python variants/fetch_solves.py --all            # fetches, using $BOP_SOLVES_DIR
python variants/fetch_solves.py --all --check    # report only, fetch nothing
python variants/fetch_solves.py --all --from "/some/other/path"   # override for one run
```

`--from` beats the environment variable, and with neither set the command reports what
is missing and fetches nothing. If the path is wrong the tool says so and stops rather
than reporting everything as missing.

If you are working on the BGN or KP14 economies, which is where the queue currently
points, the clone alone is enough and you never need the folder at all.

**Do not re-solve instead.** The five GS21 exposure types are about five hours each on a cluster
node. A `solve_id` is deliberately independent of the library stack, so two machines that agree on
an id can still differ in the last digits of a table; `fetch_solves.py` reads the id the solver
embedded in the file and only complains when *that* disagrees.

---

## 6. Running on your own cluster

See **"Running this somewhere other than ASU"** in [`RUNS.md`](RUNS.md) for the table of what is
site-specific. The short version: it is plain SLURM, and the four things to translate are the
partition/QoS/account names, your module system's conda activation (**keep the
`export PATH="$CONDA_PREFIX/bin:$PATH"` line after `source activate` — without it every task dies
in about a second with `ModuleNotFoundError: numpy`**), a partition whose wall clock allows 6–8 h
per seed, and your own scratch paths. Nothing about the protocol, the precommitment order, the
penalty gate or the solve registry is site-specific.

---

## 7. First day, in order

```bash
conda env create -f environment.yml && conda activate bop   # or: pip install -r requirements.txt pyarrow
git config core.hooksPath hooks
python -m pytest tests/ -q                                  # expect 275 passed

python variants/fetch_solves.py --all                       # what artifacts am I missing?
python variants/aggregate_seeds.py                          # rebuild the results table from what is committed
python variants/penalty_gate.py                             # expect GATE PASSED
```

Then read [`RESULTS.md`](RESULTS.md) "The answer so far", and
[`NEXTUP.md`](NEXTUP.md) from the top — its first section states which levers are closed and which
experiment is next, which is the fastest way to know where the project actually is.
