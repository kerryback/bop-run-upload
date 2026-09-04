"""Build per-type G and per-(type, y-node) integral tables for kp_vy.
usage: KP_PARAM_OVERRIDES='{...}' python build_vy_tables.py <prefix>

TWO STAGES, INDEPENDENTLY CACHED
--------------------------------
  G      kp14_fd_vy.py -> G_<prefix><f>.csv          EXPENSIVE (~45 min per type)
  integ  integ_kp14.py -> integ_<prefix><f>_<iy>.npz cheap (63 short jobs)

Each stage has its own content-addressed solve_id (variants/common/solstamp.py),
and the integ stage declares the G tables as upstream `inputs`. That decomposition
is the point: editing the cheap producer must NOT invalidate the expensive one,
while a change to G (or to any parameter) still invalidates integ. It is
utils/solfile_stamp.py's "mode 3: upstream moved, downstream did not".

The driver itself is deliberately NOT in either stage's source list. It only
orchestrates -- the numerics live entirely in the two producers plus
parameters_kp14.py -- and including it would recreate exactly the coupling this
split exists to remove.

Caching used to be a hand-written 5-key stamp {type_bv, gamma_v, kappa_y, bv_comp,
rho_ty}, which measurably MISSED delta, theta_eps, theta_u, sigma_eps, lambda_H,
mu_H and mu_L. The 2026-09-04 regime-label fix changed mu_H/mu_L and would have
silently reused stale tables.

    KP_VY_FORCE=1   rebuild both stages even on a cache hit
    KP_VY_ADOPT=1   record what is already on disk, without re-solving
"""
import os, sys, subprocess, time
from joblib import Parallel, delayed

prefix = sys.argv[1] if len(sys.argv) > 1 else "vy"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # variants/, for `common`

import parameters_kp14 as P
from common import solstamp

ntypes, NY = P.ntypes, P.NY
FORCE = bool(os.environ.get("KP_VY_FORCE"))
ADOPT = bool(os.environ.get("KP_VY_ADOPT"))

G_SOURCES = [os.path.join(HERE, f) for f in ("kp14_fd_vy.py", "parameters_kp14.py")]
I_SOURCES = [os.path.join(HERE, f) for f in ("integ_kp14.py", "parameters_kp14.py")]

G_ARTIFACTS = [os.path.join(HERE, f"G_{prefix}{f}.csv") for f in range(ntypes)]
I_ARTIFACTS = [os.path.join(HERE, f"integ_{prefix}{f}_{iy}.npz")
               for f in range(ntypes) for iy in range(NY)]

g_snap = solstamp.snapshot(P, G_SOURCES, model="kp_vy", stage="G",
                           extra={"prefix": prefix})
print(f"[solstamp] kp_vy prefix={prefix} stage=G solve_id={g_snap.solve_id}", flush=True)


def _integ_snapshot():
    """Built AFTER the G stage, so it hashes the G tables actually on disk."""
    return solstamp.snapshot(P, I_SOURCES, model="kp_vy", stage="integ",
                             extra={"prefix": prefix},
                             inputs=solstamp.artifact_digests(G_ARTIFACTS))


def _report_prior(snap, artifacts, label):
    prior = solstamp._find_by_artifacts([a for a in artifacts if os.path.exists(a)])
    if prior and prior["solve_id"] != snap.solve_id:
        print(f"[solstamp] {label} tables on disk belong to solve_id "
              f"{prior['solve_id']}; this run wants {snap.solve_id}. Differences:")
        for k, a, b in solstamp.diff_params(prior["params"], snap.params)[:10]:
            print(f"    {k}: {a!r} -> {b!r}")


# ------------------------------------------------------------------ adopt ----
if ADOPT:
    missing = [a for a in G_ARTIFACTS + I_ARTIFACTS if not os.path.exists(a)]
    if missing:
        sys.exit(f"cannot adopt: {len(missing)} artifact(s) missing, e.g. {missing[0]}")
    note = ("Adopted from artifacts already on disk (KP_VY_ADOPT=1); presence and "
            "hashes verified, provenance asserted by the operator.")
    gm = solstamp.record(g_snap, G_ARTIFACTS, tag=prefix, note=note)
    im = solstamp.record(_integ_snapshot(), I_ARTIFACTS, tag=prefix, note=note)
    print(f"[solstamp] adopted G     {g_snap.solve_id}  {gm['total_bytes']:,} B")
    print(f"[solstamp] adopted integ {im['solve_id']}  {im['total_bytes']:,} B")
    sys.exit(0)

t0 = time.time()

# ------------------------------------------------------------ stage: G -------
g_hit = solstamp.lookup(g_snap.solve_id)
if g_hit and not FORCE and not solstamp.artifact_problems(g_hit):
    print(f"[stage G] cached: solve_id {g_snap.solve_id} "
          f"({len(g_hit['artifacts'])} tables) -- skipping {ntypes} solve(s)")
else:
    if g_hit and not FORCE:
        print("[stage G] manifest exists but artifacts do not match; re-solving:")
        for p in solstamp.artifact_problems(g_hit)[:5]:
            print(f"  - {p}")
    else:
        _report_prior(g_snap, G_ARTIFACTS, "G")
    for f in range(ntypes):
        env = dict(os.environ, KP_VY_TYPE=str(f), KP_VY_GOUT=G_ARTIFACTS[f])
        # kp14_fd_vy.py prints its factorisation time and per-20-iteration error
        # with flush=True. That used to go to DEVNULL, which made a multi-hour
        # solve indistinguishable from a hung one. Stream it, prefixed.
        proc = subprocess.Popen(
            [sys.executable, "-W", "ignore", os.path.join(HERE, "kp14_fd_vy.py")],
            env=env, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(f"  [G type {f}] {line.rstrip()}", flush=True)
        if proc.wait() != 0:
            sys.exit(f"G solve failed for type {f}")
        print(f"[stage G] type {f} solved ({time.time()-t0:.0f}s)", flush=True)
    gm = solstamp.record(g_snap, G_ARTIFACTS, tag=prefix)
    print(f"[solstamp] recorded G {g_snap.solve_id} ({gm['total_bytes']:,} B)")

# -------------------------------------------------------- stage: integ -------
i_snap = _integ_snapshot()
print(f"[solstamp] kp_vy prefix={prefix} stage=integ solve_id={i_snap.solve_id}", flush=True)

i_hit = solstamp.lookup(i_snap.solve_id)
if i_hit and not FORCE and not solstamp.artifact_problems(i_hit):
    print(f"[stage integ] cached: solve_id {i_snap.solve_id} "
          f"({len(i_hit['artifacts'])} tables) -- nothing to do")
else:
    if i_hit and not FORCE:
        print("[stage integ] manifest exists but artifacts do not match; rebuilding:")
        for p in solstamp.artifact_problems(i_hit)[:5]:
            print(f"  - {p}")
    else:
        _report_prior(i_snap, I_ARTIFACTS, "integ")

    def one(f, iy):
        env = dict(os.environ, KP_VY_TYPE=str(f), KP_GAMY_YIDX=str(iy),
                   KP_GAMY_GIN=G_ARTIFACTS[f],
                   KP_GAMY_IOUT=os.path.join(HERE, f"integ_{prefix}{f}_{iy}.npz"))
        # stderr is kept: integ_kp14.py now RAISES when the CIR density loses mass,
        # and swallowing that would restore the silence this was built to remove.
        r = subprocess.run([sys.executable, "-W", "ignore",
                            os.path.join(HERE, "integ_kp14.py")],
                           env=env, cwd=HERE, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"integ failed for type {f}, y-node {iy}:\n{r.stderr[-2000:]}")
        return (f, iy)

    jobs = [(f, iy) for f in range(ntypes) for iy in range(NY)]
    done = Parallel(n_jobs=6, verbose=5)(delayed(one)(f, iy) for f, iy in jobs)
    im = solstamp.record(i_snap, I_ARTIFACTS, tag=prefix)
    print(f"[stage integ] built {len(done)} tables")
    print(f"[solstamp] recorded integ {i_snap.solve_id} ({im['total_bytes']:,} B)")

print(f"done in {time.time()-t0:.0f}s")
