"""Named reduced-form economies. Two families: the VALIDATION ladder, whose expected outcomes are
read off docs/RESULTS.md and must be reproduced before the screen is trusted anywhere else, and
the EXPLORATION set, which asks what no structural economy has been able to ask.

Each entry is a dict of overrides on rf_model.DEFAULTS. Keep the defaults ("base") fixed while a
ladder moves ONE knob, so a difference between two rows is attributable."""

BASE = {}   # rf_model.DEFAULTS: OU state, half-life 24 months, three centred types, footprint lev 1 + q 1 + r 0.5,
            # lam_s/sigma_f 0.5, lam_m/sigma_f 0.05. Chosen 2026-10-06 because lev+q reproduces KP14's signature
            # (room ~ room_s: the room is rank-nonlinear GIVEN the state) where r alone reproduces BGN/GS's

CONFIGS = {
    "base": BASE,

    # ---- validation ladder: each row has an expected sign from the record ----------------------
    # finding 11 / "what each model can express": no exposure heterogeneity -> no room, no gap
    "notype": {"J": 1},
    # the type is in the economy but leaves no footprint in any characteristic: room ~ 0 because
    # the ceilings are defined over characteristic bases; no estimator can find it either
    "invisible": {"r": 0.0, "lev": 0.0, "q": 0.0},
    # clause (iii) / GS21 ladder: a MONOTONE persistent footprint only; a rank on size reveals it,
    # so the linear side takes it and the fair gap is ~0
    "levels_only": {"r": 0.0, "q": 0.0},
    # curvature only, no state dependence: rank-nonlinear and time-invariant
    "curvature_only": {"r": 0.0},
    # state dependence only: the BGN/GS signature, room that is linear in (chars x state)
    "state_only": {"lev": 0.0, "q": 0.0, "r": 1.0},
    # finding 12's speed ladder, OU version (half-life in months); gap should rise as the state speeds up
    "hl120": {"halflife": 120.0},
    "hl48": {"halflife": 48.0},
    "hl12": {"halflife": 12.0},
    "hl6": {"halflife": 6.0},
    # finding 9's price ladder: lam_s / sigma_f = 0.25, 0.375, 0.5, 0.75
    "ps02": {"lam_s": 0.010},
    "ps03": {"lam_s": 0.015},
    "ps075": {"lam_s": 0.030},
    # finding 8: a market that spans the economy (GS21): large classical gap, zero fair gap
    "bigmkt": {"lam_m": 0.016, "lam_s": 0.008},
    # finding 16: density -- twenty types at N=500 against three
    "dense20": {"J": 20, "b_ladder": (-1.0, 1.0)},
    # finding 17: N doubled, nothing else
    "n1000": {"N": 1000},
    # finding 12 in its own clothes: BGN-style regime on the PRICE, unpriced switch, x0.2 / x1 / x4
    "reg_slow": {"state": "regime", "g_kind": "regime", "gmult": (0.2, 3.5), "p01": 0.25 / 12 * 0.2, "p10": 0.50 / 12 * 0.2, "r": 0.0, "lev": 0.5},
    "reg_mid": {"state": "regime", "g_kind": "regime", "gmult": (0.2, 3.5), "p01": 0.25 / 12, "p10": 0.50 / 12, "r": 0.0, "lev": 0.5},
    "reg_fast": {"state": "regime", "g_kind": "regime", "gmult": (0.2, 3.5), "p01": 0.25 / 12 * 4, "p10": 0.50 / 12 * 4, "r": 0.0, "lev": 0.5},

    # ---- exploration -----------------------------------------------------------------------
    # GS21-style countercyclical price on a fast state with a type ladder in levels
    "gs_like": {"g_kind": "clip", "g_slope": 0.4, "g_lo": 0.05, "g_hi": 1.5, "r": 0.0, "lev": 1.0, "lam_m": 0.012, "lam_s": 0.006},
    # state-dependent exposure magnitude on top of the footprint
    "u1": {"u1": 0.5},
    # the footprint hides at a non-central state value
    "s0": {"s0": 1.0},
    # curvature of value in exposure: q = 0 makes the type premium rank-linear given the state
    "q0": {"q": 0.0},
    "q05": {"q": 0.5},
    "q2": {"q": 2.0},
    # state dependence of the footprint
    "r0": {"r": 0.0},
    "r1": {"r": 1.0},
    "r2": {"r": 2.0},
    # idiosyncratic noise halves / doubles (the estimation-variance side of the loss identity)
    "eps05": {"sig_eps": 0.05},
    "eps20": {"sig_eps": 0.20},
}

# one-seed grid over the footprint knobs, for the population map (room, room_s, theta) and a first
# look at the gap in every cell
GRID = {}
for lev in (0.5, 1.0, 2.0):
    for q in (0.0, 0.5, 1.0, 2.0):
        for r in (0.0, 0.3, 0.6, 1.0):
            GRID[f"g_l{lev:g}_q{q:g}_r{r:g}"] = {"lev": lev, "q": q, "r": r}
CONFIGS.update(GRID)

# speed x price x curvature cross, three seeds: the interaction the structural runs could never afford
CROSS = {}
for hl in (6.0, 12.0, 24.0, 48.0, 120.0):
    for ls in (0.010, 0.015, 0.020, 0.030):
        for q in (0.0, 1.0, 2.0):
            CROSS[f"x_h{hl:g}_p{int(ls*1000)}_q{q:g}"] = {"halflife": hl, "lam_s": ls, "q": q}
CONFIGS.update(CROSS)

# round 2 (2026-10-06, late): the questions the first ladders raise
ROUND2 = {
    # can a BGN-style economy carry the KP14 shape? regime price x curvature footprint, no state dependence
    **{f"rq_{spd}_q{q:g}": {"state": "regime", "g_kind": "regime", "gmult": (0.2, 3.5), "p01": 0.25 / 12 * m, "p10": 0.50 / 12 * m, "r": 0.0, "q": q}
       for spd, m in (("slow", 0.2), ("mid", 1.0), ("fast", 4.0)) for q in (0.0, 1.0, 2.0)},
    # finding 8 as a ladder: the market's share of SR_max via lam_m
    "mkt1": {"lam_m": 0.001}, "mkt4": {"lam_m": 0.004}, "mkt8": {"lam_m": 0.008}, "mkt16": {"lam_m": 0.016},
    # density as a ladder: firms per distinct premium
    "J2": {"J": 2, "b_ladder": (-1.0, 1.0)}, "J5": {"J": 5, "b_ladder": (-1.0, 1.0)}, "J10": {"J": 10, "b_ladder": (-1.0, 1.0)},
    # KP14's own asymmetric, all-positive ladder (0.02 : 0.07 : 0.14), so the market loads on the state
    "kpladder": {"b_ladder": (0.2, 0.7, 1.4)},
    # heterogeneous idiosyncratic vol
    "epsdisp": {"sig_eps_disp": 0.4},
}
CONFIGS.update(ROUND2)

# ---- the K ladder (2026-10-08): extra priced factors with nonlinear characteristic loadings, on
# top of the no-type economy, so the ONLY premium outside the market is the sum of K nonlinear
# components. Two normalisations: a fixed per-factor price (SR_max grows ~ sqrt(K)) and a fixed total
# price (lam_f ~ 1/sqrt(K), SR_max held). Shapes: tanh (KP's bounded share), sq (pure curvature), lin.
KFAC = {}
for shape in ("tanh", "sat", "sq", "lin"):
    for nf in (1, 2, 4, 8, 16):
        KFAC[f"kf_{shape}_{nf}"] = {"J": 1, "nf": nf, "fac_shape": shape, "lam_f": 0.012}
        KFAC[f"kfh_{shape}_{nf}"] = {"J": 1, "nf": nf, "fac_shape": shape, "lam_f": 0.012 * 2.0 / nf ** 0.5}
    KFAC[f"kf_{shape}_0"] = {"J": 1, "nf": 0}
# the same factors on top of the base economy's type machinery (KP footprint + types)
for nf in (1, 2, 4, 8):
    KFAC[f"kfb_sat_{nf}"] = {"nf": nf, "fac_shape": "sat", "lam_f": 0.012}
CONFIGS.update(KFAC)
# the same fixed-total-price ladder with directions ORTHOGONAL TO SIZE (2026-10-17): removes the value-weighting artefact
KFO = {}
for shape in ("sq", "lin", "tanh"):
    for nf in (1, 4, 16):
        KFO[f"kfo_{shape}_{nf}"] = {"J": 1, "nf": nf, "fac_shape": shape, "lam_f": 0.012 * 2.0 / nf ** 0.5, "fac_orth_size": True}
CONFIGS.update(KFO)

# ---- the LATENT ladder (2026-10-10): factors loaded by products of latent firm states that the characteristics
# reveal only up to observation noise (rf_model.py nlat / obs_noise / nf_lat). L latents in {3, 5}, K product factors
# from 1 to the number of pairs, noise in {0, 0.5, 1}; lf_ = fixed per-factor price, lfh_ = fixed total price.
LFAC = {}
for nlat in (3, 5):
    npairs = nlat * (nlat - 1) // 2
    for nf in sorted({1, 2, 3, 5, 10} & set(range(1, npairs + 1))):
        for on in (0.0, 0.5, 1.0):
            LFAC[f"lf_L{nlat}_K{nf}_n{on:g}"] = {"J": 1, "nlat": nlat, "obs_noise": on, "nf_lat": nf, "lam_lat": 0.012}
            LFAC[f"lfh_L{nlat}_K{nf}_n{on:g}"] = {"J": 1, "nlat": nlat, "obs_noise": on, "nf_lat": nf, "lam_lat": 0.012 * 2.0 / nf ** 0.5}
CONFIGS.update(LFAC)

# ---- the SIGN ladder (2026-10-26): one latent-product factor (lfh_L5_K1's economy, loading standardised each month)
# with the two factors of the product shifted by (m_a, m_b): (0, 0) is the mean-zero product, (0, m) a sign-switching
# type times a one-signed leverage of CV 1/m, (m, m) two one-signed factors as in KP14 and BGN (m = 2: CV 0.5).
# lrt_ is the debt-service-coverage ratio, one-signed and convex in a single index. Noise 0 and 1 as the latent ladder.
LSH = {}
for ma, mb in ((0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (0, 3), (3, 3)):
    for on in (0.0, 1.0):
        LSH[f"lsh_a{ma}_b{mb}_n{on:g}"] = {"J": 1, "nlat": 5, "obs_noise": on, "nf_lat": 1, "lam_lat": 0.012 * 2.0,
                                          "lat_shift": (float(ma), float(mb), 0.0, 0.0, 0.0), "lat_std": True}
for on in (0.0, 1.0):
    LSH[f"lrt_n{on:g}"] = {"J": 1, "nlat": 5, "obs_noise": on, "nf_lat": 1, "lam_lat": 0.012 * 2.0, "lat_form": "ratio", "lat_std": True}
CONFIGS.update(LSH)

# ---- the CURVATURE-in-a-latent ladder (2026-10-30, 'a non-monotonic version of the reduced form'): one factor loading on the
# square of ONE partly hidden latent, (s + m)^2 - 1, standardised each month; m = 0 is mean-zero curvature (the structural
# gvyzz economies: exposure quadratic in a firm's own productivity), m = 1, 2 add a linear part. Noise 0 and 1 as the sign ladder.
LSQ = {}
for ma in (0, 1, 2):
    for on in (0.0, 1.0):
        LSQ[f"lsq_a{ma}_n{on:g}"] = {"J": 1, "nlat": 5, "obs_noise": on, "nf_lat": 1, "lam_lat": 0.012 * 2.0, "lat_form": "square",
                                     "lat_shift": (float(ma), 0.0, 0.0, 0.0, 0.0), "lat_std": True}
CONFIGS.update(LSQ)
# lsqb_: the same curvature in the BOOK-TO-MARKET latent (index 1). Latent 0 is size and market value is exp(size), so the
# value-weighted market inside the Fama-French set holds size^2 (lsq_a0_n0: FF 0.565 of SR_max 0.60, fair gap -0.008); bm's square
# has no such carrier. The shift is applied to the squared latent.
LSQB = {}
for ma in (0, 1, 2):
    for on in (0.0, 1.0):
        LSQB[f"lsqb_a{ma}_n{on:g}"] = {"J": 1, "nlat": 5, "obs_noise": on, "nf_lat": 1, "lam_lat": 0.012 * 2.0, "lat_form": "square",
                                       "lat_sq_idx": 1, "lat_shift": (0.0, float(ma), 0.0, 0.0, 0.0), "lat_std": True}
CONFIGS.update(LSQB)
# lsqr_: the same curvature in the PROFITABILITY latent (index 3, roe): the reduced-form twin of GS21's exposure quadratic in the
# firm's own productivity (gvyzzh10), beside the size square (KP14's and BGN's hump across size) and the book-to-market square.
LSQR = {}
for on in (0.0, 1.0):
    LSQR[f"lsqr_a0_n{on:g}"] = {"J": 1, "nlat": 5, "obs_noise": on, "nf_lat": 1, "lam_lat": 0.012 * 2.0, "lat_form": "square",
                                "lat_sq_idx": 3, "lat_shift": (0.0, 0.0, 0.0, 0.0, 0.0), "lat_std": True}
CONFIGS.update(LSQR)

LADDERS = {
    "lfac": sorted(LFAC),
    "signs": sorted(LSH),
    "lsq": sorted(LSQ),
    "lsqb": sorted(LSQB),
    "lsqr": sorted(LSQR),
    "kfac": sorted(KFAC),
    "kfo": sorted(KFO),
    "speed": ["hl120", "hl48", "base", "hl12", "hl6"],
    "price": ["ps02", "ps03", "base", "ps075"],
    "regime": ["reg_slow", "reg_mid", "reg_fast"],
    "visibility": ["invisible", "levels_only", "curvature_only", "state_only", "base", "s0", "u1"],
    "curvature": ["q0", "q05", "base", "q2"],
    "state": ["r0", "base", "r1", "r2"],
    "structure": ["notype", "base", "bigmkt", "dense20", "n1000"],
    "noise": ["eps05", "base", "eps20"],
    "grid": sorted(GRID),
    "cross": sorted(CROSS),
    "round2": sorted(ROUND2),
    "market": ["mkt1", "base", "mkt4", "mkt8", "mkt16"],
    "density": ["J2", "base", "J5", "J10", "dense20"],
    "regime_q": sorted(k for k in ROUND2 if k.startswith("rq_")),
    # the reduced grid against the protocol grid (--full): does rff 3600 / two draws change a conclusion?
    "fullcheck": ["base", "q0", "q2", "state_only", "curvature_only", "hl6", "ps075"],
    # the same seven beside their protocol-grid twins (rf_screen.py --full writes <name>_full)
    "fullcheck_pairs": [x for c in ("base", "q0", "q2", "state_only", "curvature_only", "hl6", "ps075") for x in (c, c + "_full")],
}
LADDERS["validation"] = sorted({c for k in ("speed", "price", "regime", "visibility", "curvature", "state", "structure", "noise") for c in LADDERS[k]})
