"""Reduced-form conditional factor economy: the panel generator behind variants/reduced/.

WHAT THIS IS. A simulated cross-section whose TRUE conditional moments are known in closed form,
built so that the only things separating two economies are the handful of knobs that the record in
docs/RESULTS.md says matter for a complexity gap. No equilibrium solve, no value function: a panel
costs seconds, the true moments cost one low-rank matrix per month, and the real estimators
(variants/common/dkkm_functions.py, fama_functions.py) are run on it unchanged.

THE MODEL (monthly). docs/REDUCED_FORM.md states it with the mapping to the three structural models.

  aggregate state   s_t          OU with persistence rho (unit stationary variance), or a two-state
                                 Markov chain with switch probabilities p01, p10
  priced factors    f_{t+1}      K = 2 Gaussian shocks, sd sigma_f: a market/cash-flow shock and the
                                 state's own innovation (so the state's risk is a priced factor, as
                                 in KP14's y and GS21's x)
  prices of risk    lam_t        (lam_m, lam_s) x g(s_t); g is 1, a clipped linear slope in the
                                 state (GS21 gamma(x)), or a regime multiplier (BGN gmult)
  firm type         tau_i        J types in shares pi, exposure ladder b_tau to the state factor
  characteristics   X_it (5)     persistent latent AR(1) draws z_it, PLUS the type's footprint:
                                   size += r * b~_tau * (s~_t - s0) + lev * b~_tau + q * b~_tau^2
                                   bm   -= the same
                                 r is state-dependent visibility (KP14's compensation term: at
                                 s = s0 the type is invisible in value, and the ordering flips
                                 sign across the state); lev is level visibility (GS21's ladder,
                                 which a rank on size reveals at every date); q is the
                                 CURVATURE of value in exposure (a claim on exp(b y) has log
                                 value convex in b), so near s0 the two extreme types look
                                 alike in value while earning opposite premia: a U-shape no
                                 rank-linear map carries, even one given the state
  loadings          beta_it      [1 + a . X~_it ,  b_tau u(s_t)]      u(s) = 1 + u1 s~
  premium           mu_it        beta_it' lam_t      -- exact factor pricing, mu = -cov(R, m)
  returns           xret_it      mu_it + beta_it' f_{t+1} + eps_it,   eps ~ N(0, sig_eps^2)
  covariance        Sigma_t      B_t Omega B_t' + diag(sig_eps^2)

So the cross-sectional premium has a LINEAR, always-visible part (a . X on the market factor: the
Fama-French world) and a TYPE part whose map from characteristics to premium is
    premium  = lam_s g(s) u(s) b_tau,        footprint = r b~_tau (s~ - s0) + lev b~_tau,
i.e. the regression coefficient of premium on the footprint is lam_s g(s) u(s) / (r (s~ - s0)) up to
the level term: it flips sign across the state and blows up at s0. A rolling linear fit holds one
coefficient per window; the faster the state moves relative to the window, the less of that map it
can carry. That is the mechanism findings 9, 12 and 16 point at, isolated.

Every knob the record names has a parameter here:
  exposure heterogeneity    J, b_ladder, b_scale, shares
  price of the state's risk lam_s (in return units; lam_s / sigma_f is its Sharpe)
  state speed               rho (OU half-life) or p01, p10 (regime)
  price-of-risk shape       g_kind, g_slope, g_lo, g_hi, gmult
  visibility                r (state-dependent), lev (level), s0 (where the type hides), q (curvature)
  market share of SR_max    lam_m against lam_s; a (the linear tilt)
  density                   J against N
"""
import numpy as np
import pandas as pd

CHARS = ["size", "bm", "agr", "roe", "mom"]
# monthly persistence of the five latent characteristic processes (unit stationary variance)
CHAR_RHO = {"size": 0.99, "bm": 0.98, "agr": 0.90, "roe": 0.95, "mom": 0.70}

DEFAULTS = dict(
    N=500, T=500, burnin=100,
    # state
    state="ou", halflife=24.0, p01=0.25 / 12, p10=0.50 / 12,
    # factors and prices
    sigma_f=0.04, lam_m=0.002, lam_s=0.020,
    g_kind="const", g_slope=0.0, g_lo=0.05, g_hi=1.5, gmult=(1.0, 1.0),
    # types
    J=3, b_ladder=(-1.0, 0.0, 1.0), b_scale=0.3, shares=None,
    u1=0.0,
    # characteristics: linear tilt on the market loading, and the type's footprint
    a=(0.0, 0.3, 0.0, 0.3, 0.3), r=0.5, lev=1.0, s0=0.0, q=1.0,
    # idiosyncratic
    sig_eps=0.10, sig_eps_disp=0.0,
    mve_base=7.0,
    # ---- extra priced factors with NONLINEAR characteristic loadings (2026-10-08, "K ladder") ----
    # The factor model the structural models imply: K priced aggregate shocks, each loaded through a
    # smooth, bounded, nonlinear function of a characteristic index -- KP14's growth-option share
    # (convex in value), BGN's duration and project beta, GS21's operating leverage. Factor k loads as
    #     beta_ik = fac_amp * h(w_k . X~_i),  h = tanh (bounded, KP's share), u^2 (curvature) or u (linear),
    # w_k a fixed unit direction over the 5 characteristics (drawn once from fac_dirseed, the same at
    # every seed), DEMEANED each month when fac_center so the EW market carries none of it, priced at
    # lam_f * g(s) each, sd sigma_f. The premium sum_k lam_f beta_ik is then rank-nonlinear GIVEN the
    # state, room_s = room, and the question is whether the estimators find it as K grows.
    nf=0, fac_shape="tanh", fac_amp=1.0, lam_f=0.010, fac_center=True, fac_dirseed=20261008,
    # 2026-10-17: fac_orth_size drops the size component of every direction (then renormalises), so no factor's
    # curvature loads on size^2 -- market value is exp(size), and value-weighted Fama-French legs pick up size^2 otherwise
    fac_orth_size=False,
    # ---- LATENT-STATE factors (2026-10-10): the structural lesson in reduced form. The firm's premium in KP14 and
    # BGN is a PRODUCT of a type and a within-firm state, each only partly revealed by the characteristics; GS21's is a
    # smooth function of states the characteristics reveal directly, and has no gap. Here the first nlat of the five
    # latent AR(1) processes are the firm's STATES; the observed characteristics are those states plus observation
    # noise of sd obs_noise (0 = fully revealed), the other characteristics stay pure noise; factor k (k < nf_lat)
    # loads as the demeaned PRODUCT of a pair of latents (pairs enumerated in a fixed order, cycling), priced at
    # lam_lat * g(s) each. The premium is a quadratic form in the latents: nonlinear in the ranks, hidden to the
    # degree the noise hides the states, and richer as nf_lat grows.
    nlat=0, obs_noise=0.0, nf_lat=0, lam_lat=0.012, lat_center=True,
    # ---- the SIGN of the product's factors (2026-10-26): lat_shift adds a constant to each latent before the product,
    # so factor k loads as (s_a + m_a)(s_b + m_b). m = 0 is the mean-zero product (GS21's imposed product); m > 0 makes
    # the factor one-signed with coefficient of variation 1/m (KP14's type ladder has CV 0.64, its growth-option share
    # ~0.5), and the product then has an additive part m_b s_a + m_a s_b that a linear portfolio holds. lat_form "ratio"
    # is debt service over profit: coverage c = exp(lat_ratio_mean + (s_a - s_b)/sqrt2), loading 1/max(c - 1, 1/cap),
    # one-signed and convex in ONE index. lat_std standardises each factor's loading cross-sectionally every month, so
    # the premium's dispersion is the same in every cell and only its additive share moves.
    lat_shift=0.0, lat_form="product", lat_std=False, lat_ratio_mean=1.0, lat_ratio_cap=5.0,
)


def resolve(params):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if v is not None})
    J = int(p["J"])
    lad = np.asarray(p["b_ladder"], float)
    if len(lad) != J:
        lad = np.linspace(lad.min(), lad.max(), J) if J > 1 else np.array([lad.mean()])
    p["b_ladder"] = tuple(float(x) for x in lad)
    sh = p["shares"]
    sh = np.full(J, 1.0 / J) if sh is None else np.asarray(sh, float) / np.sum(sh)
    p["shares"] = tuple(float(x) for x in sh)
    p["a"] = tuple(float(x) for x in p["a"])
    return p


class Economy:
    """One draw of the reduced-form economy. `seed` steers everything; the aggregate state path
    is drawn from a separate stream offset by the seed, as run_oracle.py does, so that two
    configurations at the same seed share both the aggregate path and the firm-level draws."""

    def __init__(self, params=None, seed=0):
        self.p = p = resolve(params)
        self.seed = int(seed)
        N, T, burn = p["N"], p["T"], p["burnin"]
        self.N, self.T = N, T
        rng_agg = np.random.default_rng(555 + self.seed)        # aggregate state and factors
        rng_firm = np.random.default_rng(1000 + self.seed)      # types, characteristics, eps
        Ttot = burn + T + 1

        # ---- aggregate state -------------------------------------------------------------
        if p["state"] == "ou":
            rho = 0.5 ** (1.0 / float(p["halflife"]))
            eta = rng_agg.standard_normal(Ttot)
            s = np.zeros(Ttot)
            for t in range(1, Ttot):
                s[t] = rho * s[t - 1] + np.sqrt(1 - rho ** 2) * eta[t]
            s_std = s                                            # unit stationary variance
            innov = eta                                          # the state's innovation
            self.rho = rho
        elif p["state"] == "regime":
            p01, p10 = float(p["p01"]), float(p["p10"])
            u = rng_agg.random(Ttot)
            s = np.zeros(Ttot, dtype=int)
            for t in range(1, Ttot):
                s[t] = (1 if u[t] < p01 else 0) if s[t - 1] == 0 else (0 if u[t] < p10 else 1)
            pi1 = p01 / (p01 + p10)
            s_std = (s - pi1) / np.sqrt(pi1 * (1 - pi1))
            innov = rng_agg.standard_normal(Ttot)               # the switch is UNPRICED (BGN)
            self.rho = None
        else:
            raise ValueError(p["state"])
        self.s_raw, self.s_std = s, s_std

        # ---- prices of risk ----------------------------------------------------------------
        if p["g_kind"] == "const":
            g = np.ones(Ttot)
        elif p["g_kind"] == "clip":
            g = np.clip(1.0 - float(p["g_slope"]) * s_std, float(p["g_lo"]), float(p["g_hi"]))
        elif p["g_kind"] == "regime":
            if p["state"] != "regime":
                raise ValueError("g_kind='regime' needs state='regime'")
            gm = np.asarray(p["gmult"], float)
            g = gm[s]
        else:
            raise ValueError(p["g_kind"])
        self.g = g
        nf = int(p["nf"]); nfl = int(p["nf_lat"]) if int(p["nlat"]) > 0 else 0
        self.lam = np.column_stack([float(p["lam_m"]) * g, float(p["lam_s"]) * g]
                                   + [float(p["lam_f"]) * g] * nf + [float(p["lam_lat"]) * g] * nfl)   # Ttot x K
        self.K = 2 + nf + nfl
        sf = float(p["sigma_f"])
        self.Omega = np.eye(self.K) * sf ** 2
        f_m = rng_agg.standard_normal(Ttot) * sf
        f_x = rng_agg.standard_normal((Ttot, nf + nfl)) * sf if (nf + nfl) else np.zeros((Ttot, 0))
        self.f = np.column_stack([f_m, innov * sf, f_x])                              # f_{t+1} stored at t+1
        self.nf = nf; self.nfl = nfl; self.nlat = int(p["nlat"])
        if nfl:
            pairs = [(a, b) for a in range(self.nlat) for b in range(a + 1, self.nlat)]
            assert pairs, "nf_lat > 0 needs nlat >= 2"
            self.lat_pairs = [pairs[k % len(pairs)] for k in range(nfl)]
        if nf:
            W = np.random.default_rng(int(p["fac_dirseed"])).standard_normal((nf, 5))
            if p.get("fac_orth_size"):
                W[:, 0] = 0.0                                       # size is characteristic 0
            self.fac_dirs = W / np.linalg.norm(W, axis=1, keepdims=True)

        # ---- firms ---------------------------------------------------------------------------
        J = int(p["J"])
        self.types = rng_firm.choice(J, size=N, p=np.asarray(p["shares"]))
        b_lad = np.asarray(p["b_ladder"]) * float(p["b_scale"])
        self.b = b_lad[self.types]
        b_mean = float(np.dot(np.asarray(p["shares"]), b_lad))
        b_sd = float(np.sqrt(np.dot(np.asarray(p["shares"]), (b_lad - b_mean) ** 2)))
        self.b_tilde = (self.b - b_mean) / (b_sd if b_sd > 0 else 1.0)
        sig = float(p["sig_eps"]) * np.exp(float(p["sig_eps_disp"]) * rng_firm.standard_normal(N))
        self.sig_eps = sig

        # latent characteristics: AR(1) in time, unit stationary variance, independent across firms
        z = np.zeros((Ttot, N, 5))
        z[0] = rng_firm.standard_normal((N, 5))
        for j, c in enumerate(CHARS):
            rho_c = CHAR_RHO[c]
            e = rng_firm.standard_normal((Ttot, N)) * np.sqrt(1 - rho_c ** 2)
            for t in range(1, Ttot):
                z[t, :, j] = rho_c * z[t - 1, :, j] + e[t]
        self.z = z
        # observation noise on the characteristics that carry the latent states (independent AR(1) draws, same
        # persistence as the characteristic, scaled by obs_noise), drawn AFTER everything else so nlat = 0 economies
        # and the pre-2026-10-10 configurations keep their exact streams
        self.obs = None
        if int(p["nlat"]) > 0 and float(p["obs_noise"]) > 0:
            on = np.zeros((Ttot, N, int(p["nlat"])))
            on[0] = rng_firm.standard_normal((N, int(p["nlat"])))
            for j in range(int(p["nlat"])):
                rho_c = CHAR_RHO[CHARS[j]]
                e = rng_firm.standard_normal((Ttot, N)) * np.sqrt(1 - rho_c ** 2)
                for t in range(1, Ttot):
                    on[t, :, j] = rho_c * on[t - 1, :, j] + e[t]
            self.obs = on * float(p["obs_noise"])
        self.eps = rng_firm.standard_normal((Ttot, N)) * sig[None, :]
        self.burn = burn

    # ---- per-month objects ---------------------------------------------------------------------
    def chars(self, t):
        """Observed characteristics at (absolute) month t: latent draws plus the type footprint."""
        p = self.p
        X = self.z[t].copy()
        if self.obs is not None:
            X[:, :self.nlat] += self.obs[t]
        foot = (float(p["r"]) * self.b_tilde * (self.s_std[t] - float(p["s0"])) + float(p["lev"]) * self.b_tilde
                + float(p["q"]) * self.b_tilde ** 2)
        X[:, 0] += foot          # size
        X[:, 1] -= foot          # book-to-market moves against value
        return X

    def loadings(self, t, X):
        p = self.p
        a = np.asarray(p["a"])
        Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        u = 1.0 + float(p["u1"]) * self.s_std[t]
        cols = [1.0 + Xs @ a, self.b * u]
        if self.nf:
            idx = Xs @ self.fac_dirs.T                                  # N x nf characteristic indices
            shape = p["fac_shape"]
            # tanh: near-linear in the ranks (R^2 ~0.9); sat: tanh(2.5 u), a bounded share with most firms
            # at the extremes (KP14's growth-option share); sq: pure curvature; lin: linear
            h = (np.tanh(idx) if shape == "tanh" else np.tanh(2.5 * idx) if shape == "sat"
                 else idx ** 2 if shape == "sq" else idx)
            if p["fac_center"]:
                h = h - h.mean(0)
            cols.append(float(p["fac_amp"]) * h)
        if self.nfl:
            s_lat = self.z[t][:, :self.nlat]                            # the true states, noise-free
            sh = np.asarray(p["lat_shift"], float).ravel()
            sh = np.full(self.nlat, float(sh[0])) if sh.size == 1 else np.concatenate([sh, np.zeros(self.nlat - sh.size)])
            if p["lat_form"] == "product":
                L = np.column_stack([(s_lat[:, a] + sh[a]) * (s_lat[:, b] + sh[b]) for a, b in self.lat_pairs])
            elif p["lat_form"] == "ratio":
                m, cap = float(p["lat_ratio_mean"]), float(p["lat_ratio_cap"])
                L = np.column_stack([1.0 / np.maximum(np.exp(m + (s_lat[:, a] - s_lat[:, b]) / np.sqrt(2.0)) - 1.0, 1.0 / cap)
                                     for a, b in self.lat_pairs])
            else:
                raise ValueError(p["lat_form"])
            if p["lat_std"]:
                L = L / np.where(L.std(0) > 0, L.std(0), 1.0)
            if p["lat_center"]:
                L = L - L.mean(0)
            cols.append(L)
        return np.column_stack(cols)                                    # N x K

    def moments(self, t, X=None):
        """True conditional mean and covariance of xret at month t (dense N x N)."""
        X = self.chars(t) if X is None else X
        B = self.loadings(t, X)
        mu = B @ self.lam[t]
        Sigma = B @ self.Omega @ B.T
        Sigma[np.diag_indices_from(Sigma)] += self.sig_eps ** 2
        return mu, Sigma, B

    def sr_max(self, t):
        """sqrt(mu' Sigma^-1 mu) by Woodbury: O(N K^2)."""
        mu, _, B = self.moments(t)
        d = 1.0 / self.sig_eps ** 2
        BtD = B.T * d
        M = np.linalg.inv(self.Omega) + BtD @ B
        v = BtD @ mu
        return float(np.sqrt(max(mu @ (d * mu) - v @ np.linalg.solve(M, v), 0.0)))

    # ---- the panel -------------------------------------------------------------------------------
    def panel(self):
        """Long panel in run_oracle.py's layout: month (0..T-1 after burn-in), firmid, mve, xret,
        the five characteristics and rf_stand (the standardised aggregate state)."""
        p = self.p
        rows = []
        for m in range(self.T):
            t = self.burn + m
            X = self.chars(t)
            B = self.loadings(t, X)
            mu = B @ self.lam[t]
            xret = mu + B @ self.f[t + 1] + self.eps[t + 1]
            df = pd.DataFrame(X, columns=CHARS)
            df.insert(0, "firmid", np.arange(self.N))
            df.insert(0, "month", m)
            df["mve"] = np.exp(float(p["mve_base"]) + X[:, 0])
            df["xret"] = xret
            df["rf_stand"] = float(self.s_std[t])
            df["type"] = self.types
            rows.append(df)
        return pd.concat(rows, ignore_index=True)

    def month_moments(self, m):
        return self.moments(self.burn + m)

    def describe(self):
        p = self.p
        sf = float(p["sigma_f"])
        return (f"state={p['state']}" + (f" halflife={p['halflife']}" if p['state'] == 'ou' else f" p01={p['p01']:.4f} p10={p['p10']:.4f}")
                + f" g={p['g_kind']} lam_m/sf={p['lam_m']/sf:.3f} lam_s/sf={p['lam_s']/sf:.3f}"
                + f" J={p['J']} b={tuple(round(x*p['b_scale'],3) for x in p['b_ladder'])} r={p['r']} q={p['q']} lev={p['lev']} s0={p['s0']}"
                + f" a={p['a']} sig_eps={p['sig_eps']} N={p['N']} T={p['T']}"
                + (f" nf={p['nf']} fac={p['fac_shape']} lam_f/sf={p['lam_f']/sf:.3f} amp={p['fac_amp']}" if int(p['nf']) else "")
                + (f" nlat={p['nlat']} obs_noise={p['obs_noise']} nf_lat={p['nf_lat']} lam_lat/sf={p['lam_lat']/sf:.3f}"
                   f" lat_form={p['lat_form']} lat_shift={p['lat_shift']} lat_std={p['lat_std']}" if int(p['nlat']) else ""))
