export const meta = {
  name: 'voc-gap-diagnosis',
  description: 'Diagnose why DKKM does not beat FF/FM in BGN/KP14/GS21 simulations and propose adversarially-vetted minimal model changes',
  phases: [
    { title: 'Diagnose', detail: 'per-model structural analysis + pipeline/metric audit + VoC theory' },
    { title: 'Propose', detail: 'minimal model alterations per model + cross-cutting' },
    { title: 'Critique', detail: 'adversarial lenses: economic structure, mechanism skepticism, referee' },
    { title: 'Synthesize', detail: 'rank and integrate' },
  ],
}

const CTX = `
## Project context

Paper: Back, Ober & Pruitt, "The Virtue of Complexity in Simple Economic Models".
Two repos (you have read access; READ THE ACTUAL FILES, do not guess):

SIM REPO: /Users/sjpruitt/GitHub/bop-run-upload
  config.py                              all economic + estimation parameters
  main.py                                7-step pipeline driver
  utils_bgn/panel_functions_bgn.py       BGN (Berk-Green-Naik 1999) simulation
  utils_bgn/sdf_compute_bgn.py           BGN true conditional moments + true MVE/HJ portfolio
  utils_bgn/vasicek.py, loadings_compute_bgn.py
  utils_kp14/panel_functions_kp14.py     KP14 (Kogan-Papanikolaou 2014) simulation
  utils_kp14/sdf_compute_kp14.py         KP14 true conditional moments
  utils_kp14/kp14_fd.py, integ_kp14.py, loadings_compute_kp14.py
  utils_gs21/panel_functions_gs21.py     GS21 (Gomes-Schmid 2021) simulation
  utils_gs21/sdf_compute_gs21.py         GS21 true conditional moments
  utils_factors/dkkm_functions.py        RFF construction + ridge MVE (the "DKKM" estimator)
  utils_factors/fama_functions.py        FF 2x3 sorts + Fama-MacBeth portfolios + OLS MVE
  utils_factors/factor_utils.py          prepare_panel
  utils/generate_panel.py, generate_dkkm_factors.py, generate_fama_factors.py
  utils/estimate_sdf_dkkm.py, estimate_sdf_fama.py, calculate_moments.py, evaluate_sdfs.py
  utils/generate_25_portfolios.py

ANALYZE REPO: /Users/sjpruitt/GitHub/bop-analyze-remote
  create_slides_tables.py      Sharpe and HJD definitions (lines ~30-100)
  compute_pricing_errors.py    alpha' Sigma^-1 alpha and GRS on 25 size/BM portfolios
  analyze.py                   same Sharpe/HJD aggregation
  MLP Analyze/Full run/results/correlations.csv   corr(method SDF weight, TRUE SDF weight)
  MLP Analyze/README.md, common.py, Full run/run.py

## The problem to solve

In real US stock data, the DKKM estimator (random Fourier features of characteristics
+ ridge, "virtue of complexity") delivers a MUCH higher out-of-sample Sharpe ratio and
MUCH lower pricing error than Fama-French sorted factors (FF) and Fama-MacBeth
cross-sectional-regression factors (FM). In these three simulated structural models it
does NOT: DKKM is roughly tied with FF/FM. Something the models lack is what makes
complexity pay in real data. We need MINIMAL, ECONOMICALLY DEFENSIBLE alterations
(possibly different per model) that create a large DKKM advantage in Sharpe and HJD.

## Established facts (verified by reading the code; treat as facts)

METRICS
- Sharpe: for each eval month t, mean_t = w_t . rp_t and stdev_t = sqrt(w_t' Sigma_t w_t),
  where rp_t and Sigma_t are the model's TRUE conditional risk premia and TRUE conditional
  covariance of individual stock returns (from *_moments.pkl). Sharpe = mean over the 360
  eval months of (mean_t/stdev_t). So it is a POPULATION CONDITIONAL Sharpe of the
  estimated weights: no evaluation sampling noise, only estimation noise in w_t.
  (utils/evaluate_sdfs.py lines 182-209; create_slides_tables.py lines 34-42)
- HJD: sqrt( mean_t (xret_method,t - sdf_ret_t)^2 ), where xret_method,t = w_t . realized
  excess returns, and sdf_ret_t is the realized excess return on the model's TRUE
  unit-cost minimum-second-moment (Hansen-Jagannathan) portfolio:
  port = ER^{-1} 1 over {riskfree, N stocks}; port /= port.sum(); sdf_ret = -(port[1:] .
  (gross return - gross riskfree)).  (create_slides_tables.py lines 44-56;
  sdf_compute_*.py near the end of each sdf_loop)
- compute_pricing_errors.py additionally computes alpha' Sigma^-1 alpha and GRS with each
  method's MVE return as a SINGLE factor against 25 size/BM portfolios.

DIAGNOSTIC DATA ALREADY IN HAND (MLP Analyze/Full run/results/correlations.csv):
  corr_actual_vs_truth = mean cross-sectional corr(method stock weight, true SDF stock weight)
  corr_fitted_vs_truth = same after fitting a 5x32 MLP of the weight on rank-standardized chars
    model  method  corr_actual   corr_fitted
    bgn    true    1.000         0.685     <- truth is only 68% recoverable from chars
    bgn    fm      0.0025        0.091
    bgn    ff      0.00001       0.044
    bgn    dkkm    0.0010        0.035     <- DKKM WORST of the three
    gs21   true    1.000         0.983     <- truth is ~fully a function of chars
    gs21   fm      0.0058        0.093
    gs21   ff      0.0036        0.086
    gs21   dkkm    0.0045        0.069     <- DKKM WORST
    kp14   true    1.000         0.568
    kp14   fm     -0.0080       -0.148     <- all methods NEGATIVELY correlated with truth
    kp14   ff     -0.0053       -0.117
    kp14   dkkm   -0.0069       -0.122
  So: no method recovers the true SDF weights at all, and DKKM is not better than FF/FM.

ESTIMATOR SETUP
- N=1000 firms, T=720 post-burnin months, burnin=200. Rolling window = 360 months.
  Evaluation = last 360 months.
- chars: bgn/kp14 = [size, bm, agr, roe, mom]; gs21 = those + mkt_lev. BGN's RFF also gets
  rf_stand (a date-level, cross-sectionally CONSTANT variable) appended as a 6th input.
- DKKM features: [6, 36, 360, 3600]; NMAT=5 independent W draws, stock weights averaged.
- RFF: W = gamma * N(0,1) with gamma ~ Uniform{0.5,0.6,...,1.0} (config.GAMMA_GRID),
  chars rank-standardized to approx [-0.5,0.5] (variance 1/12 each). Features are
  [sin(Wx); cos(Wx)] with NO bias term, and then the sin/cos features are themselves
  rank-standardized again (dkkm_functions.rff line ~110).
- Ridge penalty in dkkm_functions.ridge_regr is 360*z where z = nfeatures*alpha, alpha in
  ALPHA_LST = [0, .001, .01, .05, .1, 1] (bgn/kp14) or those /100 (gs21).
- FF/FM MVE uses sklearn Ridge with alpha=0, i.e. plain OLS on 6 (or 7) factors over 360
  months. FF SMB is a simple value-weighted size split; other factors are 2x3 size x char
  sorts; FM portfolios are X(X'X)^{-1} on raw (unstandardized) chars, equal-weighted market.

AGGREGATE RISK DIMENSION (verify this yourself)
- BGN: one log-SDF shock nu plus a correlated interest-rate shock xi (corr_zr = beta_zr /
  (sigma_z*sigma_r)); cash-flow shocks load on nu through corr_zj.
- KP14: two aggregate shocks, x (aggregate productivity) and z (investment-specific), with
  prices of risk gamma_x = 0.69, gamma_z = -0.35. Firm-level shocks eps (CIR) and uj (CIR).
- GS21: ONE aggregate shock x; z is purely idiosyncratic.

## Working hypotheses from Claude (VERIFY OR REFUTE — do not assume true)

H1 SPAN. With only 1-2 priced aggregate shocks and only 5-6 characteristics that are
   near-deterministic functions of 2-3 latent states, the true conditional MVE is already
   (over-)spanned by 6 linear characteristic portfolios. Complexity has nothing to buy.
H2 RATIO NONLINEARITY. With a K-factor covariance Sigma = B Omega B' + D, the true MVE
   weight is w* = D^{-1} B c. If idiosyncratic variance d_i and loadings b_i are both
   functions of characteristics, w*(x) = b(x)/d(x) is a RATIO — nonlinear even when b and
   d are each linear. FF value-weighted sorts and FM's (X'X)^{-1}X' cannot form it; a rich
   nonlinear basis can. Currently d_i may be too homogeneous or too weakly char-linked.
H3 RFF BANDWIDTH MIS-SCALING. With L=5 rank-standardized chars, Var(w'x) = gamma^2 * L/12,
   so sd(w'x) <= 1.0*sqrt(5/12) = 0.65. sin is essentially LINEAR over that range, so the
   RFF basis is approximately {linear index, quadratic index} — barely richer than FM.
   Real-data DKKM has L ~ 130 chars, giving sd(w'x) ~ 3.3*gamma, genuinely nonlinear.
   The bandwidth should arguably be rescaled by sqrt(130/L).
H4 CHARACTERISTIC POVERTY. Real DKKM uses ~130 characteristics with many weak, partly
   independent signals; here there are 5-6, mutually redundant and noiselessly measured.
   The "virtue of complexity" theory requires the truth to be DENSE and HIGH-DIMENSIONAL
   in the feature space; with a 5-dim, low-noise, near-linear truth, a small model is
   simply the right model and ridge on 3600 features cannot win.
H5 HJD SCALE MISMATCH. sdf_ret (the true HJ portfolio return, normalized to unit cost) and
   the method returns w_t . xret have DIFFERENT and possibly wildly different scales. If
   sd(sdf_ret) >> sd(xret_method), then HJD ~ sd(sdf_ret) for every method and the metric
   is structurally incapable of separating them regardless of the economics.
H6 EVALUATION HORIZON / SNR. If the model panels have far higher cross-sectional return
   predictability (R^2) than real data (~0.5%/month), estimation is easy, OLS on 6 factors
   is efficient, and the ridge-on-many-features advantage never appears.
`;

const DIAG_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['scope', 'findings', 'verdict_on_hypotheses', 'quantitative_claims', 'blockers'],
  properties: {
    scope: { type: 'string', description: 'what this agent analyzed' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['claim', 'evidence', 'file_refs', 'confidence'],
        properties: {
          claim: { type: 'string' },
          evidence: { type: 'string', description: 'derivation or code reading that supports it' },
          file_refs: { type: 'array', items: { type: 'string' } },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
    verdict_on_hypotheses: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['hypothesis', 'verdict', 'reasoning'],
        properties: {
          hypothesis: { type: 'string', description: 'H1..H6' },
          verdict: { type: 'string', enum: ['CONFIRMED', 'PARTIALLY_CONFIRMED', 'REFUTED', 'NOT_APPLICABLE', 'UNDETERMINED'] },
          reasoning: { type: 'string' },
        },
      },
    },
    quantitative_claims: {
      type: 'array',
      description: 'numbers you derived or computed (magnitudes, dimensions, variances)',
      items: { type: 'string' },
    },
    blockers: {
      type: 'array',
      description: 'the specific structural reasons DKKM cannot beat FF/FM here, ranked most-binding first',
      items: { type: 'string' },
    },
  },
};

const PROP_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['target', 'proposals'],
  properties: {
    target: { type: 'string' },
    proposals: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['name', 'one_line', 'economic_story', 'mechanism_for_dkkm_gain', 'why_ff_fm_cannot_capture_it', 'exact_change', 'parameter_choice_and_derivation', 'predicted_effect', 'minimality', 'risks', 'implementation_cost'],
        properties: {
          name: { type: 'string' },
          one_line: { type: 'string' },
          economic_story: { type: 'string', description: 'the economics: what real-world feature this represents and why a serious referee would accept it' },
          mechanism_for_dkkm_gain: { type: 'string', description: 'formal argument: how this makes the true MVE weight function unspanned by low-dim linear char portfolios' },
          why_ff_fm_cannot_capture_it: { type: 'string' },
          exact_change: { type: 'string', description: 'concrete: which file, which lines, what code/parameter changes. Be specific.' },
          parameter_choice_and_derivation: { type: 'string', description: 'how to pick the new parameter values and what they should be calibrated to' },
          predicted_effect: { type: 'string', description: 'direction and rough magnitude on DKKM Sharpe, FF/FM Sharpe, HJD, and max_sr' },
          minimality: { type: 'string', enum: ['reparametrization_only', 'small_code_addition', 'moderate_restructuring', 'major_rewrite'] },
          risks: { type: 'array', items: { type: 'string' } },
          implementation_cost: { type: 'string', description: 'note: sdf_compute is O(N^2) per month and the true conditional second moment must remain analytically computable' },
        },
      },
    },
  },
};

const CRIT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['lens', 'target', 'verdicts', 'missed_ideas'],
  properties: {
    lens: { type: 'string' },
    target: { type: 'string' },
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['proposal_name', 'verdict', 'critique', 'would_it_actually_produce_the_gap', 'salvage'],
        properties: {
          proposal_name: { type: 'string' },
          verdict: { type: 'string', enum: ['ENDORSE', 'ENDORSE_WITH_CHANGES', 'DOUBTFUL', 'REJECT'] },
          critique: { type: 'string' },
          would_it_actually_produce_the_gap: { type: 'string', description: 'be adversarial: argue why FF or FM could ALSO capture it, or why the gap would be small' },
          salvage: { type: 'string', description: 'the strongest version of the idea that survives your critique, or empty if none' },
        },
      },
    },
    missed_ideas: { type: 'array', items: { type: 'string' }, description: 'alterations the proposer should have considered but did not' },
  },
};

// ---------------------------------------------------------------- PHASE 1
phase('Diagnose');

const DIAG_TASKS = [
  {
    key: 'bgn',
    prompt: `${CTX}

YOUR TASK: Deep structural analysis of the **BGN model** as implemented.

Read in full: utils_bgn/panel_functions_bgn.py, utils_bgn/sdf_compute_bgn.py,
utils_bgn/vasicek.py, utils_bgn/loadings_compute_bgn.py, and the BGN blocks of config.py.

Answer with derivations, not vibes:
1. How many PRICED aggregate shocks are there? Write down the SDF and each firm's exposure.
2. Derive the firm's conditional risk exposure as a function of primitives (beta_avg,
   number of live projects, the interest rate r, growth-option value G(r), assets in place A).
   Berk-Green-Naik 1999's whole point is that book-to-market and size are near-SUFFICIENT
   STATISTICS for risk. Is that literally true in this implementation? Show the mapping
   chars -> conditional beta and assess how close to linear/monotone it is.
3. Characterize the conditional covariance matrix built in sdf_compute_bgn.sdf_loop.
   What is the idiosyncratic ("diagonal") variance as a function of firm observables?
   How dispersed is it cross-sectionally? Is d_i strongly char-linked (H2)?
4. The true MVE portfolio: port = ER^{-1}1, normalized. Given the factor structure, express
   w* in the form D^{-1} B c and say what function of characteristics it is.
   Why is the MLP-on-chars R (corr 0.685) not higher — what part of w* is NOT char-measurable?
5. HEADROOM: is there any portfolio of the 6 FF factors that gets close to the true max_sr?
   Argue in population, not simulation. If FF already attains ~the max, nothing DKKM does can help.
6. rf_stand: BGN's RFF gets a cross-sectionally CONSTANT variable appended. What does
   rank_standardize do to a constant column, and what does that imply for the BGN RFF basis?
   (Read dkkm_functions.rank_standardize and rff carefully. This may be a bug.)
7. Note anything in the code that looks like a bug or an unintended degeneracy.

Return the structured object. Be quantitative wherever you can (compute variances,
dimensions, orders of magnitude from the config parameters).`,
  },
  {
    key: 'kp14',
    prompt: `${CTX}

YOUR TASK: Deep structural analysis of the **KP14 model** as implemented.

Read in full: utils_kp14/panel_functions_kp14.py, utils_kp14/sdf_compute_kp14.py,
utils_kp14/kp14_fd.py, utils_kp14/integ_kp14.py, utils_kp14/loadings_compute_kp14.py,
and the KP14 blocks of config.py.

Answer with derivations:
1. The two priced aggregate shocks x and z with gamma_x = 0.69, gamma_z = -0.35. Write down
   each firm's exposure to each. KP14's economics: assets-in-place load on x; growth options
   load on z (the IST shock) with a NEGATIVE price of risk. Confirm this is what the code does.
2. Firm heterogeneity in the code is: lambda_f (a firm-fixed Pareto-ish project arrival rate),
   the H/L arrival regime, the CIR processes eps (firm-level) and uj (project-level), and the
   project vintage/capital distribution. Which of these are recoverable from
   {size, bm, agr, roe, mom}? Which are NOT? Quantify the information loss.
3. Idiosyncratic variance: KP14_SIGMA_EPS = 0.1*0.2 and KP14_SIGMA_U = 0.1*1.5, i.e. both were
   cut by a factor of 10 at some point. Derive what that did to (a) the cross-sectional
   dispersion of d_i, (b) the conditioning of cond_var, (c) the size of the diversifiable vs
   undiversifiable component. Is the panel now nearly noiseless in the cross-section (H6)?
4. There is a known artifact: the true MVE weight ends up proportional to size because idio
   return variance scales like 1/size, so it ANTI-correlates with the empirical size premium
   (this explains the NEGATIVE corr_fitted_vs_truth of -0.12 to -0.15 in correlations.csv).
   Verify this from the code. Is the true MVE dominated by the inverse-idiosyncratic-variance
   term rather than by risk premia? Quantify the split of max_sr into "diversification" vs
   "priced risk exposure" components.
5. HEADROOM: with 2 priced shocks and 5 chars, can 6 FF factors already span the tangency?
6. Note bugs, degeneracies, or calibration choices that make the cross-section too easy.

Return the structured object. Be quantitative.`,
  },
  {
    key: 'gs21',
    prompt: `${CTX}

YOUR TASK: Deep structural analysis of the **GS21 model** as implemented.

Read in full: utils_gs21/panel_functions_gs21.py, utils_gs21/sdf_compute_gs21.py,
utils_gs21/loadings_compute_gs21.py, and the GS21 blocks of config.py. Also read
/Users/sjpruitt/GitHub/bop-analyze-remote/analyze_gs21_panels.py.

Answer with derivations:
1. GS21 appears to have exactly ONE aggregate shock x (z is idiosyncratic). Confirm.
   If there is one priced factor, the true conditional MVE is (up to the D^{-1} tilt) a
   single factor-mimicking portfolio. What does that imply for the maximum possible
   DKKM-over-FF gap? Be blunt.
2. correlations.csv says the TRUE GS21 SDF weight is 98% explained by an MLP on the 6 chars,
   yet every method achieves under 0.10. Explain this gap precisely. Is it (a) the methods
   optimize expected-return-per-risk from noisy realized returns rather than matching w*,
   (b) w* is dominated by the D^{-1} idiosyncratic-variance tilt which buys diversification
   but little expected return, (c) estimation noise, or (d) something else? Decompose.
3. Leverage and DEFAULT: GS21 has endogenous default (P <= 0 triggers replacement) and
   debt refinancing with a kinked payoff. This is the one genuine NONLINEARITY across the
   three models: value is a kinked/convex function of (z, x, b). Characterize how nonlinear
   the mapping from (bm, mkt_lev, size, roe) to conditional beta actually is. Does the
   default region create the size x value x leverage INTERACTION that would reward a
   nonlinear basis? If the default probability is tiny, the nonlinearity is measure-zero:
   compute the implied default frequency from GS21_ZETA and the price function.
4. Study the default-firm handling: defaulted firms get P=0, are replaced, and their SDF
   weights blow up (the MLP README mentions true SDF weights up to 1e9 in GS21). Is the
   true MVE being driven by a handful of near-default firms? Does that corrupt both metrics?
5. GS21's sdf_compute does NOT exclude zero/negative-book firms the way BGN and KP14 now do.
   Assess whether GS21 has an analogous singularity problem.
6. HEADROOM and bugs.

Return the structured object. Be quantitative.`,
  },
  {
    key: 'pipeline',
    prompt: `${CTX}

YOUR TASK: Forensic audit of the **estimation and evaluation pipeline** — is the DKKM
estimator, as implemented, even capable of showing a virtue of complexity? And are the
metrics capable of measuring it?

Read in full: utils_factors/dkkm_functions.py, utils_factors/fama_functions.py,
utils_factors/factor_utils.py, utils_factors/sdf_utils.py, utils/generate_dkkm_factors.py,
utils/generate_fama_factors.py, utils/estimate_sdf_dkkm.py, utils/estimate_sdf_fama.py,
utils/evaluate_sdfs.py, utils/calculate_moments.py, utils/generate_25_portfolios.py, config.py,
and in the analyze repo: create_slides_tables.py, analyze.py, compute_pricing_errors.py.

Answer precisely:
1. RFF BANDWIDTH (H3). Compute the actual distribution of the pre-activation Wx with
   gamma ~ U{0.5..1.0} and L rank-standardized chars. Quantify how nonlinear sin and cos
   really are at that scale (e.g. what fraction of the variance of sin(Wx) is explained by
   a linear projection on x, and of cos(Wx) by a quadratic). Then state what gamma range
   would reproduce the effective bandwidth of the real-data DKKM implementation with ~130
   chars. This is potentially the single highest-leverage change and it is not a model change.
2. The DOUBLE rank-standardization in dkkm_functions.rff (line ~110 rank-standardizes the
   sin/cos features after they are formed). What does this do to the RKHS / the span?
   Does it destroy the amplitude information needed to represent w*(x) = b(x)/d(x)?
   Is it in DKKM's actual algorithm?
3. NO BIAS TERM: the code computes [sin(Wx); cos(Wx)] with no random phase b ~ U[0,2pi].
   Assess whether that matters for the span given both sin and cos are included.
4. Ridge scaling: the effective penalty is 360 * nfeatures * alpha with alpha in
   [0,.001,.01,.05,.1,1]. Given the factor returns' scale, is this grid centered anywhere
   near the optimal shrinkage? Compute the implied effective degrees of freedom at
   nfeatures=3600, T=360. Is the reported "best alpha" at a grid boundary (which would mean
   the grid is mis-centered)? Note the analysis code picks the best alpha per model
   post hoc, so a mis-centered grid caps DKKM's measured performance.
5. NMAT=5 AVERAGING: estimate_sdf_dkkm averages the STOCK WEIGHTS across 5 independent W
   draws. Is averaging weights across independent random feature draws equivalent to, better
   than, or worse than one big feature set? Does it shrink DKKM toward a smoother/flatter
   (i.e. more linear) function and thereby destroy exactly the nonlinearity that complexity
   is supposed to buy?
6. FAIRNESS: FF/FM MVE is OLS (alpha=0) on 6-7 factors over 360 months — well-conditioned
   and low variance. DKKM at nfeatures=3600 with T=360 is in the over-parameterized regime.
   Is the comparison as implemented biased toward FF/FM? Would FF/FM also improve with ridge?
   Also: BGN's RFF gets rf_stand (a conditioning variable) that FF/FM do NOT get — is that
   an unfair advantage or, given rank_standardize collapses constant columns, a dead input?
7. THE SHARPE METRIC: it evaluates w_t against TRUE conditional moments, so there is no
   evaluation noise — only estimation noise in w_t. Note that this metric rewards getting
   the DIRECTION of w* right and is scale-invariant per month. Is it therefore a fair
   playing field, or does it have a subtle bias (e.g. is mean(mean_t/stdev_t) over months
   the right aggregation vs a proper unconditional Sharpe)?
8. THE HJD METRIC (H5). Establish the scale of sdf_ret versus the scale of w_t . xret for
   each method. port = ER^{-1}1 normalized by port.sum() — work out what units that leaves
   sdf_ret in, and what units the method MVE weights (from ridge on y=ones) leave xret in.
   If they differ by an order of magnitude, HJD is structurally uninformative. Propose the
   correct scale-free version (e.g. regress method return on sdf_ret and report 1-R^2, or
   normalize both to unit variance, or the true HJ distance
   sqrt(min E[(m - m_hat)^2]) computed properly).
9. Anything else in the pipeline that mechanically prevents a complexity advantage.

Return the structured object. Do real arithmetic; cite line numbers.`,
  },
  {
    key: 'theory',
    prompt: `${CTX}

YOUR TASK: Establish, from asset-pricing and statistical-learning theory, the NECESSARY AND
SUFFICIENT conditions under which a high-dimensional ridge-on-random-features SDF estimator
beats a low-dimensional sorted-factor / cross-sectional-regression estimator. Then say what
the real-data cross-section has that these three models lack.

You may search the web and use your knowledge of:
- Didisheim, Ke, Kelly, Malamud "Complexity in Factor Pricing Models" / "APT or AIPT"
- Kelly, Malamud, Zhou "The Virtue of Complexity in Return Prediction" (JF)
- Kelly, Pruitt, Su IPCA; Kozak-Nagel-Santosh "Shrinking the cross-section"
- Random matrix theory / ridgeless interpolation double descent, the c = P/T ratio
- Fama-French factor spanning, Barillas-Shanken

Deliver:
1. The precise conditions for a virtue of complexity: state them in terms of (a) the
   dimension and sparsity of the true SDF weight function g*(x), (b) the signal-to-noise
   ratio, (c) the ratio c = P/T, (d) the eigenvalue spectrum of the managed-portfolio
   second-moment matrix. Be formal. In particular: if g* is exactly linear in K << P
   characteristics and SNR is high, what does the theory predict about ridge-on-P-features
   versus OLS-on-K-factors? (Answer honestly — it predicts NO advantage, or a disadvantage.)
2. Quantify the real-data benchmark: roughly how much higher is the DKKM out-of-sample
   Sharpe than FF5/FF6 and Fama-MacBeth in the published results? What monthly R^2 and
   what number of characteristics do they use? What is the empirical shape of the
   "complexity" curve (Sharpe vs number of features)?
3. List, ranked, the FEATURES OF REAL EQUITY CROSS-SECTIONS that create the virtue of
   complexity and that a 2-shock production-based model with 5 characteristics does not
   have. For each: name it, state the economic mechanism, and state how it would show up in
   a structural model.
4. Critically: distinguish gains that come from (i) SPAN (the true g* is nonlinear /
   interactive in chars), (ii) DENSITY (many weak signals, so shrinkage over many features
   dominates selection over few), (iii) CONDITIONING (time-varying risk prices, so weights
   must depend on macro state), (iv) EFFICIENT WEIGHTING (the D^{-1} tilt: inverse
   idiosyncratic variance), (v) pure estimation-theoretic double-descent. For each, say
   whether the Sharpe metric used here (population conditional Sharpe of the estimated
   weights against true conditional moments) can even detect it.
5. Give a checklist a structural modeler can apply to any candidate model change:
   "does this change create a virtue of complexity?" — with a diagnostic that can be run
   on a simulated panel WITHOUT rerunning the full estimator (e.g. regress the true w* on
   linear chars vs on a nonlinear basis and compare R^2; the gap IS the maximum attainable
   DKKM advantage).

Return the structured object. The 'blockers' field should be your ranked list of what the
models are missing.`,
  },
];

const diagnoses = await parallel(
  DIAG_TASKS.map((t) => () =>
    agent(t.prompt, { label: `diag:${t.key}`, phase: 'Diagnose', schema: DIAG_SCHEMA, effort: 'high' })
  )
);

const diagOk = diagnoses.filter(Boolean);
log(`Diagnosis complete: ${diagOk.length}/${DIAG_TASKS.length} agents returned`);

const diagDigest = diagOk
  .map((d) => `### ${d.scope}
BLOCKERS (ranked): ${(d.blockers || []).join(' | ')}
HYPOTHESIS VERDICTS: ${(d.verdict_on_hypotheses || []).map((v) => `${v.hypothesis}=${v.verdict} (${v.reasoning})`).join(' ;; ')}
KEY FINDINGS: ${(d.findings || []).map((f) => `[${f.confidence}] ${f.claim} -- ${f.evidence}`).join('\n  ')}
NUMBERS: ${(d.quantitative_claims || []).join(' | ')}`)
  .join('\n\n');

// ---------------------------------------------------------------- PHASE 2+3 (pipelined)
phase('Propose');

const PROPOSAL_TARGETS = [
  {
    key: 'bgn',
    brief: `Propose minimal alterations to the **BGN model** (utils_bgn/) that would make DKKM
substantially beat FF and FM on Sharpe and HJD. BGN's identity is: firms are portfolios of
heterogeneous-risk projects plus growth options; book-to-market and size are (by design)
near-sufficient statistics for risk. Any change must not destroy that identity outright but
may relax the sufficiency. Levers available in this code: the project systematic-exposure
distribution (currently a single shifted exponential, beta ~ -Expon(loc=-beta_star, scale)),
the cash-flow volatility rule sigmaj ~ U[|beta|/sigma_z, |beta|/sigma_z + 0.1*0.3*|Cbar|]
(note the hand-inserted 0.1 factor on line 54 of panel_functions_bgn.py), the survival
probability PI=0.99, the investment threshold exp(Cbar - beta)*D(r) > 1, the interest-rate
process, and the SDF/rate correlation beta_zr. Also consider what additional CHARACTERISTICS
the econometrician could be given that the model already generates (num projects, investment
rate, cash-flow volatility, payout yield, firm age, multi-horizon past returns, lagged bm).`,
  },
  {
    key: 'kp14',
    brief: `Propose minimal alterations to the **KP14 model** (utils_kp14/) that would make DKKM
substantially beat FF and FM. KP14's identity is: two priced aggregate shocks (neutral
productivity x with positive price of risk, investment-specific z with NEGATIVE price of
risk gamma_z=-0.35); growth options load on z, assets in place on x; firm heterogeneity in
project arrival rate lambda_f and the H/L regime. Levers: gamma_x/gamma_z, sigma_x/sigma_z,
the firm-level CIR processes (SIGMA_EPS=0.02, SIGMA_U=0.15, THETA_EPS=0.35, THETA_U=0.5 —
note both sigmas were cut 10x from their original values), the lambda_f distribution
(mu_lambda=2, sigma_lambda=2, exponential-in-log-uniform), the H/L regime parameters, alpha
(returns to scale 0.85), delta. Also consider additional characteristics the model already
generates (R&D/investment intensity, capital vintage/age, arrival-regime proxies, cash-flow
volatility, project count). Address head-on the known artifact that the true MVE weight is
proportional to size via the 1/size idiosyncratic-variance channel and therefore
ANTI-correlates with the size premium.`,
  },
  {
    key: 'gs21',
    brief: `Propose minimal alterations to the **GS21 model** (utils_gs21/) that would make DKKM
substantially beat FF and FM. GS21's identity is: a production economy with an Epstein-Zin
representative agent (GAMMA=10, PSI=2), ONE aggregate shock x, idiosyncratic z, endogenous
investment with a random fixed cost, endogenous leverage with a Poisson refinancing
opportunity (ZETA=0.01), tax shields (TAU), issuance/bankruptcy costs, and ENDOGENOUS
DEFAULT. The one-aggregate-shock structure is the binding constraint; but the default and
refinancing kinks are the one genuine nonlinearity available across the three models. Levers:
sigma_x/rho_x, sigma_z/rho_z, ZETA (refinancing frequency), TAU, KAPPA_B, the investment
cost distribution U[IMIN=0, IMAX=2000], the default boundary, and the value-function grids
in GS21_solfiles (changing the model's solved policy functions requires rerunning the MATLAB
solver GS21.m — flag anything that would require that). Also consider adding a SECOND
aggregate shock (e.g. an aggregate financing/credit-spread shock, or stochastic volatility
of x, or a shock to the refinancing/issuance cost) and say exactly how much resolving
machinery that would require. Also consider additional characteristics the model already
generates (book leverage, distance-to-default, investment rate, payout, cash-flow vol, age
since default).`,
  },
  {
    key: 'crosscut',
    brief: `Propose CROSS-CUTTING, model-agnostic changes that apply to all three models
and/or to the econometrician's information set and the estimator. This is the place for:
(a) recalibrating the RFF bandwidth GAMMA_GRID to the low-dimensional characteristic space;
(b) expanding the characteristic set from 5-6 to a realistic number (with model-generated,
individually weak, partially redundant chars, including lags and multi-horizon transforms);
(c) adding realistic accounting measurement error, reporting lags, and staleness to
characteristics — with an argument about whether errors-in-variables actually helps or hurts
a nonlinear estimator relative to linear sorts;
(d) adding conditioning/macro-state variables to the feature set, and the FAIRNESS question
of whether FF/FM must be given the same conditioning;
(e) fixing the double rank-standardization and the NMAT weight-averaging if they are
destroying nonlinearity;
(f) recentering the ridge alpha grid;
(g) replacing HJD with a scale-free version so the metric can actually separate methods;
(h) raising idiosyncratic volatility so the cross-section has realistic signal-to-noise;
(i) whether to increase N and/or T.
For each, be explicit about whether it is a MODEL change, a MEASUREMENT change (what the
econometrician sees), an ESTIMATOR change, or a METRIC change — and about the referee risk
of each category. Estimator/metric changes are cheap but a referee will ask whether you
tuned the estimator to win.`,
  },
];

const CRITIC_LENSES = [
  {
    key: 'structure',
    instruction: `You are an ADVERSARIAL critic with the ECONOMIC STRUCTURE lens. You are a
production-based asset pricing theorist who knows Berk-Green-Naik (1999), Kogan-Papanikolaou
(2014) and Gomes-Schmid (2021) intimately. For each proposal ask: does this change break the
model's core economic mechanism or its identity as "the BGN/KP14/GS21 model"? Does it violate
internal consistency (e.g. is the SDF still the marginal rate of substitution of the stated
agent? does the firm's policy remain optimal given the change? does the value function
solution in the shipped solfiles remain valid, or does the MATLAB/finite-difference solver
have to be rerun?). Does it break the model's calibration targets (aggregate equity premium,
market Sharpe, volatility, investment rate, leverage, default rate, size/value premia)? Is
the change ad hoc — a free parameter inserted purely to manufacture the result — or does it
correspond to a real economic force with independent empirical support? Would the paper's
claim "these are SIMPLE ECONOMIC MODELS" survive the change? Be harsh. Name any proposal that
amounts to rigging.`,
  },
  {
    key: 'mechanism',
    instruction: `You are an ADVERSARIAL critic with the MECHANISM-SKEPTIC lens. Your default
position is that the proposed change will NOT produce a DKKM advantage, and you must try to
prove that. For each proposal, construct the strongest argument that (a) Fama-MacBeth
cross-sectional-regression portfolios would capture the same thing, since FM spans ANY LINEAR
function of the characteristics and is re-estimated every month — so only genuine
NONLINEARITY or genuine EXTRA DIMENSIONS help; (b) the FF 2x3 sorts would capture it via the
size interaction already built into the double sort; (c) the gain is second-order in
magnitude (do the arithmetic: how many basis points of Sharpe?); (d) the change adds noise
faster than signal, so ridge-on-3600-features loses; (e) the metric used (population
conditional Sharpe against true conditional moments, and HJD) cannot detect the change even
if the economics is right. Also check for the failure mode where the change makes the TRUE
MVE weight blow up on a handful of extreme firms (near-default, near-zero-book, tiny idio
variance), which corrupts max_sr and both metrics without teaching the estimator anything.
Where you cannot refute a proposal, say so explicitly and explain why it survives.`,
  },
  {
    key: 'referee',
    instruction: `You are an ADVERSARIAL critic with the REFEREE / PUBLICATION lens. You are
a hostile referee at the Journal of Finance reading a paper that claims to show the virtue of
complexity arises in simple structural models. For each proposal ask: does this look like the
authors changed the model until they got the answer they wanted? Is the change disclosed and
defensible, or would it need to be buried? What robustness check would you demand, and would
the result survive it? Is the comparison between DKKM and FF/FM FAIR after the change — same
information set, same regularization opportunity, same number of tuning decisions? (Pay
particular attention to any proposal that gives DKKM information or tuning that FF/FM does
not get, e.g. conditioning variables, a re-centered penalty grid, post-hoc selection of the
best alpha and best nfeatures.) Would you accept the resulting paper? What is the single
sentence in your report that would kill it? Also: rank the proposals by how much of the
result would be attributable to economics versus to estimator/metric plumbing, since a
referee will read a plumbing-driven result as no result at all.`,
  },
];

const critiques = await pipeline(
  PROPOSAL_TARGETS,
  (t) =>
    agent(
      `${CTX}

## Diagnosis phase results (from five parallel analyst agents)

${diagDigest}

## YOUR TASK

${t.brief}

Ground rules:
- MINIMAL changes preferred. Rank your proposals by (impact on the DKKM-vs-FF/FM gap) / (size
  of the change). State minimality honestly.
- DERIVE the reasoning. For every parameter value you propose, say what it is calibrated to
  and show the arithmetic. "Raise sigma" is not a proposal; "raise sigma_eps from 0.02 to X
  because that makes the cross-sectional dispersion of idiosyncratic monthly volatility match
  the 10th-90th percentile range of Y in real data, which makes the true MVE weight's
  nonlinear-in-chars component Z% of its variance" is a proposal.
- For each proposal you MUST state the formal reason FM (which spans every linear function of
  the characteristics, re-estimated monthly) and FF (2x3 double sorts, which already contain a
  size interaction) cannot capture what DKKM would capture. If you cannot state that reason,
  the proposal is dead — say so.
- Respect implementation reality: sdf_compute_*.py must still be able to compute the exact
  N x N conditional second-moment matrix analytically or by quadrature each month; the true
  conditional moments are what both metrics are scored against. A change that makes ER
  incomputable is not viable. Say when a change requires re-solving the model
  (KP14 finite differences in kp14_fd.py; GS21's MATLAB-produced solfiles).
- Give 4 to 6 proposals. Include at least one that you consider high-risk/high-reward and at
  least one that is nearly free to implement.
- Read the actual code before proposing. Cite file and line numbers in exact_change.

Return the structured object.`,
      { label: `propose:${t.key}`, phase: 'Propose', schema: PROP_SCHEMA, effort: 'high' }
    ),
  (prop, t) => {
    if (!prop) return null;
    const propText = (prop.proposals || [])
      .map(
        (p, i) => `--- PROPOSAL ${i + 1}: ${p.name} (${p.minimality})
one line: ${p.one_line}
economics: ${p.economic_story}
mechanism for DKKM gain: ${p.mechanism_for_dkkm_gain}
why FF/FM cannot: ${p.why_ff_fm_cannot_capture_it}
exact change: ${p.exact_change}
parameters + derivation: ${p.parameter_choice_and_derivation}
predicted effect: ${p.predicted_effect}
risks: ${(p.risks || []).join('; ')}
implementation cost: ${p.implementation_cost}`
      )
      .join('\n\n');
    return parallel(
      CRITIC_LENSES.map((lens) => () =>
        agent(
          `${CTX}

## Diagnosis phase results

${diagDigest}

## Proposals under review (target: ${t.key})

${propText}

## YOUR ROLE

${lens.instruction}

Read the underlying code yourself where you need to check a claim — do not take the
proposer's word for what the code does. Be specific and cite files/lines.
For EVERY proposal give a verdict. Then list ideas the proposer MISSED.

Return the structured object with lens="${lens.key}" and target="${t.key}".`,
          { label: `crit:${t.key}:${lens.key}`, phase: 'Critique', schema: CRIT_SCHEMA, effort: 'high' }
        )
      )
    ).then((cs) => ({ target: t.key, proposals: prop, critiques: cs.filter(Boolean) }));
  }
);

const bundles = critiques.filter(Boolean);
log(`Proposal+critique complete for ${bundles.length}/${PROPOSAL_TARGETS.length} targets`);

// ---------------------------------------------------------------- PHASE 4
phase('Synthesize');

const bundleText = bundles
  .map((b) => {
    const props = (b.proposals.proposals || [])
      .map(
        (p) => `  * ${p.name} [${p.minimality}]: ${p.one_line}
      mechanism: ${p.mechanism_for_dkkm_gain}
      why FF/FM cannot: ${p.why_ff_fm_cannot_capture_it}
      change: ${p.exact_change}
      params: ${p.parameter_choice_and_derivation}
      predicted: ${p.predicted_effect}
      risks: ${(p.risks || []).join('; ')}`
      )
      .join('\n');
    const crits = (b.critiques || [])
      .map(
        (c) => `  [lens=${c.lens}]
${(c.verdicts || []).map((v) => `      ${v.proposal_name}: ${v.verdict} -- ${v.critique} | gap-check: ${v.would_it_actually_produce_the_gap} | salvage: ${v.salvage}`).join('\n')}
      MISSED: ${(c.missed_ideas || []).join(' | ')}`
      )
      .join('\n');
    return `## TARGET: ${b.target}\nPROPOSALS:\n${props}\nCRITIQUES:\n${crits}`;
  })
  .join('\n\n');

const FINAL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['root_cause', 'recommended_program', 'per_model', 'rejected', 'diagnostics_to_run_first', 'open_questions'],
  properties: {
    root_cause: {
      type: 'string',
      description: 'the definitive, evidence-backed explanation of why DKKM does not beat FF/FM here. 2-4 paragraphs.',
    },
    recommended_program: {
      type: 'array',
      description: 'ordered plan of what to change, cheapest-and-most-certain first',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['step', 'category', 'change', 'rationale', 'expected_gap_effect', 'survived_critique', 'residual_objection'],
        properties: {
          step: { type: 'string' },
          category: { type: 'string', enum: ['estimator', 'metric', 'measurement', 'calibration', 'model_structure'] },
          change: { type: 'string', description: 'concrete: file, lines, parameter values' },
          rationale: { type: 'string', description: 'the derivation' },
          expected_gap_effect: { type: 'string' },
          survived_critique: { type: 'string', description: 'which critics endorsed and on what grounds' },
          residual_objection: { type: 'string', description: 'the strongest surviving objection and how to answer it' },
        },
      },
    },
    per_model: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['model', 'binding_constraint', 'primary_recommendation', 'secondary_recommendation', 'do_not_do'],
        properties: {
          model: { type: 'string' },
          binding_constraint: { type: 'string' },
          primary_recommendation: { type: 'string' },
          secondary_recommendation: { type: 'string' },
          do_not_do: { type: 'string' },
        },
      },
    },
    rejected: {
      type: 'array',
      description: 'proposals killed in critique, with the reason',
      items: { type: 'string' },
    },
    diagnostics_to_run_first: {
      type: 'array',
      description: 'cheap checks on EXISTING simulated panels that would confirm or refute the diagnosis before any expensive rerun',
      items: { type: 'string' },
    },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
};

const synthesis = await agent(
  `${CTX}

## Diagnosis phase results

${diagDigest}

## Proposals and adversarial critiques

${bundleText}

## YOUR TASK

Synthesize all of the above into a single decision-ready program. You are the senior author.

Requirements:
1. ROOT CAUSE. State definitively why DKKM does not beat FF/FM in these simulations.
   Distinguish sharply between (i) estimator/metric plumbing defects, (ii) calibration
   choices, and (iii) genuine missing economics. Say which one dominates and why. If the
   evidence says the dominant cause is plumbing rather than economics, SAY SO — the authors
   need the truth, not a confirmation of their premise.
2. RECOMMENDED PROGRAM. An ordered list. Put the cheap, certain, high-leverage fixes first
   (these are usually estimator/metric), then measurement, then calibration, then genuine
   model-structure changes. For each, give the exact change and the derivation of any
   parameter value. Only include items that survived adversarial critique, and for each,
   record the strongest surviving objection and the answer to it.
3. PER MODEL. For BGN, KP14, GS21: the binding constraint, the primary and secondary
   recommendation, and the one thing NOT to do.
4. REJECTED. Everything killed in critique and the killing argument, so the authors do not
   re-derive it later.
5. DIAGNOSTICS TO RUN FIRST. Cheap checks on panels the authors ALREADY have that would
   confirm or refute the diagnosis before committing cluster time to a rerun. The most
   important one: regress the true SDF stock weight w* on (a) the linear span of the
   characteristics and (b) a rich nonlinear basis of the same characteristics, and compare
   R^2 — the gap IS an upper bound on the attainable DKKM-over-FM advantage. Specify these
   concretely enough to code, naming the existing pickles and columns.
6. OPEN QUESTIONS.

Be decisive and quantitative. Do not hedge everything; rank and commit.`,
  { label: 'synthesis', phase: 'Synthesize', schema: FINAL_SCHEMA, effort: 'high' }
);

return { synthesis, diagnoses: diagOk, bundles };
