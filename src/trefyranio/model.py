"""Dynamic Bayesian poll-aggregation model (Phase 3).

Top-down national model of the eight Riksdag parties (+ "other"), in
additive-log-ratio (ALR) space relative to S. The latent vote share follows a
**damped local-linear-trend** — it carries both a *level* and a *velocity*, so
momentum is estimated and projected to election day. This is the deliberate fix
for the inertia that makes plain moving-averages / random-walks lag a party
that is genuinely rising or falling.

  velocity_t = phi * velocity_{t-1} + sigma_vel * z          (phi < 1 damps it)
  level_t    = level_{t-1} + velocity_{t-1} + sigma_lvl * z

Polls enter through a Dirichlet-Multinomial likelihood (Multinomial plus
over-dispersion for nonsampling error), with a per-pollster house effect added
in ALR space. The walk is anchored at the known previous election result.

Output: a weekly trend (mean + credible band per party) and a posterior sample
of election-day shares that Phase 4 turns into seats.

Refs: Economist/Gelman dynamic Bayesian model; Harvey local-linear-trend.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.diagnostics import split_gelman_rubin

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Party order; index 0 (S) is the ALR reference category.
PARTY_ORDER = ["S", "M", "SD", "C", "V", "KD", "MP", "L", "Övr"]
REF = 0
K = len(PARTY_ORDER)
KM1 = K - 1

# Riksdagsval election days (second Sunday of September).
ELECTION_DATES = {
    1998: dt.date(1998, 9, 20), 2002: dt.date(2002, 9, 15),
    2006: dt.date(2006, 9, 17), 2010: dt.date(2010, 9, 19),
    2014: dt.date(2014, 9, 14), 2018: dt.date(2018, 9, 9),
    2022: dt.date(2022, 9, 11), 2026: dt.date(2026, 9, 13),
}


@dataclass
class CycleConfig:
    """One election cycle: the inter-election window the model fits over."""
    start: dt.date            # day after the previous election (walk anchor)
    election_day: dt.date     # the election we forecast
    prev_election_year: int   # whose result anchors the walk start


def cycle_for(year: int) -> CycleConfig:
    """Build the cycle config for an election year (4-year cycles)."""
    prev = year - 4
    return CycleConfig(
        start=ELECTION_DATES[prev] + dt.timedelta(days=1),
        election_day=ELECTION_DATES[year],
        prev_election_year=prev,
    )


CYCLE_2026 = cycle_for(2026)

# Election-day polling-miss spread, in SHARE space (pp). Applied in
# post-processing (no likelihood). Share space — not ALR — because realized poll
# errors are ~uniform in pp across party sizes.
#
# HONEST CAVEAT: this spread is a calibrated add-on, NOT emergent from the latent
# walk — whose innovation variance is estimated tiny on the slowly-moving
# inter-election series and so projects to election day with false certainty. A
# hand-fit term carries the forecast error. The principled fix (model-carried
# error: an Economist-style backward-from-election-day random walk with
# horizon-accumulating innovations + an explicit election-day fundamentals prior)
# is deliberate future work; see README "Uncertainty model".
#
# HORIZON-DEPENDENT: calibrated at TWO horizons (election-eve H≈0 and H=14w) on
# the converged model across FOUR cycles (2010/2014/2018/2022) → a variance-
# accumulation curve. H = weeks from the last poll to election; uncertainty grows
# with H (the forecast tightens as the election nears). Coverage-calibrated (85%
# of the 80% interval) rather than moment-matched: with 4 cycles the realized
# errors are heavy-tailed, and the converged model's own posterior already widens
# with horizon, so moment-match under-covers far out (66% at H=14).
MISS_SIGMA_FLOOR = 0.015        # 1.50pp election-eve (H≈0), coverage-cal
MISS_SIGMA_VAR_SLOPE = 3.375e-6  # share² added per week → 1.65pp at H=14


def miss_sigma_for_horizon(weeks: float) -> float:
    """Share-space polling-miss sigma at ``weeks`` from the last poll to election."""
    return float(np.sqrt(MISS_SIGMA_FLOOR ** 2 + MISS_SIGMA_VAR_SLOPE * max(0.0, weeks)))


MISS_SIGMA = miss_sigma_for_horizon(14)  # ≈0.0225; default for the live ~14w horizon

# Cross-party correlation of the polling miss. Real misses co-move within a bloc
# (parties draw from a shared pool / a common pro-anti-incumbent mood), so an
# iid miss understates the variance of BLOC totals — and the headline government
# probabilities are bloc arithmetic. Modelled as a factor: each party's miss =
# sqrt(rho)·(shared bloc factor) + sqrt(1-rho)·(idiosyncratic). This keeps each
# party's MARGINAL sigma = MISS_SIGMA (so the share-space calibration still
# holds) while giving within-bloc correlation = rho; renormalisation supplies the
# cross-bloc anti-correlation. rho is an INFORMED ASSUMPTION, set deliberately
# modest: cross-national evidence says within-bloc misses correlate and the
# dangerous direction is UNDER-correlating (iid → overconfident bloc tails), but
# Sweden's 2-cycle backtest shows realized bloc errors ≤ the iid prediction, so a
# large rho would over-widen. 0.2 hedges toward the theory without over-claiming a
# magnitude the data can't confirm. Calibrate with more cycles. Groups mirror
# simulate's blocs. (rho=0.2 lifts P(Tidö maj) ~11%→14% vs iid.)
MISS_RHO = 0.2
_MISS_GROUP = {"S": 1, "M": 0, "SD": 0, "C": 2, "V": 1, "KD": 0, "MP": 1, "L": 0, "Övr": 3}
MISS_GROUP_IDX = np.array([_MISS_GROUP[p] for p in PARTY_ORDER])


@dataclass
class ModelData:
    counts: np.ndarray        # (n_polls, K) integer party counts
    totals: np.ndarray        # (n_polls,) row sums
    week: np.ndarray          # (n_polls,) week index of each poll
    pollster: np.ndarray      # (n_polls,) pollster index
    n_weeks: int              # T (election week inclusive)
    election_week: int        # index of election day
    anchor_alr: np.ndarray    # (KM1,) ALR of the previous result
    pollsters: list[str]
    weeks_dates: list[dt.date]
    # Phase-2 ratings, wired in (zeros/ones/zeros when use_ratings=False):
    house_prior_alr: np.ndarray      # (P, KM1) per-pollster ALR house-effect prior MEAN
    poll_weight: np.ndarray          # (n_polls,) accuracy weight of each poll's pollster
    industry_shift_share: np.ndarray # (K,) share-space election-day field-bias correction


def _alr(p: np.ndarray) -> np.ndarray:
    """Additive-log-ratio relative to the reference category."""
    p = np.clip(p, 1e-6, None)
    return np.log(np.delete(p, REF) / p[REF])


# Industry-bias shrinkage. The field-wide miss (e.g. the persistent SD/S
# underestimate) is estimated on only 4 elections and pollsters have partly
# adapted, so we apply only a fraction of it as an election-day correction.
# NOTE: when the model-carried-error rework lands, this share-space shift becomes
# the election-day fundamentals/anchor prior's mean offset — a one-line move.
SHRINK_IND = 0.30


def _build_house_priors(pollsters: list[str], anchor_share: np.ndarray,
                        house_eff: pd.DataFrame, industry: pd.DataFrame) -> np.ndarray:
    """Per-pollster ALR house-effect prior MEANS from history → (P, KM1).

    ratings.py measures house effects vs the ACTUAL result (= a pollster's lean
    vs the field PLUS the field-wide industry bias) in SHARE space. The model's
    `house` is vs the poll CONSENSUS in ALR. So we (1) de-bias — subtract the
    field-wide `industry_bias` so each prior is the pollster's lean *relative to
    the field* (≈ consensus), which is also what the model's centring expects;
    (2) map share→ALR exactly via the Jacobian at the `anchor` baseline. New
    entrants (no history) get a zero prior. No explicit centring here: the
    de-bias already leaves matched priors ≈field-mean-zero, and the model's
    `house -= house.mean(0)` enforces identifiability."""
    P = len(pollsters)
    ind = {r.party: r.bias for r in industry.itertuples()}
    industry_vec = np.array([ind.get(p, 0.0) for p in PARTY_ORDER])      # (K,)
    he = house_eff.set_index(["pollster", "party"])["house_effect"]
    prior = np.zeros((P, KM1))
    for i, name in enumerate(pollsters):
        effect = np.array([he.get((name, p), 0.0) for p in PARTY_ORDER])  # (K,) share
        if not np.any(effect):                                            # new entrant
            continue
        debiased = effect - industry_vec
        perturbed = np.clip(anchor_share + debiased, 1e-6, None)
        perturbed = perturbed / perturbed.sum()
        prior[i] = _alr(perturbed) - _alr(anchor_share)
    return prior


def _load_pollster_weights(pollsters: list[str], ratings: pd.DataFrame) -> np.ndarray:
    """Per-pollster accuracy weight (centred on 1.0); 1.0 for new entrants → (P,)."""
    w = ratings.set_index("pollster")["weight"]
    return np.array([float(w.get(name, 1.0)) for name in pollsters])


def prepare(polls: pd.DataFrame, results: pd.DataFrame,
            cycle: CycleConfig = CYCLE_2026, as_of: dt.date | None = None,
            use_ratings: bool = True) -> ModelData:
    """Assemble model inputs from the poll + result spines for a cycle.

    Polls are restricted to the cycle window [start, election_day]. ``as_of``
    additionally drops polls taken after that date — used to backtest at a
    chosen *horizon* (e.g. cut 14 weeks before the election to match how far the
    live forecast sits from its last poll). The forecast still targets election
    day, so the gap between ``as_of`` and election day is the forecast horizon.

    ``use_ratings`` wires in the Phase-2 ratings layer (house-effect priors,
    accuracy weights, industry-bias correction). Backtests set it False: the
    ratings span 2010–2022, so applying them to a pre-2022 cycle would leak
    future results — and the backtest's job is to test the uncertainty model,
    not the warm-start."""
    cutoff = min(cycle.election_day, as_of) if as_of else cycle.election_day
    df = polls.copy()
    df["date"] = df["field_end"].fillna(df["pub_date"])
    df = df[(df["date"].dt.date >= cycle.start)
            & (df["date"].dt.date <= cutoff)
            & df["n"].notna()]

    # Wide shares per poll; fold FI into the "other" bucket to match results.
    wide = df.pivot_table(index="poll_id", columns="party", values="share", aggfunc="first")
    if "FI" in wide:
        wide["Övr"] = wide["Övr"].fillna(0) + wide["FI"].fillna(0)
    wide = wide.reindex(columns=PARTY_ORDER).fillna(0.0)

    meta = df.groupby("poll_id").agg(
        n=("n", "first"), date=("date", "first"), pollster=("pollster", "first")
    )
    wide, meta = wide.align(meta, join="inner", axis=0)

    counts = np.round(wide.to_numpy() * meta["n"].to_numpy()[:, None]).astype(int)
    counts = np.clip(counts, 0, None)
    totals = counts.sum(axis=1)
    keep = totals > 0
    counts, totals = counts[keep], totals[keep]
    meta = meta[keep]

    week = ((meta["date"].dt.date - cycle.start).apply(lambda d: d.days) // 7).to_numpy()
    election_week = (cycle.election_day - cycle.start).days // 7
    n_weeks = election_week + 1

    pollsters = sorted(meta["pollster"].unique())
    p_idx = {h: i for i, h in enumerate(pollsters)}
    pollster = meta["pollster"].map(p_idx).to_numpy()

    prev = results[results["election_year"] == cycle.prev_election_year].set_index("party")["share"]
    anchor = np.array([prev.get(p, 1e-6) for p in PARTY_ORDER])
    anchor = anchor / anchor.sum()

    weeks_dates = [cycle.start + dt.timedelta(weeks=int(w)) for w in range(n_weeks)]

    # Phase-2 ratings layer. Guarded: missing files (or use_ratings=False) →
    # neutral defaults (zero priors, unit weights, no industry shift).
    P = len(pollsters)
    house_prior = np.zeros((P, KM1))
    poll_weight = np.ones(len(counts))
    industry_shift = np.zeros(K)
    he_path = PROCESSED_DIR / "pollster_house_effects.parquet"
    ind_path = PROCESSED_DIR / "industry_bias.parquet"
    rat_path = PROCESSED_DIR / "pollster_ratings.parquet"
    if use_ratings and he_path.exists() and ind_path.exists() and rat_path.exists():
        industry = pd.read_parquet(ind_path)
        house_prior = _build_house_priors(pollsters, anchor,
                                          pd.read_parquet(he_path), industry)
        pollster_w = _load_pollster_weights(pollsters, pd.read_parquet(rat_path))
        poll_weight = pollster_w[pollster]
        ind = {r.party: r.bias for r in industry.itertuples()}
        # Field UNDERSTATES a party (bias<0) → push its election-day share UP.
        industry_shift = np.array([-SHRINK_IND * ind.get(p, 0.0) for p in PARTY_ORDER])
    elif use_ratings:
        print("  ratings parquets missing → neutral priors/weights (run ratings.build())")

    return ModelData(
        counts=counts, totals=totals, week=week, pollster=pollster,
        n_weeks=n_weeks, election_week=election_week, anchor_alr=_alr(anchor),
        pollsters=pollsters, weeks_dates=weeks_dates,
        house_prior_alr=house_prior, poll_weight=poll_weight,
        industry_shift_share=industry_shift,
    )


def _full_logits(alr: jnp.ndarray) -> jnp.ndarray:
    """Prepend the reference category's logit (0) to an ALR vector → full
    logits over all K parties. Reference is index 0 (S)."""
    pad = jnp.zeros(alr.shape[:-1] + (1,))
    return jnp.concatenate([pad, alr], axis=-1)


# Latent scales — FIXED, not sampled. Sampling a global scale that multiplies many
# non-centred innovations is Neal's funnel, and that (× the velocity) was the main
# cause of non-convergence (r-hat 15-34, ESS≈2). Fixing the scales removes the
# funnels; values picked so the trend tracks the polls without overfitting.
SIGMA_LVL = 0.03      # per-week level innovation (ALR)
SIGMA_HOUSE = 0.05    # per-pollster house-effect scale (ALR)
KAPPA = 200.0         # Dirichlet-Multinomial concentration (overdispersion)
DRIFT_SIGMA = 0.0015  # per-week per-party drift prior (ALR) — momentum, projected
SIGMA_VEL = 0.006     # velocity innovation (only if use_velocity; also fixed)
VEL_DAMP = 0.8        # velocity damping


def model(data: ModelData, use_velocity: bool = False):
    """Local-level random walk in ALR with FIXED scales (convergence fix) + per-
    pollster house effects (centred). All scale parameters are constants, so the
    target is well-conditioned and NUTS mixes (no Neal's funnel). Velocity is OFF
    by default — it was the main mixing culprit and its momentum gain was marginal;
    the optional branch keeps a *fixed*-scale damped velocity for A/B only.
    Likelihood: Dirichlet-Multinomial with fixed overdispersion."""
    T, P = data.n_weeks, len(data.pollsters)

    # House effects: non-centred around a per-pollster PRIOR MEAN from history
    # (data.house_prior_alr; zeros when ratings off). The residual ~Normal(0,
    # SIGMA_HOUSE) lets the current cycle override. Still centred across pollsters
    # so the latent path is the consensus (identifiability vs the level).
    house_prior = jnp.asarray(data.house_prior_alr)
    house = numpyro.sample("house", dist.Normal(0, SIGMA_HOUSE).expand([P, KM1]).to_event(2))
    house = house_prior + house
    house = house - house.mean(axis=0)
    numpyro.deterministic("house_effective", house)  # the actual per-pollster effect

    level0 = numpyro.sample(
        "level0", dist.Normal(jnp.asarray(data.anchor_alr), 0.15).to_event(1)
    )
    # Per-party drift: a single well-identified slope per party (no funnel) that
    # carries momentum — projected forward to election day. Replaces the
    # per-week velocity, which broke mixing.
    drift = numpyro.sample("drift", dist.Normal(0, DRIFT_SIGMA).expand([KM1]).to_event(1))
    z_lvl = numpyro.sample("z_lvl", dist.Normal(0, 1).expand([T, KM1]).to_event(2))
    z_vel = (numpyro.sample("z_vel", dist.Normal(0, 1).expand([T, KM1]).to_event(2))
             if use_velocity else jnp.zeros((T, KM1)))

    def step(carry, zs):
        level, vel = carry
        zl, zv = zs
        vel = VEL_DAMP * vel + SIGMA_VEL * zv
        level = level + drift + vel + SIGMA_LVL * zl
        return (level, vel), level

    _, levels = jax.lax.scan(step, (level0, jnp.zeros(KM1)), (z_lvl, z_vel))
    numpyro.deterministic("levels", levels)

    obs_alr = levels[data.week] + house[data.pollster]      # (n_polls, KM1)
    p_obs = jax.nn.softmax(_full_logits(obs_alr), axis=-1)  # (n_polls, K)
    # Accuracy weights enter as per-poll concentration: a more accurate pollster
    # gets higher Dirichlet concentration → lower over-dispersion → its poll
    # constrains the latent more tightly. (Scaling total_count would be invalid —
    # it's the integer trial count.) Unit weights when ratings off.
    kappa_vec = KAPPA * jnp.asarray(data.poll_weight)[:, None]
    numpyro.sample(
        "counts",
        dist.DirichletMultinomial(kappa_vec * jnp.clip(p_obs, 1e-7, 1.0),
                                  total_count=jnp.asarray(data.totals)),
        obs=jnp.asarray(data.counts),
    )


def fit(data: ModelData, warmup: int = 600, samples: int = 500, seed: int = 0,
        use_velocity: bool = False, num_chains: int = 4) -> dict:
    # Multiple chains (vectorized) → a seed-robust posterior + an r-hat check.
    # init_to_median starts at the prior medians (drift≈0, level0=anchor) — avoids
    # the extreme-logit init the cumulative drift can otherwise produce.
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy=init_to_median)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=num_chains,
                chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), data, use_velocity=use_velocity)

    # Convergence check on the election-day shares.
    lev = np.asarray(mcmc.get_samples(group_by_chain=True)["levels"])[:, :, data.election_week, :]
    full = np.concatenate([np.zeros(lev.shape[:-1] + (1,)), lev], axis=-1)
    e = np.exp(full - full.max(-1, keepdims=True))
    sh = e / e.sum(-1, keepdims=True)
    worst = max(float(split_gelman_rubin(sh[:, :, k])) for k in range(sh.shape[-1]))
    print(f"convergence: worst election-day r-hat {worst:.3f}"
          + ("" if worst < 1.05 else "  ⚠️ NOT CONVERGED"))
    return mcmc.get_samples()


def _shares_from_levels(levels: np.ndarray) -> np.ndarray:
    """(samples, T, KM1) ALR levels → (samples, T, K) shares."""
    pad = np.zeros(levels.shape[:-1] + (1,))
    full = np.concatenate([pad, levels], axis=-1)
    e = np.exp(full - full.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def summarize(samples: dict, data: ModelData) -> tuple[pd.DataFrame, np.ndarray]:
    # Trend line: the smooth latent path (no miss term).
    shares = _shares_from_levels(np.asarray(samples["levels"]))  # (S, T, K)
    mean = shares.mean(0)
    lo, hi = np.quantile(shares, [0.1, 0.9], axis=0)
    rows = []
    for t, d in enumerate(data.weeks_dates):
        for k, party in enumerate(PARTY_ORDER):
            rows.append((d, party, mean[t, k], lo[t, k], hi[t, k]))
    trend = pd.DataFrame(rows, columns=["date", "party", "mean", "lo", "hi"])

    # The pure trend (ALR) at the election week — miss error added separately.
    election_alr_trend = np.asarray(samples["levels"])[:, data.election_week, :]
    return trend, election_alr_trend


def _softmax_with_ref(alr: np.ndarray) -> np.ndarray:
    """(..., KM1) ALR → (..., K) shares, reference logit 0 prepended."""
    full = np.concatenate([np.zeros(alr.shape[:-1] + (1,)), alr], axis=-1)
    e = np.exp(full - full.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def forecast_with_miss(
    election_alr_trend: np.ndarray, sigma_miss: float = MISS_SIGMA,
    rho: float = MISS_RHO, seed: int = 0, industry_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Election-day share samples (S, K): the trend at election week mapped to
    shares, optionally shifted by the field-bias correction, plus a SHARE-space
    polling-miss error, clipped to >=0 and renormalized.

    ``industry_shift`` (K,) is the heavily-shrunk field-wide bias correction
    (e.g. push SD/S up where the industry persistently understates them). Applied
    to the central forecast BEFORE the miss; it shifts the mean, the miss adds
    spread around it.

    The miss is correlated within blocs via a factor model (see MISS_RHO): each
    party's miss = sqrt(rho)·shared-bloc-factor + sqrt(1-rho)·idiosyncratic. This
    leaves the marginal per-party sigma at ``sigma_miss`` (so the Phase-5
    share-space calibration holds) while inflating bloc-total variance — which is
    what the government-formation probabilities depend on. Share space (not ALR)
    because realized poll errors are ~uniform in pp across party sizes."""
    rng = np.random.default_rng(seed)
    base = _softmax_with_ref(election_alr_trend)              # (S, K) shares
    if industry_shift is not None:
        base = np.clip(base + industry_shift, 0.0, None)
        base = base / base.sum(axis=-1, keepdims=True)
    ngroups = int(MISS_GROUP_IDX.max()) + 1
    factor = rng.standard_normal((base.shape[0], ngroups))[:, MISS_GROUP_IDX]  # shared per bloc
    idio = rng.standard_normal(base.shape)
    miss = sigma_miss * (np.sqrt(rho) * factor + np.sqrt(1.0 - rho) * idio)
    noisy = np.clip(base + miss, 0.0, None)
    return noisy / noisy.sum(axis=-1, keepdims=True)


def build(warmup: int = 600, samples: int = 600) -> None:
    # 600 samples is adequate for the election-day mean (2000 gave an identical
    # result). Seed-robustness comes from multi-chain averaging — fit() runs 4
    # vectorized chains with an r-hat check — not from sample count.
    polls = pd.read_parquet(PROCESSED_DIR / "polls.parquet")
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    data = prepare(polls, results)
    print(f"fitting: {len(data.counts)} polls, {len(data.pollsters)} pollsters, "
          f"{data.n_weeks} weeks → election week {data.election_week}")

    posterior = fit(data, warmup=warmup, samples=samples)
    trend, election_alr_trend = summarize(posterior, data)

    # Forecast horizon = weeks from the last poll to election day → horizon-
    # dependent miss sigma (tighter as the election nears).
    horizon = int(data.election_week - data.week.max())
    sigma = miss_sigma_for_horizon(horizon)
    shift = data.industry_shift_share
    print(f"horizon {horizon}w → miss sigma {sigma*100:.2f}pp; "
          f"industry shift (pp): " + ", ".join(
              f"{p}{shift[k]*100:+.2f}" for k, p in enumerate(PARTY_ORDER) if abs(shift[k]) > 1e-4))
    election = forecast_with_miss(election_alr_trend, sigma_miss=sigma, industry_shift=shift)

    trend.to_parquet(PROCESSED_DIR / "model_trend.parquet", index=False)
    np.savez(
        PROCESSED_DIR / "forecast_samples.npz",
        shares=election, election_alr_trend=election_alr_trend,
        parties=np.array(PARTY_ORDER), election_day=str(CYCLE_2026.election_day),
        miss_sigma=sigma, horizon_weeks=horizon, industry_shift=shift,
    )
    _report(trend, election, CYCLE_2026.election_day)


def _report(trend: pd.DataFrame, election: np.ndarray, election_day) -> None:
    print(f"\nelection-day forecast ({election_day}) — mean [10–90%]:")
    order = np.argsort(-election.mean(0))
    for k in order:
        p = PARTY_ORDER[k]
        s = election[:, k]
        flag = ""
        if p not in ("Övr",):
            below = (s < 0.04).mean()
            if 0 < below < 1:
                flag = f"   P(<4%) = {below*100:.0f}%"
        print(f"  {p:4} {s.mean()*100:5.1f}%  [{np.quantile(s,0.1)*100:4.1f}–"
              f"{np.quantile(s,0.9)*100:4.1f}]{flag}")


if __name__ == "__main__":
    build()
