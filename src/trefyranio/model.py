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
from numpyro.infer import MCMC, NUTS

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Party order; index 0 (S) is the ALR reference category.
PARTY_ORDER = ["S", "M", "SD", "C", "V", "KD", "MP", "L", "Övr"]
REF = 0
K = len(PARTY_ORDER)
KM1 = K - 1

# Riksdagsval election days (second Sunday of September).
ELECTION_DATES = {
    2010: dt.date(2010, 9, 19), 2014: dt.date(2014, 9, 14),
    2018: dt.date(2018, 9, 9), 2022: dt.date(2022, 9, 11),
    2026: dt.date(2026, 9, 13),
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
# 2018 & 2022 → a variance-accumulation curve. H = weeks from the last poll to
# election; uncertainty grows with H (the forecast tightens as the election nears).
MISS_SIGMA_FLOOR = 0.016        # ≈1.6pp election-eve (H≈0), coverage-cal
MISS_SIGMA_VAR_SLOPE = 1.79e-5  # share² added per week (from the 0→14w calibration)


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


def _alr(p: np.ndarray) -> np.ndarray:
    """Additive-log-ratio relative to the reference category."""
    p = np.clip(p, 1e-6, None)
    return np.log(np.delete(p, REF) / p[REF])


def prepare(polls: pd.DataFrame, results: pd.DataFrame,
            cycle: CycleConfig = CYCLE_2026, as_of: dt.date | None = None) -> ModelData:
    """Assemble model inputs from the poll + result spines for a cycle.

    Polls are restricted to the cycle window [start, election_day]. ``as_of``
    additionally drops polls taken after that date — used to backtest at a
    chosen *horizon* (e.g. cut 14 weeks before the election to match how far the
    live forecast sits from its last poll). The forecast still targets election
    day, so the gap between ``as_of`` and election day is the forecast horizon."""
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
    return ModelData(
        counts=counts, totals=totals, week=week, pollster=pollster,
        n_weeks=n_weeks, election_week=election_week, anchor_alr=_alr(anchor),
        pollsters=pollsters, weeks_dates=weeks_dates,
    )


def _full_logits(alr: jnp.ndarray) -> jnp.ndarray:
    """Prepend the reference category's logit (0) to an ALR vector → full
    logits over all K parties. Reference is index 0 (S)."""
    pad = jnp.zeros(alr.shape[:-1] + (1,))
    return jnp.concatenate([pad, alr], axis=-1)


def model(data: ModelData, use_velocity: bool = True):
    """Damped local-linear-trend poll model. With ``use_velocity=False`` the
    velocity component is dropped, leaving a plain local-level random walk — the
    A/B baseline for testing whether momentum reduces rising-party lag."""
    T, P = data.n_weeks, len(data.pollsters)

    sigma_lvl = numpyro.sample("sigma_lvl", dist.HalfNormal(0.05))
    kappa = numpyro.sample("kappa", dist.Gamma(2.0, 0.02))          # over-dispersion
    sigma_house = numpyro.sample("sigma_house", dist.HalfNormal(0.1))

    house = numpyro.sample("house", dist.Normal(0, sigma_house).expand([P, KM1]).to_event(2))
    # Identify house effects: center them so the latent path is the consensus.
    house = house - house.mean(axis=0)

    level0 = numpyro.sample(
        "level0", dist.Normal(jnp.asarray(data.anchor_alr), 0.15).to_event(1)
    )
    z_lvl = numpyro.sample("z_lvl", dist.Normal(0, 1).expand([T, KM1]).to_event(2))

    if use_velocity:
        sigma_vel = numpyro.sample("sigma_vel", dist.HalfNormal(0.01))  # small: shrink momentum
        phi = numpyro.sample("phi", dist.Beta(8.0, 2.0))               # damping ~0.8
        z_vel = numpyro.sample("z_vel", dist.Normal(0, 1).expand([T, KM1]).to_event(2))
    else:
        sigma_vel, phi, z_vel = 0.0, 0.0, jnp.zeros((T, KM1))

    def step(carry, zs):
        level, vel = carry
        zl, zv = zs
        vel = phi * vel + sigma_vel * zv
        level = level + vel + sigma_lvl * zl
        return (level, vel), level

    init = (level0, jnp.zeros(KM1))
    _, levels = jax.lax.scan(step, init, (z_lvl, z_vel))   # levels: (T, KM1)
    numpyro.deterministic("levels", levels)

    obs_alr = levels[data.week] + house[data.pollster]      # (n_polls, KM1)
    p_obs = jax.nn.softmax(_full_logits(obs_alr), axis=-1)  # (n_polls, K)
    numpyro.sample(
        "counts",
        dist.DirichletMultinomial(kappa * p_obs, total_count=jnp.asarray(data.totals)),
        obs=jnp.asarray(data.counts),
    )


def fit(data: ModelData, warmup: int = 600, samples: int = 600, seed: int = 0,
        use_velocity: bool = True) -> dict:
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), data, use_velocity=use_velocity)
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
    rho: float = MISS_RHO, seed: int = 0,
) -> np.ndarray:
    """Election-day share samples (S, K): the trend at election week mapped to
    shares, plus a SHARE-space polling-miss error, clipped to >=0 and
    renormalized.

    The miss is correlated within blocs via a factor model (see MISS_RHO): each
    party's miss = sqrt(rho)·shared-bloc-factor + sqrt(1-rho)·idiosyncratic. This
    leaves the marginal per-party sigma at ``sigma_miss`` (so the Phase-5
    share-space calibration holds) while inflating bloc-total variance — which is
    what the government-formation probabilities depend on. Share space (not ALR)
    because realized poll errors are ~uniform in pp across party sizes."""
    rng = np.random.default_rng(seed)
    base = _softmax_with_ref(election_alr_trend)              # (S, K) shares
    ngroups = int(MISS_GROUP_IDX.max()) + 1
    factor = rng.standard_normal((base.shape[0], ngroups))[:, MISS_GROUP_IDX]  # shared per bloc
    idio = rng.standard_normal(base.shape)
    miss = sigma_miss * (np.sqrt(rho) * factor + np.sqrt(1.0 - rho) * idio)
    noisy = np.clip(base + miss, 0.0, None)
    return noisy / noisy.sum(axis=-1, keepdims=True)


def build(warmup: int = 800, samples: int = 2000) -> None:
    # More posterior samples → a stabler election-day mean. With only ~600 the
    # forecast drifted run-to-run on the same data (near-threshold parties like KD
    # are sensitive); 2000 calms the Monte-Carlo noise for a recompute-on-demand
    # forecast. Costs a few extra minutes per fit.
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
    print(f"horizon {horizon}w → miss sigma {sigma*100:.2f}pp")
    election = forecast_with_miss(election_alr_trend, sigma_miss=sigma)

    trend.to_parquet(PROCESSED_DIR / "model_trend.parquet", index=False)
    np.savez(
        PROCESSED_DIR / "forecast_samples.npz",
        shares=election, election_alr_trend=election_alr_trend,
        parties=np.array(PARTY_ORDER), election_day=str(CYCLE_2026.election_day),
        miss_sigma=sigma, horizon_weeks=horizon,
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
