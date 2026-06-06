"""Backtest & calibration (Phase 5).

Retrodicts past elections from polls-only: for each cycle we refit the model on
just the polls taken before that election and compare the forecast to the known
result. Two jobs:

1. **Calibrate MISS_SIGMA.** The latent walk's own variance is over-confident;
   the polling-miss term must be sized so the election-day predictive spread
   matches realized poll-vs-result error. We moment-match it across cycles:
       MISS_SIGMA² = mean(error²) − mean(trend_variance)      (in ALR space)
   then check 80% interval coverage.

2. **Test the momentum thesis.** Refit each cycle with velocity OFF (plain
   random walk) and compare signed error for *rising* parties. If the velocity
   term works, it should reduce the systematic under-prediction of parties
   trending up into the election.

Fits are slow (~3 min each, 4 fits), so `fit_all()` saves raw posteriors and
`analyze()` reads them — run analysis as many times as you like without refit.

  python -m trefyranio.backtest fit       # ~12 min, writes backtests/
  python -m trefyranio.backtest analyze    # instant
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trefyranio.model import (
    PARTY_ORDER,
    PROCESSED_DIR,
    _alr,
    _softmax_with_ref,
    cycle_for,
    fit,
    prepare,
    summarize,
)

BACKTEST_YEARS = [2018, 2022]   # current party system, dense polling
BT_DIR = PROCESSED_DIR / "backtests"
N_PARTIES = 8                   # the eight Riksdag parties (exclude Övr)
Z80 = 1.2816                    # 10–90% half-width in sd units
# Forecast horizon to calibrate at: weeks from last poll to election. Matches
# the live 2026 forecast (~14 weeks from the last poll to election day).
HORIZON_WEEKS = 14
COVERAGE_TARGET = 0.85          # mildly conservative interval coverage


def _tag(year: int, use_velocity: bool) -> str:
    return f"bt_{year}_{'vel' if use_velocity else 'lvl'}"


def actual_shares(results: pd.DataFrame, year: int) -> np.ndarray:
    r = results[results["election_year"] == year].set_index("party")["share"]
    v = np.array([r.get(p, 1e-6) for p in PARTY_ORDER])
    return v / v.sum()


def fit_all(warmup: int = 400, samples: int = 400) -> None:
    polls = pd.read_parquet(PROCESSED_DIR / "polls.parquet")
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    BT_DIR.mkdir(parents=True, exist_ok=True)

    for year in BACKTEST_YEARS:
        cycle = cycle_for(year)
        data = prepare(polls, results, cycle)
        for use_velocity in (True, False):
            post = fit(data, warmup=warmup, samples=samples, use_velocity=use_velocity)
            trend, election_alr_trend = summarize(post, data)
            tw = (trend.pivot(index="date", columns="party", values="mean")
                  .reindex(columns=PARTY_ORDER))
            np.savez(
                BT_DIR / f"{_tag(year, use_velocity)}.npz",
                election_alr_trend=election_alr_trend,
                trend_mean=tw.to_numpy(),
                actual=actual_shares(results, year),
                parties=np.array(PARTY_ORDER),
                election_week=data.election_week,
                n_polls=len(data.counts),
            )
            print(f"  fit {year} velocity={use_velocity}: {len(data.counts)} polls saved")


def fit_horizon(warmup: int = 400, samples: int = 400) -> None:
    """Refit each cycle with polls cut HORIZON_WEEKS before the election
    (velocity on) — the right horizon to size the live forecast's uncertainty."""
    polls = pd.read_parquet(PROCESSED_DIR / "polls.parquet")
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    BT_DIR.mkdir(parents=True, exist_ok=True)
    for year in BACKTEST_YEARS:
        cycle = cycle_for(year)
        as_of = cycle.election_day - dt.timedelta(weeks=HORIZON_WEEKS)
        data = prepare(polls, results, cycle, as_of=as_of)
        post = fit(data, warmup=warmup, samples=samples, use_velocity=True)
        _, election_alr_trend = summarize(post, data)
        np.savez(
            BT_DIR / f"bt_{year}_h{HORIZON_WEEKS}.npz",
            election_alr_trend=election_alr_trend,
            actual=actual_shares(results, year),
            n_polls=len(data.counts), last_week=int(data.week.max()),
        )
        print(f"  fit {year} @H={HORIZON_WEEKS}w: {len(data.counts)} polls "
              f"(last poll wk {int(data.week.max())} of {data.election_week})")


def _load(year: int, use_velocity: bool):
    return np.load(BT_DIR / f"{_tag(year, use_velocity)}.npz", allow_pickle=True)


def _load_horizon(year: int):
    return np.load(BT_DIR / f"bt_{year}_h{HORIZON_WEEKS}.npz", allow_pickle=True)


def coverage(errs: np.ndarray, sd_trend: np.ndarray, miss: float) -> float:
    total_sd = np.sqrt(sd_trend ** 2 + miss ** 2)
    return float((np.abs(errs / total_sd) < Z80).mean())


def calibrate_coverage(errs: np.ndarray, sd_trend: np.ndarray,
                       target: float = COVERAGE_TARGET) -> float:
    """Smallest MISS_SIGMA whose 80% interval reaches the coverage target —
    robust to heavy tails (a few big misses won't over-widen the whole band)."""
    grid = np.linspace(0.0, 0.1, 201)  # share-space pp, 0.05pp resolution
    covs = np.array([coverage(errs, sd_trend, m) for m in grid])
    reached = grid[covs >= target]
    return float(reached[0]) if len(reached) else float(grid[-1])


def calibrate_horizon() -> dict:
    """Calibrate the SHARE-space MISS_SIGMA at the live forecast's horizon.

    Share space (not ALR): realized poll errors are ~uniform in pp across party
    sizes, so a scalar share-space miss is well-behaved where an ALR one is not.
    Calibrated on the eight Riksdag parties."""
    errs, sd_trend = [], []
    for year in BACKTEST_YEARS:
        d = _load_horizon(year)
        shares = _softmax_with_ref(d["election_alr_trend"])      # (S, K)
        errs.append((shares.mean(0) - d["actual"])[:N_PARTIES])  # share-space
        sd_trend.append(shares.std(0)[:N_PARTIES])
    errs, sd_trend = np.concatenate(errs), np.concatenate(sd_trend)
    miss_mm = miss_from_moments(errs, sd_trend)
    miss_cov = calibrate_coverage(errs, sd_trend)
    return {
        "miss_moment_match": miss_mm,
        "miss_coverage": miss_cov,
        "coverage_at_mm": coverage(errs, sd_trend, miss_mm),
        "coverage_at_cov": coverage(errs, sd_trend, miss_cov),
        "errs": errs, "sd_trend": sd_trend,
    }


def miss_from_moments(errs: np.ndarray, sd_trend: np.ndarray) -> float:
    """Added-variance moment match: size the polling-miss term so total
    predictive variance (trend variance + miss²) equals realized squared error.
    Clamped at 0 when the trend spread already covers the errors."""
    return float(np.sqrt(max(0.0, (errs ** 2).mean() - (sd_trend ** 2).mean())))


def calibrate() -> tuple[float, np.ndarray, np.ndarray]:
    """Moment-match MISS_SIGMA across the velocity-on backtests (ALR space)."""
    errs, sd_trend = [], []
    for year in BACKTEST_YEARS:
        d = _load(year, True)
        eat = d["election_alr_trend"]                 # (S, KM1)
        errs.append(_alr(d["actual"]) - eat.mean(0))
        sd_trend.append(eat.std(0))
    errs, sd_trend = np.concatenate(errs), np.concatenate(sd_trend)
    return miss_from_moments(errs, sd_trend), errs, sd_trend


def analyze() -> None:
    print(f"=== MISS_SIGMA calibration @ horizon {HORIZON_WEEKS}w (live-forecast gap) ===")
    print("    share-space (pp); calibrated on the 8 Riksdag parties")
    h = calibrate_horizon()
    print(f"  moment-match : {h['miss_moment_match']*100:.2f}pp  "
          f"→ 80% coverage {h['coverage_at_mm']*100:.0f}%")
    print(f"  coverage-cal : {h['miss_coverage']*100:.2f}pp  "
          f"→ 80% coverage {h['coverage_at_cov']*100:.0f}%  (target {COVERAGE_TARGET*100:.0f}%)  ← MISS_SIGMA")

    print("\n=== point accuracy (mean forecast vs actual, velocity on, H=0) ===")
    for year in BACKTEST_YEARS:
        d = _load(year, True)
        fc = _softmax_with_ref(d["election_alr_trend"]).mean(0)
        mae = np.abs(fc - d["actual"])[:N_PARTIES].mean() * 100
        print(f"  {year}: MAE {mae:.2f}pp   (n_polls={int(d['n_polls'])})")

    _momentum()


def _momentum() -> None:
    """Signed election-day error for rising parties, velocity on vs off.

    Rising = trend rose >0.5pp over the final 12 weeks. Under-prediction shows
    as positive signed error (actual > forecast); velocity should shrink it."""
    print("\n=== momentum check: rising-party signed error (velocity on vs off) ===")
    agg = {"vel": [], "lvl": []}
    for year in BACKTEST_YEARS:
        for use_velocity, key in ((True, "vel"), (False, "lvl")):
            d = _load(year, use_velocity)
            tm, actual, ew = d["trend_mean"], d["actual"], int(d["election_week"])
            fc = tm[ew]
            window = tm[max(0, ew - 12):ew + 1]
            rising = (window[-1] - window[0])[:N_PARTIES] > 0.005
            signed = (actual - fc)[:N_PARTIES]
            if rising.any():
                agg[key].append(signed[rising])
                names = [PARTY_ORDER[i] for i in range(N_PARTIES) if rising[i]]
                print(f"  {year} {key}: rising {names}  "
                      f"mean signed err {signed[rising].mean()*100:+.2f}pp")
    for key, label in (("vel", "velocity ON "), ("lvl", "velocity OFF")):
        if agg[key]:
            m = np.concatenate(agg[key]).mean() * 100
            print(f"  → {label}: mean under-prediction of rising parties {m:+.2f}pp")
    if agg["vel"] and agg["lvl"]:
        v, l = np.concatenate(agg["vel"]).mean(), np.concatenate(agg["lvl"]).mean()
        verdict = "REDUCES" if abs(v) < abs(l) else "does NOT reduce"
        print(f"  verdict: velocity {verdict} rising-party under-prediction "
              f"({abs(l)*100:.2f} → {abs(v)*100:.2f} pp)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("fit", "all"):
        fit_all()
    if cmd in ("fit-horizon", "all"):
        fit_horizon()
    if cmd in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
