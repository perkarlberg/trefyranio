"""Backtest & calibration (Phase 5).

Retrodicts past elections from polls-only: for each cycle we refit the converged
(fixed-scale) model on just the polls taken before that election and compare the
forecast to the known result. Two jobs:

1. **Calibrate the MISS_SIGMA sigma(H) curve.** The latent walk's own variance is
   over-confident; the polling-miss term must be sized so the election-day
   predictive spread matches realized poll-vs-result error. We fit at two
   horizons — election-eve (H=0) and the live forecast's gap (H=14w) — pooled
   across four cycles, and coverage-calibrate (smallest miss whose 80% interval
   reaches 85% coverage). Moment-matching under-covers far out (the converged
   posterior already widens with horizon), so coverage is the honest target.
   Result: sigma(H) = sqrt(floor^2 + slope*H), ~1.50pp at H=0 → ~1.65pp at H=14.

2. **Test the momentum thesis.** Project the recent ALR slope forward, damped by
   phi over the H-week gap (Gardner & McKenzie 1985), and check whether any
   (phi, W) beats a flat carry of the trend at the cut. On 4 cycles the effect is
   noise-level and the argmax is undamped (overfit), so the live model ships with
   NO separate momentum term — the per-party drift already captures trend.

Four cycles: 2010, 2014, 2018, 2022 (current party system, dense polling). Fits
are slow (~3 min each, 8 fits), so `fit_all()` saves raw posteriors to
conv_{year}_h{H}.npz and `analyze()` reads them — re-run analysis without refit.

  python -m trefyranio.backtest fit        # ~25 min, writes backtests/conv_*.npz
  python -m trefyranio.backtest analyze     # instant
"""

from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

from trefyranio.model import (
    FWD_FLOOR_PQ,
    MISS_RHO,
    PARTY_ORDER,
    PROCESSED_DIR,
    _alr,
    _softmax_with_ref,
    cycle_for,
    fit,
    miss_sigma_for_horizon,
    prepare,
    project_to_election,
    summarize,
)

BACKTEST_YEARS = [2010, 2014, 2018, 2022]   # current party system, dense polling
HORIZONS = [0, 14]                          # election-eve and the live ~14w gap
BT_DIR = PROCESSED_DIR / "backtests"
N_PARTIES = 8                   # the eight Riksdag parties (exclude Övr)
Z80 = 1.2816                    # 10–90% half-width in sd units
H_FAR = 14                      # live-forecast horizon (weeks from last poll)
COVERAGE_TARGET = 0.85          # mildly conservative interval coverage


def actual_shares(results: pd.DataFrame, year: int) -> np.ndarray:
    r = results[results["election_year"] == year].set_index("party")["share"]
    v = np.array([r.get(p, 1e-6) for p in PARTY_ORDER])
    return v / v.sum()


def _path(year: int, H: int):
    return BT_DIR / f"conv_{year}_h{H}.npz"


def fit_all(warmup: int = 500, samples: int = 500, force: bool = False) -> None:
    """Refit each cycle at each horizon with the converged model; cache to disk.
    Skips files that already exist unless ``force``."""
    polls = pd.read_parquet(PROCESSED_DIR / "polls.parquet")
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    BT_DIR.mkdir(parents=True, exist_ok=True)
    for year in BACKTEST_YEARS:
        cycle = cycle_for(year)
        for H in HORIZONS:
            out = _path(year, H)
            if out.exists() and not force:
                print(f"  skip {out.name} (exists)", flush=True)
                continue
            as_of = cycle.election_day - dt.timedelta(weeks=H) if H else None
            # use_ratings=False: ratings span 2010–2022, so applying them to a
            # past cycle leaks future results. Backtests test the uncertainty
            # model, not the warm-start.
            data = prepare(polls, results, cycle, as_of=as_of, use_ratings=False)
            post = fit(data, warmup=warmup, samples=samples, seed=0, num_chains=2)
            trend, election_alr_trend = summarize(post, data)
            tw = (trend.pivot(index="date", columns="party", values="mean")
                  .reindex(columns=PARTY_ORDER).to_numpy())
            last_week = int(data.week.max())
            np.savez(
                out,
                election_alr_trend=election_alr_trend,
                trend_mean=tw,
                actual=actual_shares(results, year),
                election_week=int(data.election_week),
                last_week=last_week,
                # Model-carried calibration inputs: the last-poll-week latent + drift
                # that project_to_election projects forward.
                last_alr=np.asarray(post["levels"])[:, last_week, :],
                drift=np.asarray(post["drift"]),
            )
            print(f"  fit {year} H={H}: {len(data.counts)} polls "
                  f"(last wk {int(data.week.max())}/{data.election_week}) → {out.name}",
                  flush=True)


def _load(year: int, H: int):
    return np.load(_path(year, H), allow_pickle=True)


# --- Model-carried calibration: tune the forward projection, not an added miss.
# project_to_election starts from the well-pinned LAST-POLL-WEEK latent, so
# sigma_share is the TOTAL election-day predictive share-space sigma (the thing
# that should match realized error), not a miss added on top of a latent spread.

def _project(d, H: int, sigma: float, floor_pq: float, rho: float):
    """Election-day share samples (S, K) for a cached cycle, via the forward
    projection at an explicit sigma_share / floor_pq / rho."""
    return project_to_election(d["last_alr"], d["drift"], H, sigma_share=sigma,
                               floor_pq=floor_pq, rho=rho, seed=0)


def _coverage(H: int, sigma: float, floor_pq: float, rho: float,
              parties=range(N_PARTIES), lo_hi=(10, 90)) -> float:
    """Pooled 80%-interval coverage of the projected shares vs actual, over the
    given parties × cycles at horizon H."""
    inside = total = 0
    for year in BACKTEST_YEARS:
        d = _load(year, H)
        sh = _project(d, H, sigma, floor_pq, rho)
        lo, hi = np.percentile(sh, lo_hi, axis=0)
        for k in parties:
            total += 1
            inside += int(lo[k] <= d["actual"][k] <= hi[k])
    return inside / total if total else float("nan")


def calibrate_forward(rho: float = MISS_RHO, floor_pq: float = FWD_FLOOR_PQ) -> dict:
    """Find the sigma(H) curve whose forward projection reaches the coverage
    target at H=0 and H=14, pooled over the 8 Riksdag parties."""
    grid = np.linspace(0.005, 0.06, 111)

    def smallest(H: int) -> float:
        for s in grid:
            if _coverage(H, s, floor_pq, rho) >= COVERAGE_TARGET:
                return float(s)
        return float(grid[-1])

    floor = smallest(0)
    s14 = max(smallest(H_FAR), floor)             # monotone in horizon
    return {"floor": floor, "s14": s14, "slope": (s14 ** 2 - floor ** 2) / H_FAR}


def _near_threshold_parties(d, lo=0.03, hi=0.08):
    return [k for k in range(N_PARTIES) if lo <= d["actual"][k] <= hi]


def momentum_test() -> dict:
    """Damped recent-momentum on the H=14 fits: does projecting the recent ALR
    slope forward (damped by phi over the 14wk gap) beat a flat carry?"""
    cycles = []
    for year in BACKTEST_YEARS:
        d = _load(year, H_FAR)
        tm = d["trend_mean"]                       # (T, K) share path
        lvl = np.log(np.clip(tm[:, 1:], 1e-9, 1) / np.clip(tm[:, :1], 1e-9, 1))
        cycles.append((lvl, int(d["last_week"]), _alr(d["actual"])))

    def damped_factor(phi: float, H: int) -> float:
        if phi <= 0:
            return 0.0
        if phi >= 1:
            return float(H)
        return phi * (1 - phi ** H) / (1 - phi)

    rows, best = [], None
    for W in (4, 6, 8, 10, 12):
        for phi in (0.0, 0.3, 0.5, 0.7, 0.85, 1.0):
            sse, n = 0.0, 0
            for lvl, last, actual_alr in cycles:
                if last - W < 0:
                    continue
                slope_w = (lvl[last] - lvl[last - W]) / W
                fc = lvl[last] + slope_w * damped_factor(phi, H_FAR)
                err = fc - actual_alr
                sse += float((err ** 2).sum()); n += err.size
            if n:
                rmse = np.sqrt(sse / n)
                rows.append((W, phi, rmse))
                if best is None or rmse < best[2]:
                    best = (W, phi, rmse)
    flat = min(r[2] for r in rows if r[1] == 0.0)
    return {"rows": rows, "best": best, "flat_rmse": flat}


def analyze() -> None:
    print(f"=== model-carried sigma(H) — forward projection, "
          f"{len(BACKTEST_YEARS)} cycles {BACKTEST_YEARS} ===")
    print(f"    share-space (pp), 8 Riksdag parties, coverage-calibrated to "
          f"{COVERAGE_TARGET*100:.0f}% (rho={MISS_RHO}, floor_pq={FWD_FLOOR_PQ})")
    s = calibrate_forward()
    print(f"  → MISS_SIGMA_FLOOR     = {s['floor']:.4f}   ({s['floor']*100:.2f}pp at H=0)")
    print(f"  → MISS_SIGMA_VAR_SLOPE = {s['slope']:.3e}   "
          f"→ sigma(14) = {s['s14']*100:.2f}pp")

    print("\n=== coverage at the live constants (per-party 80% interval) ===")
    for H in HORIZONS:
        sig = miss_sigma_for_horizon(H)
        allp = _coverage(H, sig, FWD_FLOOR_PQ, MISS_RHO)
        # Near-threshold coverage, pooled over cycles' parties in 3–8%.
        inside = total = 0
        for year in BACKTEST_YEARS:
            d = _load(year, H)
            sh = _project(d, H, sig, FWD_FLOOR_PQ, MISS_RHO)
            lo, hi = np.percentile(sh, [10, 90], axis=0)
            for k in _near_threshold_parties(d):
                total += 1
                inside += int(lo[k] <= d["actual"][k] <= hi[k])
        nt = f"{inside}/{total}" if total else "n/a"
        print(f"  H={H:<2} sigma {sig*100:.2f}pp: all-party {allp*100:.0f}%  near-threshold {nt}")

    print("\n=== point accuracy (mean forecast vs actual, H=0) ===")
    for year in BACKTEST_YEARS:
        d = _load(year, 0)
        fc = _project(d, 0, miss_sigma_for_horizon(0), FWD_FLOOR_PQ, MISS_RHO).mean(0)
        mae = np.abs(fc - d["actual"])[:N_PARTIES].mean() * 100
        print(f"  {year}: MAE {mae:.2f}pp")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("fit", "all"):
        fit_all(force="--force" in sys.argv)
    if cmd in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
