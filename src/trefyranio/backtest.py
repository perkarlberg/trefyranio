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
    PARTY_ORDER,
    PROCESSED_DIR,
    _alr,
    _softmax_with_ref,
    cycle_for,
    fit,
    prepare,
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
            data = prepare(polls, results, cycle, as_of=as_of)
            post = fit(data, warmup=warmup, samples=samples, seed=0, num_chains=2)
            trend, election_alr_trend = summarize(post, data)
            tw = (trend.pivot(index="date", columns="party", values="mean")
                  .reindex(columns=PARTY_ORDER).to_numpy())
            np.savez(
                out,
                election_alr_trend=election_alr_trend,
                trend_mean=tw,
                actual=actual_shares(results, year),
                election_week=int(data.election_week),
                last_week=int(data.week.max()),
            )
            print(f"  fit {year} H={H}: {len(data.counts)} polls "
                  f"(last wk {int(data.week.max())}/{data.election_week}) → {out.name}",
                  flush=True)


def _load(year: int, H: int):
    return np.load(_path(year, H), allow_pickle=True)


def miss_from_moments(errs: np.ndarray, sd_trend: np.ndarray) -> float:
    """Added-variance moment match: size the miss so total predictive variance
    (trend variance + miss²) equals realized squared error. Clamped at 0."""
    return float(np.sqrt(max(0.0, (errs ** 2).mean() - (sd_trend ** 2).mean())))


def coverage(errs: np.ndarray, sd_trend: np.ndarray, miss: float) -> float:
    total = np.sqrt(sd_trend ** 2 + miss ** 2)
    return float((np.abs(errs / total) < Z80).mean())


def calibrate_coverage(errs: np.ndarray, sd_trend: np.ndarray,
                       target: float = COVERAGE_TARGET) -> float:
    """Smallest MISS_SIGMA whose 80% interval reaches the coverage target —
    robust to heavy tails (a few big misses won't over-widen the whole band)."""
    grid = np.linspace(0.0, 0.1, 201)  # share-space, 0.05pp resolution
    covs = np.array([coverage(errs, sd_trend, m) for m in grid])
    reached = grid[covs >= target]
    return float(reached[0]) if len(reached) else float(grid[-1])


def _pool_horizon(H: int):
    """Pool share-space errors + posterior spreads across cycles at horizon H,
    over the 8 Riksdag parties."""
    errs, sd_trend = [], []
    for year in BACKTEST_YEARS:
        d = _load(year, H)
        shares = _softmax_with_ref(d["election_alr_trend"])      # (S, K)
        errs.append((shares.mean(0) - d["actual"])[:N_PARTIES])
        sd_trend.append(shares.std(0)[:N_PARTIES])
    return np.concatenate(errs), np.concatenate(sd_trend)


def calibrate_sigma_curve() -> dict:
    """Coverage-calibrate sigma at H=0 and H=14, derive the variance curve."""
    out = {}
    for H in HORIZONS:
        errs, sd_trend = _pool_horizon(H)
        out[H] = {
            "mm": miss_from_moments(errs, sd_trend),
            "cov": calibrate_coverage(errs, sd_trend),
            "rmse": float(np.sqrt((errs ** 2).mean())),
            "cov_at_mm": coverage(errs, sd_trend, miss_from_moments(errs, sd_trend)),
        }
    s0 = max(out[0]["cov"], 1e-4)
    s14 = max(out[H_FAR]["cov"], s0)              # monotone in horizon
    out["floor"] = s0
    out["slope"] = (s14 ** 2 - s0 ** 2) / H_FAR
    return out


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
    print(f"=== MISS_SIGMA sigma(H) curve — converged model, "
          f"{len(BACKTEST_YEARS)} cycles {BACKTEST_YEARS} ===")
    print("    share-space (pp), 8 Riksdag parties, coverage-calibrated to "
          f"{COVERAGE_TARGET*100:.0f}%")
    s = calibrate_sigma_curve()
    for H in HORIZONS:
        h = s[H]
        print(f"  H={H:<2}: moment-match {h['mm']*100:5.2f}pp (cov {h['cov_at_mm']*100:.0f}%) | "
              f"coverage-cal {h['cov']*100:5.2f}pp | realized RMSE {h['rmse']*100:5.2f}pp")
    print(f"  → MISS_SIGMA_FLOOR     = {s['floor']:.4f}   ({s['floor']*100:.2f}pp at H=0)")
    print(f"  → MISS_SIGMA_VAR_SLOPE = {s['slope']:.3e}   "
          f"→ sigma(14) = {np.sqrt(s['floor']**2 + s['slope']*H_FAR)*100:.2f}pp")

    print("\n=== point accuracy (mean forecast vs actual, H=0) ===")
    for year in BACKTEST_YEARS:
        d = _load(year, 0)
        fc = _softmax_with_ref(d["election_alr_trend"]).mean(0)
        mae = np.abs(fc - d["actual"])[:N_PARTIES].mean() * 100
        print(f"  {year}: MAE {mae:.2f}pp")

    print("\n=== damped recent-momentum @H=14 (ALR RMSE, lower=better) ===")
    m = momentum_test()
    W, phi, rmse = m["best"]
    print(f"  flat carry (phi=0): {m['flat_rmse']:.4f}")
    print(f"  best: phi={phi:.2f}, W={W} → {rmse:.4f}  "
          f"({(m['flat_rmse']-rmse)/m['flat_rmse']*100:.1f}% better than flat)")
    if phi == 0.0 or (m["flat_rmse"] - rmse) / m["flat_rmse"] < 0.05:
        print("  → effect is noise-level; live model ships phi=0 (no momentum term, "
              "per-party drift only)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("fit", "all"):
        fit_all(force="--force" in sys.argv)
    if cmd in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
