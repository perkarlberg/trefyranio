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
    GOVERNMENTS,
    MISS_RHO,
    PARTY_ORDER,
    PROCESSED_DIR,
    _alr,
    _softmax_with_ref,
    cost_of_ruling,
    cycle_for,
    fit,
    fund_weight,
    fundamentals_prior,
    miss_sigma_for_horizon,
    prepare,
    project_to_election,
    summarize,
)

BACKTEST_YEARS = [2010, 2014, 2018, 2022]   # current party system, dense polling
EARLY_YEARS = [2002, 2006]                  # pre-SD-in-riksdag; Phase-3 A/B only
ALL_YEARS = EARLY_YEARS + BACKTEST_YEARS
HORIZONS = [0, 14, 30]                       # election-eve, live ~14w gap, early-cycle
BT_DIR = PROCESSED_DIR / "backtests"
N_PARTIES = 8                   # the eight Riksdag parties (exclude Övr)
Z80 = 1.2816                    # 10–90% half-width in sd units
H_FAR = 14                      # live-forecast horizon (weeks from last poll)
COVERAGE_TARGET = 0.85          # mildly conservative interval coverage
# Blocs for the within-bloc miss-correlation estimate (mirror model._MISS_GROUP
# and simulate's blocs). Only the multi-party blocs identify rho.
_RHO_BLOCS = {"RIGHT": ["M", "SD", "KD", "L"], "LEFT": ["S", "V", "MP"]}


def actual_shares(results: pd.DataFrame, year: int) -> np.ndarray:
    r = results[results["election_year"] == year].set_index("party")["share"]
    v = np.array([r.get(p, 1e-6) for p in PARTY_ORDER])
    return v / v.sum()


def _path(year: int, H: int):
    return BT_DIR / f"conv_{year}_h{H}.npz"


def fit_all(warmup: int = 500, samples: int = 500, force: bool = False,
            years: list[int] = BACKTEST_YEARS) -> None:
    """Refit each cycle at each horizon with the converged model; cache to disk.
    Skips files that already exist unless ``force``."""
    polls = pd.read_parquet(PROCESSED_DIR / "polls.parquet")
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    BT_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
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
              years: list[int] = BACKTEST_YEARS,
              parties=range(N_PARTIES), lo_hi=(10, 90)) -> float:
    """Pooled 80%-interval coverage of the projected shares vs actual, over the
    given parties × cycles at horizon H."""
    inside = total = 0
    for year in years:
        d = _load(year, H)
        sh = _project(d, H, sigma, floor_pq, rho)
        lo, hi = np.percentile(sh, lo_hi, axis=0)
        for k in parties:
            total += 1
            inside += int(lo[k] <= d["actual"][k] <= hi[k])
    return inside / total if total else float("nan")


def calibrate_forward(years: list[int] = BACKTEST_YEARS,
                      rho: float = MISS_RHO, floor_pq: float = FWD_FLOOR_PQ) -> dict:
    """Find the sigma(H) curve whose forward projection reaches the coverage
    target at H=0 and H=14, pooled over the 8 Riksdag parties."""
    grid = np.linspace(0.005, 0.06, 111)

    def smallest(H: int) -> float:
        for s in grid:
            if _coverage(H, s, floor_pq, rho, years) >= COVERAGE_TARGET:
                return float(s)
        return float(grid[-1])

    floor = smallest(0)
    s14 = max(smallest(H_FAR), floor)             # monotone in horizon
    return {"floor": floor, "s14": s14, "slope": (s14 ** 2 - floor ** 2) / H_FAR}


def calibrate_rho(years: list[int] = BACKTEST_YEARS, H: int = H_FAR,
                  floor_pq: float = FWD_FLOOR_PQ) -> dict:
    """Estimate the within-bloc miss correlation from BLOC-TOTAL coverage. With
    iid misses the bloc-total interval is too narrow; rho widens it. Smallest rho
    whose projected bloc-total 80% interval covers the realized bloc totals at the
    target, pooled over the multi-party blocs (RIGHT, LEFT) × cycles."""
    bloc_idx = {b: [PARTY_ORDER.index(p) for p in ps] for b, ps in _RHO_BLOCS.items()}
    sigma = miss_sigma_for_horizon(H)

    def bloc_coverage(rho: float) -> float:
        inside = total = 0
        for year in years:
            d = _load(year, H)
            sh = _project(d, H, sigma, floor_pq, rho)
            for idx in bloc_idx.values():
                lo, hi = np.percentile(sh[:, idx].sum(1), [10, 90])
                act = float(np.asarray(d["actual"])[idx].sum())
                total += 1
                inside += int(lo <= act <= hi)
        return inside / total if total else float("nan")

    grid = np.linspace(0.0, 0.8, 33)
    covs = [(float(r), bloc_coverage(float(r))) for r in grid]
    reached = [r for r, c in covs if c >= COVERAGE_TARGET]
    return {"rho": reached[0] if reached else float(grid[-1]),
            "iid_coverage": bloc_coverage(0.0), "curve": covs}


def _near_threshold_parties(d, lo=0.03, hi=0.08):
    return [k for k in range(N_PARTIES) if lo <= d["actual"][k] <= hi]


def calibrate_fundamentals(years: list[int] = BACKTEST_YEARS,
                           horizons=(14, 30)) -> dict:
    """Does the cost-of-ruling prior improve the point forecast? Grid the blend
    slope; for each horizon, apply the (leave-one-out) fundamentals blend to the
    cached projection and pool the mean-forecast MAE over the 8 Riksdag parties ×
    cycles. The gate: ship the slope that lowers MAE at H=14; report H=30 (where
    fundamentals should help more). slope 0 = polls only."""
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    funds = {}
    for year in years:
        delta = cost_of_ruling(results, exclude_year=year)          # leave-one-out
        funds[year] = fundamentals_prior(actual_shares(results, year - 4),
                                         GOVERNMENTS.get(year, set()), delta)

    def mae(H: int, slope: float) -> float:
        fw = min(0.5, slope * H)
        errs = []
        for year in years:
            d = _load(year, H)
            sh = project_to_election(d["last_alr"], d["drift"], H,
                                     sigma_share=miss_sigma_for_horizon(H),
                                     floor_pq=FWD_FLOOR_PQ, rho=MISS_RHO,
                                     fundamentals=funds[year], fund_w=fw)
            errs.append(np.abs(sh.mean(0) - d["actual"])[:N_PARTIES])
        return float(np.concatenate(errs).mean())

    grid = np.linspace(0.0, 0.03, 16)        # slope per week (fund_w = slope·H)
    rows = {H: [(float(s), mae(H, float(s))) for s in grid] for H in horizons}
    best14 = min(rows[14], key=lambda r: r[1])
    base14 = rows[14][0][1]                  # slope 0 (polls only)
    return {"rows": rows, "best_slope_14": best14[0], "mae14_best": best14[1],
            "mae14_base": base14, "grid": grid}


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


def ab() -> None:
    """Phase-3 A/B: does adding 2002 & 2006 (pre-SD-in-riksdag) change the
    calibration? Three arms. Every arm is EVALUATED on the 2010+ cycles (the
    regime the 2026 forecast lives in) so the comparison is on the relevant
    population, not the training set."""
    arms = {
        "(i) 4-cycle [2010+]":        {"sigma_years": BACKTEST_YEARS, "rho_years": BACKTEST_YEARS},
        "(ii) 6-cycle [all]":         {"sigma_years": ALL_YEARS,      "rho_years": ALL_YEARS},
        "(iii) 6-cycle sigma, rho 2010+": {"sigma_years": ALL_YEARS,  "rho_years": BACKTEST_YEARS},
    }
    print("=== Phase-3 A/B: 2002/2006 regime-mismatch test ===")
    print("    calibrate on the arm's cycles; EVALUATE coverage on 2010+ (the 2026 regime)\n")
    for name, a in arms.items():
        s = calibrate_forward(a["sigma_years"])
        r = calibrate_rho(a["rho_years"])
        # Evaluate THIS arm's constants on the 2010+ cycles.
        cov0 = _coverage(0, s["floor"], FWD_FLOOR_PQ, r["rho"], BACKTEST_YEARS)
        cov14 = _coverage(H_FAR, s["s14"], FWD_FLOOR_PQ, r["rho"], BACKTEST_YEARS)
        print(f"{name}")
        print(f"    floor {s['floor']*100:.2f}pp  sigma(14) {s['s14']*100:.2f}pp  "
              f"rho {r['rho']:.2f} (iid bloc-cov {r['iid_coverage']*100:.0f}%)")
        print(f"    → 2010+ coverage: H=0 {cov0*100:.0f}%  H=14 {cov14*100:.0f}%\n")


def fundamentals() -> None:
    """Phase-6 gate: report cost-of-ruling MAE with vs without the prior."""
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    print("=== cost-of-ruling fundamentals — point-forecast MAE (8 parties, 4 cycles) ===")
    print(f"    pooled cost of ruling: {cost_of_ruling(results)*100:+.2f}pp "
          f"(leave-one-out per cycle in the gate)\n")
    c = calibrate_fundamentals()
    impr = {}
    for H in (14, 30):
        base = c["rows"][H][0][1]
        best = min(c["rows"][H], key=lambda r: r[1])
        impr[H] = (base - best[1]) / base
        print(f"  H={H:<2}: polls-only MAE {base*100:.3f}pp | "
              f"best slope {best[0]:.4f} → MAE {best[1]*100:.3f}pp ({impr[H]*100:+.1f}%)")
    # Gate: a genuine fundamentals signal must (a) beat polls at H=14 by a margin
    # above noise AND (b) be corroborated by a LARGER gain further out (H=30,
    # where polls carry less). Otherwise the H=14 'best' is grid-argmin overfit.
    real = impr[14] >= 0.03 and impr[30] >= impr[14]
    slope = c["best_slope_14"] if real else 0.0
    print(f"\n  → ship FUND_WEIGHT_PER_WEEK = {slope:.4f}", end="")
    if real:
        print(f" (fund_w@15w = {min(0.5, slope*15):.2f})")
    else:
        print("  — fundamentals are noise-level (help <3% at H=14, no larger gain at\n"
              "    H=30): dense Swedish polling already prices in the cost of ruling.\n"
              "    Keep the machinery, ship weight 0 (cf. the rejected momentum term).")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("fit", "all"):
        fit_all(force="--force" in sys.argv)
    if cmd == "fit-early":
        fit_all(force="--force" in sys.argv, years=EARLY_YEARS)
    if cmd in ("analyze", "all"):
        analyze()
    if cmd == "ab":
        ab()
    if cmd == "fundamentals":
        fundamentals()


if __name__ == "__main__":
    main()
