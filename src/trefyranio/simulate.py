"""Seat & government simulator (Phase 4).

Takes the election-day vote-share posterior (`forecast_samples.npz` from the
model) and, for every draw, runs the real Swedish electoral system — the 4%
gate + modified Sainte-Lague allocator — to produce a posterior over seats.
From the seat draws it computes the product's headline outputs:

* seat distribution per party (+ P in Riksdag),
* threshold survival — P(party < 4%) ("is this the year one gets eliminated?!"),
* bloc seat totals and government-formation probabilities,
* the coalition space — P(majority) for each plausible coalition,
* the kingmaker drama — how a swing party's survival flips the majority.

Government logic is deliberately simple and **config-driven** (alignments shift
election to election): blocs are partitioned, and because 349 is odd one side
always has the edge. We do NOT model coalition bargaining — we report the seat
arithmetic and leave C's choice, the real wildcard, explicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trefyranio.allocator import allocate_national, first_divisor_for_year
from trefyranio.etl.schema import OTHER, RIKSDAG_PARTIES
from trefyranio.model import PARTY_ORDER, project_to_election

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ELECTION_YEAR = 2026
MAJORITY = 175       # of 349
N_SIM = 10000        # predictive Monte-Carlo draws (latent resample × fresh miss)
# All N_SIM draws ship to the coalition-builder client (lazy-fetched as a
# separate file) so its odds match the headline coalition table exactly.
BROWSER_DRAWS = N_SIM
SEAT_DRAW_SEED = 7

# --- Bloc configuration (2026 alignments; edit as politics shifts) ----------
RIGHT = ["M", "KD", "L", "SD"]   # the Tidö constellation
LEFT = ["S", "V", "MP"]          # red-green
CENTRE = ["C"]                   # unaligned kingmaker — refuses SD, wary of V
LEADERS = {"M": "Kristersson", "S": "Andersson"}  # PM candidates by bloc

# Named coalitions to score for outright majority.
COALITIONS = {
    "Tidö (M+KD+L+SD)": ["M", "KD", "L", "SD"],
    "Red-green+C (S+V+MP+C)": ["S", "V", "MP", "C"],
    "Red-green (S+V+MP)": ["S", "V", "MP"],
    "Centre-left (S+MP+C)": ["S", "MP", "C"],
    "Right ex-SD (M+KD+L+C)": ["M", "KD", "L", "C"],
}
# Swing parties whose 4% survival can flip a bloc majority.
SWING = ["L", "KD", "MP", "C"]
MIN_COND_DRAWS = 20  # don't report a conditional computed on fewer draws


def simulate_seats(shares: np.ndarray, parties: list[str], year: int = ELECTION_YEAR) -> np.ndarray:
    """(draws, K) vote shares → (draws, 8) seats for the Riksdag parties.

    Each draw runs the real allocator: 4% gate (denominator includes Övr), the
    other bucket excluded from seats, year's first divisor."""
    fd = first_divisor_for_year(year)
    idx = {p: i for i, p in enumerate(parties)}
    out = np.zeros((shares.shape[0], len(RIKSDAG_PARTIES)), dtype=int)
    for d in range(shares.shape[0]):
        votes = {p: float(shares[d, idx[p]]) for p in parties}
        res = allocate_national(votes, first_divisor=fd, ignore_parties=frozenset({OTHER}))
        for j, p in enumerate(RIKSDAG_PARTIES):
            out[d, j] = res.seats.get(p, 0)
    return out


def seat_distribution(seats: np.ndarray) -> pd.DataFrame:
    """Per-party seat summary + threshold survival."""
    rows = []
    for j, p in enumerate(RIKSDAG_PARTIES):
        col = seats[:, j]
        rows.append({
            "party": p,
            "mean_seats": col.mean(),
            "lo": int(np.quantile(col, 0.1)),
            "hi": int(np.quantile(col, 0.9)),
            "p_in_riksdag": float((col > 0).mean()),
            "p_below_4pct": float((col == 0).mean()),  # no seats <=> under 4%
        })
    return pd.DataFrame(rows)


def _bloc_seats(seats: np.ndarray, bloc: list[str]) -> np.ndarray:
    idx = [RIKSDAG_PARTIES.index(p) for p in bloc]
    return seats[:, idx].sum(axis=1)


def coalition_table(seats: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, parties in COALITIONS.items():
        tot = _bloc_seats(seats, parties)
        rows.append({
            "coalition": name,
            "parties": "+".join(parties),
            "p_majority": float((tot >= MAJORITY).mean()),
            "mean_seats": float(tot.mean()),
            "lo": int(np.quantile(tot, 0.1)),
            "hi": int(np.quantile(tot, 0.9)),
        })
    return pd.DataFrame(rows).sort_values("p_majority", ascending=False)


def government_outlook(seats: np.ndarray) -> dict:
    """Three-way, mutually-exclusive government decomposition + kingmaker drama."""
    right = _bloc_seats(seats, RIGHT)
    left = _bloc_seats(seats, LEFT)
    # Right+Left+Centre = 349 (all eight partitioned), 349 odd.
    p_right_maj = float((right >= MAJORITY).mean())
    p_left_maj = float((left >= MAJORITY).mean())
    p_kingmaker = float(((right < MAJORITY) & (left < MAJORITY)).mean())

    # How much does each swing party's survival move the Tidö majority?
    drama = {}
    for p in SWING:
        col = seats[:, RIKSDAG_PARTIES.index(p)]
        survives, out = col > 0, col == 0
        entry = {"p_below_4pct": float(out.mean())}
        # Only report a conditional backed by enough draws (rare survival/death
        # otherwise yields a noisy, misleading fraction).
        if survives.sum() >= MIN_COND_DRAWS:
            entry["p_right_maj_if_survives"] = float((right[survives] >= MAJORITY).mean())
        if out.sum() >= MIN_COND_DRAWS:
            entry["p_right_maj_if_out"] = float((right[out] >= MAJORITY).mean())
        drama[p] = entry

    return {
        "p_right_majority": p_right_maj,        # Tidö keeps power (PM Kristersson)
        "p_left_majority": p_left_maj,          # red-green wins outright (PM Andersson)
        "p_centre_kingmaker": p_kingmaker,      # C decides — the wildcard
        "blocs": {"right": RIGHT, "left": LEFT, "centre": CENTRE},
        "leaders": LEADERS,
        "swing_drama": drama,
    }


def build() -> None:
    # Predictive Monte Carlo: resample the posterior election-day latent (with
    # replacement) and add a fresh polling-miss draw to each, to N_SIM draws.
    # Most election-day variance is the miss term, so this is a proper predictive
    # ensemble, not just the ~600 NUTS draws.
    npz = np.load(PROCESSED_DIR / "forecast_samples.npz", allow_pickle=True)
    # Model-carried error: resample the LAST-POLL-WEEK latent + drift posterior and
    # project each forward to election day with a fresh forward-innovation draw, to
    # N_SIM draws — a proper predictive ensemble, not just the ~600 NUTS draws.
    last_alr = npz["last_alr"]                               # (n_post, KM1)
    drift = npz["drift"]                                     # (n_post, KM1)
    horizon = float(npz["horizon_weeks"])
    shift = npz["industry_shift"] if "industry_shift" in npz else None
    fundamentals = npz["fundamentals"] if "fundamentals" in npz else None
    fund_w = float(npz["fund_w"]) if "fund_w" in npz else 0.0
    rng = np.random.default_rng(SEAT_DRAW_SEED)
    idx = rng.integers(0, last_alr.shape[0], N_SIM)
    shares = project_to_election(last_alr[idx], drift[idx], horizon,
                                 industry_shift=shift, fundamentals=fundamentals,
                                 fund_w=fund_w, seed=SEAT_DRAW_SEED)
    seats = simulate_seats(shares, PARTY_ORDER)             # (N_SIM, 8)

    seat_df = seat_distribution(seats)
    coal_df = coalition_table(seats)
    gov = government_outlook(seats)
    gov["n_sim"] = N_SIM

    seat_df.to_parquet(PROCESSED_DIR / "seat_forecast.parquet", index=False)
    coal_df.to_parquet(PROCESSED_DIR / "coalition_forecast.parquet", index=False)
    (PROCESSED_DIR / "government_forecast.json").write_text(json.dumps(gov, indent=2, ensure_ascii=False))
    # Ship a subset to the browser (coalition builder); full set drives the stats.
    np.savez(PROCESSED_DIR / "seat_draws.npz", seats=seats[:BROWSER_DRAWS],
             parties=np.array(RIKSDAG_PARTIES), n_sim=N_SIM)

    _report(seat_df, coal_df, gov, seats)


def _report(seat_df, coal_df, gov, seats) -> None:
    print(f"=== seat forecast ({ELECTION_YEAR}, {seats.shape[0]} draws) ===")
    for _, r in seat_df.sort_values("mean_seats", ascending=False).iterrows():
        flag = f"   P(<4% → OUT) {r['p_below_4pct']*100:3.0f}%" if r["p_below_4pct"] > 0.005 else ""
        print(f"  {r['party']:3} {r['mean_seats']:5.1f} seats  [{r['lo']:3d}–{r['hi']:3d}]{flag}")
    print(f"  total seats/draw: {seats.sum(axis=1).mean():.0f}")

    print("\n=== government outlook ===")
    print(f"  Tidö/right majority (PM {gov['leaders']['M']}):  {gov['p_right_majority']*100:4.0f}%")
    print(f"  Red-green majority  (PM {gov['leaders']['S']}):  {gov['p_left_majority']*100:4.0f}%")
    print(f"  Centern kingmaker (no bloc ≥175):       {gov['p_centre_kingmaker']*100:4.0f}%")

    print("\n=== coalition majority chances ===")
    for _, r in coal_df.iterrows():
        print(f"  {r['coalition']:26} {r['p_majority']*100:4.0f}%   {r['mean_seats']:5.1f} [{r['lo']}–{r['hi']}]")

    print("\n=== kingmaker drama: Tidö majority given a swing party's fate ===")
    for p, e in gov["swing_drama"].items():
        parts = [f"P(<4%)={e['p_below_4pct']*100:3.0f}%"]
        if "p_right_maj_if_survives" in e:
            parts.append(f"Tidö-maj if {p} in {e['p_right_maj_if_survives']*100:3.0f}%")
        if "p_right_maj_if_out" in e:
            parts.append(f"if {p} out {e['p_right_maj_if_out']*100:3.0f}%")
        print(f"  {p:3} " + "  ".join(parts))


if __name__ == "__main__":
    build()
