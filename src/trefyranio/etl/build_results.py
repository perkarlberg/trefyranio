"""Build the official-results spine -> data/processed/results_*.parquet.

Pulls national + valkrets vote results, turnout, and actual seats from SCB
(1973-2022). As an end-to-end check it runs the seat allocator on the fetched
2022 national votes and confirms it reproduces the actual seat distribution —
tying the new results data to the validated allocator.

Run:  python -m trefyranio.etl.build_results
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trefyranio.allocator import allocate_national, first_divisor_for_year
from trefyranio.etl import scb_results
from trefyranio.etl.schema import OTHER

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def build() -> dict[str, pd.DataFrame]:
    national, meta = scb_results.fetch_national()
    valkrets = scb_results.fetch_valkrets()
    seats = scb_results.fetch_seats()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results_national": national,
        "results_meta": meta,
        "results_valkrets": valkrets,
        "seats_actual": seats,
    }
    for name, df in outputs.items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    _summary(outputs)
    _crosscheck_allocator(national, seats)
    return outputs


def _summary(outputs: dict[str, pd.DataFrame]) -> None:
    for name, df in outputs.items():
        print(f"wrote {name}.parquet  —  {len(df):,} rows, "
              f"years {df['election_year'].min()}-{df['election_year'].max()}")
    nat = outputs["results_national"]
    latest = nat[nat["election_year"] == nat["election_year"].max()]
    print(f"\n2022 national vote share:")
    for _, r in latest[latest["party"] != "Övr"].iterrows():
        print(f"  {r['party']:3} {r['share']*100:5.2f}%  ({r['votes']:>9,} votes)")
    m = outputs["results_meta"]
    m22 = m[m["election_year"] == m["election_year"].max()].iloc[0]
    print(f"  turnout {m22['turnout']*100:.2f}%  (eligible {m22['eligible']:,})")


def _crosscheck_allocator(national: pd.DataFrame, seats: pd.DataFrame) -> None:
    """For every election, the allocator run on the fetched national votes
    (with that year's first divisor) must equal the actual national seats."""
    nat_seats = seats[seats["region_code"] == "VR00"]
    print("\nallocator cross-check (national seats, all elections):")
    ok = 0
    for year in sorted(national["election_year"].unique()):
        ny = national[national["election_year"] == year]
        votes = dict(zip(ny["party"], ny["votes"]))
        # 1973 elected a 350-seat Riksdag (cut to 349 in 1976 after the tie).
        n_seats = 350 if year == 1973 else 349
        predicted = {
            p: s
            for p, s in allocate_national(
                votes,
                n_seats=n_seats,
                first_divisor=first_divisor_for_year(year),
                ignore_parties=frozenset({OTHER}),
            ).seats.items()
            if s > 0
        }
        sy = nat_seats[nat_seats["election_year"] == year]
        actual = dict(zip(sy["party"], sy["seats"]))
        match = predicted == actual
        ok += match
        if not match:
            diff = {p: (predicted.get(p, 0), actual.get(p, 0))
                    for p in set(predicted) | set(actual)
                    if predicted.get(p, 0) != actual.get(p, 0)}
            print(f"  {year}: MISMATCH  pred/actual {diff}")
    print(f"  {ok}/{national['election_year'].nunique()} elections reproduced exactly")


if __name__ == "__main__":
    build()
