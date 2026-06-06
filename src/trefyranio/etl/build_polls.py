"""Build the unified poll spine -> data/processed/polls.parquet.

Phase 1 feed: SwedishPolls (primary). Wikipedia 2026 is wired in later as a
cross-check; SwedishPolls already covers all current pollsters and refreshes
within days, so it carries v1 on its own.

Run:  python -m trefyranio.etl.build_polls
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trefyranio.etl import swedish_polls
from trefyranio.etl.schema import POLLSTER_DISPLAY

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def build(refresh: bool = True) -> pd.DataFrame:
    polls = swedish_polls.load(RAW_DIR, refresh=refresh)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "polls.parquet"
    polls.to_parquet(out, index=False)
    _summary(polls, out)
    return polls


def _summary(polls: pd.DataFrame, out: Path) -> None:
    n_polls = polls["poll_id"].nunique()
    span = (polls["pub_date"].min(), polls["pub_date"].max())
    print(f"wrote {out.relative_to(REPO_ROOT)}  —  {n_polls:,} polls, {len(polls):,} rows")
    print(f"date range: {span[0].date()} … {span[1].date()}")

    recent = polls[polls["pub_date"] >= "2023-01-01"]
    counts = recent.groupby("pollster")["poll_id"].nunique().sort_values(ascending=False)
    print("\npollsters this cycle (since 2023):")
    for name, c in counts.items():
        disp = POLLSTER_DISPLAY.get(name, name)
        tag = f"  [{disp}]" if disp != name else ""
        print(f"  {name:12} {c:>3}{tag}")

    dated = polls.dropna(subset=["pub_date"])
    latest_id = dated.loc[dated["pub_date"].idxmax(), "poll_id"]
    latest = polls[polls["poll_id"] == latest_id]
    meta = latest.iloc[0]
    shares = (
        latest[latest["share"] > 0]
        .set_index("party")["share"]
        .reindex(["S", "M", "SD", "C", "V", "KD", "MP", "L"])
        .dropna()
    )
    print(f"\nlatest poll: {meta['pollster']} field-end {meta['field_end'].date()} n={meta['n']:.0f}")
    print("  " + "  ".join(f"{p} {v*100:.1f}" for p, v in shares.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the trefyranio poll spine.")
    ap.add_argument("--no-refresh", action="store_true", help="use cached raw CSV")
    args = ap.parse_args()
    build(refresh=not args.no_refresh)


if __name__ == "__main__":
    main()
