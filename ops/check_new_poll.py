#!/usr/bin/env python3
"""Guard for the scheduled refresh: is there a poll newer than what's live?

Compares the latest 2026-cycle poll in the freshly-built polls.parquet against
the ``latest_poll.date`` recorded in the committed forecast (web/src/data/
forecast.json — the same value served publicly).

Exit codes (so a shell `if` can branch on it):
  0  — a newer poll exists → the caller should refit/simulate/deploy
  10 — no new poll → the caller should stop (don't waste ~10 min refitting)
  2  — could not determine (missing parquet/json) → caller decides

Run build_polls first so polls.parquet reflects the latest fetch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from trefyranio.model import CYCLE_2026, PROCESSED_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
FORECAST_JSON = REPO_ROOT / "web" / "src" / "data" / "forecast.json"


def latest_cycle_poll_date() -> pd.Timestamp | None:
    parquet = PROCESSED_DIR / "polls.parquet"
    if not parquet.exists():
        return None
    p = pd.read_parquet(parquet)
    # field-end is the poll's true recency; fall back to pub_date when absent.
    d = p["field_end"].fillna(p["pub_date"])
    cycle = d.dt.date >= CYCLE_2026.start
    if not cycle.any():
        return None
    return pd.Timestamp(d[cycle].max())


def live_forecast_date() -> pd.Timestamp | None:
    if not FORECAST_JSON.exists():
        return None
    fc = json.loads(FORECAST_JSON.read_text())
    date = (fc.get("latest_poll") or {}).get("date")
    return pd.Timestamp(date) if date else None


def main() -> int:
    latest = latest_cycle_poll_date()
    live = live_forecast_date()
    if latest is None:
        print("check_new_poll: no cycle polls in polls.parquet (build_polls first?)")
        return 2
    if live is None:
        print(f"check_new_poll: no live forecast date on record; latest poll {latest.date()}")
        return 0  # treat as "refresh" — nothing published yet
    if latest.date() > live.date():
        print(f"check_new_poll: NEW poll {latest.date()} > live {live.date()} → refresh")
        return 0
    print(f"check_new_poll: latest poll {latest.date()} not newer than live {live.date()} → skip")
    return 10


if __name__ == "__main__":
    sys.exit(main())
