#!/usr/bin/env python3
"""Guard for the scheduled refresh: is there a poll the live forecast doesn't have?

Compares the number of 2026-cycle polls in the freshly-built polls.parquet
against ``counts.cycle_polls`` recorded in the committed forecast
(web/src/data/forecast.json — the same value served publicly).

**Why a count and not a date.** This used to compare the latest field-end date,
which silently skipped real polls: Novus and Sifo both ended fieldwork on
2026-08-16, so when Novus landed a day after Sifo was published the max date was
unchanged and the guard said "nothing new" — the scheduled job would never have
refit. Same failure when a lagging house publishes a poll whose field-end is
*older* than the newest one already in the fit. A count moves in every one of
those cases; a max-date does not.

The count is over exactly the rows the model fits (cycle start onward, ``n``
present — model.py drops polls without a sample size), so it cannot drift from
what a refit would actually see.

Exit codes (so a shell `if` can branch on it):
  0  — the spine has polls the live forecast lacks → refit/simulate/deploy
  10 — nothing new → stop (don't waste a refit reproducing the same numbers)
  2  — could not determine (missing parquet/json, or a forecast.json predating
       cycle_polls) → caller decides

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


def spine_cycle_polls() -> int | None:
    """Number of cycle polls the model would fit, or None if unavailable."""
    parquet = PROCESSED_DIR / "polls.parquet"
    if not parquet.exists():
        return None
    p = pd.read_parquet(parquet)
    # field-end is the poll's true recency; fall back to pub_date when absent.
    date = p["field_end"].fillna(p["pub_date"])
    # Mirror model.py's filter: cycle window AND a usable sample size.
    cycle = p[(date.dt.date >= CYCLE_2026.start) & p["n"].notna()]
    if cycle.empty:
        return None
    return int(cycle["poll_id"].nunique())


def live_cycle_polls() -> int | None:
    """Cycle-poll count recorded in the published forecast, or None."""
    if not FORECAST_JSON.exists():
        return None
    fc = json.loads(FORECAST_JSON.read_text())
    return (fc.get("counts") or {}).get("cycle_polls")


def main() -> int:
    spine = spine_cycle_polls()
    live = live_cycle_polls()

    if spine is None:
        print("check_new_poll: no cycle polls in polls.parquet (build_polls first?)")
        return 2
    if live is None:
        # Either nothing published yet, or a forecast.json written before
        # cycle_polls existed. Don't guess from dates — say so and let the
        # caller decide; one manual refresh re-records the count.
        print("check_new_poll: live forecast has no counts.cycle_polls on record "
              f"(spine has {spine}) → cannot compare")
        return 2
    if spine > live:
        print(f"check_new_poll: {spine - live} new poll(s) — spine {spine} > live {live} → refresh")
        return 0
    if spine < live:
        # The spine lost polls relative to what was published: an upstream
        # revision or retraction. Not a "new poll", but not normal either.
        print(f"check_new_poll: spine {spine} < live {live} — upstream dropped "
              "poll(s)? → skip, investigate by hand")
        return 10
    print(f"check_new_poll: {spine} cycle polls, same as live → skip")
    return 10


if __name__ == "__main__":
    sys.exit(main())
