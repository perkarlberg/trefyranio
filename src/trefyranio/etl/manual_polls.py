"""Hand-entered polls that upstream hasn't published yet.

SwedishPolls is the primary feed, but it lags publication by a few days. When a
poll is out in the press and we want the forecast current *now*, its numbers go
into ``data/manual_polls.csv`` — a tracked file (unlike ``data/raw/``) using the
exact upstream column layout, so the same ``to_tidy`` code path handles it.

The merge is **upstream-wins**: once SwedishPolls publishes the same poll, the
manual row is dropped automatically and the entry can be deleted from the CSV at
leisure. Identity for that comparison is ``(house, collectPeriodTo)`` — a house
does not publish two polls with the same fieldwork end date — deliberately *not*
the poll_id hash, which would differ whenever upstream's publication date or
sample size disagrees with what we typed in.

Keep entries short-lived: they are unreviewed, single-source data. Cite the
source in the ``note`` column when adding one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trefyranio.etl.schema import NAMED_PARTIES

SOURCE = "manual"

# The upstream SwedishPolls column layout, which the supplement mirrors exactly.
RAW_COLUMNS = [
    "PublYearMonth", "Company", "M", "L", "C", "KD", "S", "V", "MP", "SD",
    "FI", "Uncertain", "n", "PublDate", "collectPeriodFrom", "collectPeriodTo",
    "approxPeriod", "house",
]
# Extra column, ours alone — provenance for the human, stripped before merging.
NOTE_COLUMN = "note"

# Fields a hand-entered poll must always carry (upstream may omit them on
# decades-old rows; a poll we type in today has no such excuse).
REQUIRED = ["house", "PublDate", "collectPeriodTo"]

_DEDUPE_KEY = ["house", "collectPeriodTo"]


def load_empty() -> pd.DataFrame:
    """An empty supplement — the no-manual-polls case."""
    return pd.DataFrame(columns=RAW_COLUMNS)


def load(path: Path) -> pd.DataFrame:
    """Read and validate the supplement CSV; empty frame if it doesn't exist."""
    if not path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = pd.read_csv(path, comment="#")
    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)} — it must mirror the "
            "SwedishPolls layout exactly"
        )
    for col in REQUIRED:
        if df[col].isna().any():
            raise ValueError(f"{path.name}: column {col!r} must be set on every row")

    # Percent columns must look like percent, not fractions — a share typed as
    # 0.302 instead of 30.2 would sail through the model as a 0.3% party.
    named = df[NAMED_PARTIES].apply(pd.to_numeric, errors="coerce")
    total = named.sum(axis=1, skipna=True)
    off = df[(total < 90) | (total > 101)]
    if not off.empty:
        raise ValueError(
            f"{path.name}: named-party shares must be percentages summing to ~100; "
            f"row(s) {list(off.index)} sum to {[round(t, 1) for t in total[off.index]]}"
        )

    return df[RAW_COLUMNS]


def merge(upstream: pd.DataFrame, supplement: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Append supplement rows upstream hasn't published yet.

    Returns ``(merged, added, superseded)`` where the two lists hold
    ``"<house> <field-end>"`` labels for logging.
    """
    if supplement.empty:
        upstream = upstream.copy()
        upstream["_source"] = "SwedishPolls"
        return upstream, [], []

    def key(df: pd.DataFrame) -> pd.Series:
        end = pd.to_datetime(df["collectPeriodTo"], errors="coerce")
        return df["house"].astype(str) + "|" + end.dt.strftime("%Y-%m-%d").fillna("NaT")

    up_keys = set(key(upstream))
    sup_key = key(supplement)
    is_new = ~sup_key.isin(up_keys)

    def labels(k: pd.Series) -> list[str]:
        return [s.replace("|", " ") for s in k]

    added = labels(sup_key[is_new])
    superseded = labels(sup_key[~is_new])

    upstream = upstream.copy()
    upstream["_source"] = "SwedishPolls"
    fresh = supplement[is_new].copy()
    fresh["_source"] = SOURCE

    merged = pd.concat([upstream, fresh], ignore_index=True)
    return merged, added, superseded
