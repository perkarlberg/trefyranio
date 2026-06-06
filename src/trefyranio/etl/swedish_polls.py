"""Ingest the SwedishPolls dataset — the primary poll feed.

Source: https://github.com/MansMeg/SwedishPolls (CC0). A single CSV of every
Swedish national vote-intention poll back to 1967, refreshed within days of
publication, covering all current pollsters. We download it, melt it to the
tidy long schema, and let the modeling layer handle normalization.

Upstream columns (as of 2026):
    PublYearMonth, Company, M, L, C, KD, S, V, MP, SD, FI, Uncertain, n,
    PublDate, collectPeriodFrom, collectPeriodTo, approxPeriod, house
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import requests

from trefyranio.etl.schema import NAMED_PARTIES, OTHER, TIDY_COLUMNS, validate_polls

CSV_URL = "https://raw.githubusercontent.com/MansMeg/SwedishPolls/master/Data/Polls.csv"
SOURCE = "SwedishPolls"
_UA = {"User-Agent": "trefyranio/0.1 (election-model research)"}


def download(raw_dir: Path) -> Path:
    """Fetch the raw CSV into ``raw_dir`` and return its path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "swedish_polls.csv"
    resp = requests.get(CSV_URL, timeout=60, headers=_UA)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _poll_ids(raw: pd.DataFrame) -> pd.Series:
    """Stable, collision-proof ids for every poll.

    The identity key includes ``PublYearMonth`` — the one field always present
    even for old polls whose exact dates are missing (without it, decades of
    NaT-dated monthly polls hash to the same id). Any residual exact-key
    duplicates are disambiguated with a deterministic per-key counter.
    """
    # Build the key by stringifying each cell with Python str() (NaT/NaN ->
    # "NaT"/"nan"), then joining. Series "+" concatenation propagates NA in
    # pandas 3.0, which would collapse undated polls to one key.
    id_cols = ["house", "PublYearMonth", "PublDate", "collectPeriodFrom",
               "collectPeriodTo", "n"]
    key = raw[id_cols].apply(lambda c: c.map(str)).agg("|".join, axis=1)
    base = key.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:12])
    dup = raw.groupby(key, sort=False).cumcount()
    suffix = dup.map(lambda i: "" if i == 0 else f"_{i}")
    return SOURCE[:2].lower() + "_" + base + suffix


def to_tidy(csv_path: Path) -> pd.DataFrame:
    """Transform the raw SwedishPolls CSV into the tidy long poll table."""
    raw = pd.read_csv(csv_path)

    for col in ("PublDate", "collectPeriodFrom", "collectPeriodTo"):
        raw[col] = pd.to_datetime(raw[col], errors="coerce")

    # Percent -> fraction for the named parties and the undecided bucket.
    for p in NAMED_PARTIES:
        raw[p] = pd.to_numeric(raw[p], errors="coerce") / 100.0
    raw["uncertain"] = pd.to_numeric(raw["Uncertain"], errors="coerce") / 100.0

    # "Other" = residual of the decided-voter simplex (named shares sum to ~0.98;
    # the gap is small parties below the named set). Clamp to >= 0.
    named_sum = raw[NAMED_PARTIES].sum(axis=1, skipna=True)
    raw[OTHER] = (1.0 - named_sum).clip(lower=0.0)

    raw["poll_id"] = _poll_ids(raw)
    # `house` is the canonical pollster; `Company` names the commissioner when
    # it differs (e.g. house="Novus", Company="TV4").
    raw["pollster"] = raw["house"].astype("string")
    raw["commissioner"] = raw["Company"].where(
        raw["Company"].astype("string") != raw["house"].astype("string")
    )

    long = raw.melt(
        id_vars=[
            "poll_id", "pollster", "commissioner",
            "PublDate", "collectPeriodFrom", "collectPeriodTo", "n", "uncertain",
        ],
        value_vars=NAMED_PARTIES + [OTHER],
        var_name="party",
        value_name="share",
    ).rename(
        columns={
            "PublDate": "pub_date",
            "collectPeriodFrom": "field_start",
            "collectPeriodTo": "field_end",
        }
    )
    long["source"] = SOURCE
    long["share"] = long["share"].fillna(0.0)

    long = long[TIDY_COLUMNS].sort_values(["pub_date", "poll_id", "party"])
    return validate_polls(long.reset_index(drop=True))


def load(raw_dir: Path, refresh: bool = True) -> pd.DataFrame:
    """Download (unless cached) and return the tidy SwedishPolls table."""
    csv_path = raw_dir / "swedish_polls.csv"
    if refresh or not csv_path.exists():
        csv_path = download(raw_dir)
    return to_tidy(csv_path)
