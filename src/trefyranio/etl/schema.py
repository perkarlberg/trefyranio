"""Canonical schema for the poll spine.

The tidy poll table is *long*: one row per (poll, party). Poll-level
attributes (pollster, fieldwork window, sample size, undecided share) are
repeated on each of a poll's party rows — denormalized on purpose so a single
parquet is self-contained for the modeling layer.
"""

from __future__ import annotations

import pandas as pd

# The eight current Riksdag parties, plus FI (relevant ~2014-2018) and a
# residual "other" bucket. Shares of these sum to ~1 over decided voters.
RIKSDAG_PARTIES = ["S", "M", "SD", "C", "V", "KD", "MP", "L"]
NAMED_PARTIES = RIKSDAG_PARTIES + ["FI"]
OTHER = "Övr"  # Övriga — small parties below the named set
SIMPLEX_PARTIES = NAMED_PARTIES + [OTHER]  # sum to 1 over decided voters

# Columns of the tidy long poll table.
TIDY_COLUMNS = [
    "poll_id",       # stable hash of the poll's identity
    "source",        # provenance: "SwedishPolls", "wikipedia2026", ...
    "pollster",      # canonical pollster (the Verian lineage is labelled "Sifo")
    "commissioner",  # who ordered it, when known (else None)
    "pub_date",      # publication date
    "field_start",   # fieldwork start (may be NaT)
    "field_end",     # fieldwork end (may be NaT)
    "n",             # sample size (may be NaN)
    "uncertain",     # undecided share of all respondents (fraction; may be NaN)
    "party",         # one of SIMPLEX_PARTIES
    "share",         # reported share as a fraction (0-1) of decided voters
]

# Pollster display names / current brands. The poll CSVs keep the long-running
# house label "Sifo" even though Kantar Sifo rebranded to **Verian** in 2023;
# we keep "Sifo" as the canonical id (one continuous house) and surface the
# current brand here. The 2023 phone->web methodology breaks (Novus, Verian)
# are handled as method-eras in Phase 2, not by renaming.
POLLSTER_DISPLAY = {
    "Sifo": "Verian (Kantar Sifo)",
    "SCB": "SCB (PSU)",
}


# --- Official results (SCB ME0104) ---------------------------------------

# SCB party codes -> our canonical codes. Note FP = Liberalerna (old
# Folkpartiet name SCB still uses); ÖVRIGA folds into the "other" bucket.
SCB_PARTY_MAP = {
    "M": "M", "C": "C", "FP": "L", "KD": "KD", "MP": "MP",
    "S": "S", "V": "V", "SD": "SD", "ÖVRIGA": OTHER,
}
# Non-party rows in the same SCB variable — used for turnout, not vote share.
SCB_INVALID = "OGILTIGA"
SCB_NONVOTERS = "VALSKOLKARE"

NATIONAL_REGION = "VR00"  # "Totalt för riket"
# The 29 current riksdag constituencies (the G-suffixed codes are pre-2018
# historical boundaries and are excluded from the current allocation set).
CURRENT_VALKRETSAR = [f"VR{i}" for i in range(1, 30)]

# Canonical party ordering for results tables (parties that take seats + other).
RESULT_PARTIES = RIKSDAG_PARTIES + [OTHER]


def validate_polls(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the tidy poll table is well-formed; return it unchanged.

    Catches schema drift early (a renamed upstream column, an unmapped party,
    shares accidentally left in percent) before bad data reaches the model.
    """
    missing = set(TIDY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"tidy poll table missing columns: {sorted(missing)}")

    bad_party = set(df["party"]) - set(SIMPLEX_PARTIES)
    if bad_party:
        raise ValueError(f"unknown party codes: {sorted(bad_party)}")

    s = df["share"].dropna()
    if not s.empty and (s.min() < -1e-9 or s.max() > 1.0 + 1e-6):
        raise ValueError(
            f"shares must be fractions in [0,1]; got [{s.min():.4f}, {s.max():.4f}] "
            "(did percent->fraction conversion get skipped?)"
        )

    # Each poll's simplex shares should sum to ~1. A sum well above 1 means two
    # polls collapsed under one poll_id (id collision) — a hard structural error.
    sums = df.groupby("poll_id")["share"].sum()
    if (sums > 1.5).any():
        bad = sums[sums > 1.5]
        raise ValueError(
            f"{len(bad)} poll_id(s) have share sum > 1.5 (max {sums.max():.1f}) — "
            "id collision: distinct polls merged under one id"
        )
    # Soft check: a few polls may legitimately sum a little off (rounding, an
    # omitted party), but a large fraction signals a normalization bug.
    off = sums[(sums < 0.9) | (sums > 1.1)]
    if len(off) > 0.02 * len(sums) + 5:
        raise ValueError(
            f"{len(off)} polls have share sums far from 1.0 — check normalization"
        )
    return df
