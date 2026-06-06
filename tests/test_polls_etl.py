"""Tests for the poll-spine ETL: tidy shape, normalization, collision-proof ids.

Hermetic — builds a tiny synthetic raw CSV in the SwedishPolls format, so no
network or cached data is needed. An optional integration check runs against
the real cached parquet when it's present.
"""

from pathlib import Path

import pandas as pd
import pytest

from trefyranio.etl import swedish_polls
from trefyranio.etl.schema import SIMPLEX_PARTIES

RAW_COLUMNS = [
    "PublYearMonth", "Company", "M", "L", "C", "KD", "S", "V", "MP", "SD",
    "FI", "Uncertain", "n", "PublDate", "collectPeriodFrom", "collectPeriodTo",
    "approxPeriod", "house",
]


def _synthetic_raw(tmp_path: Path) -> Path:
    rows = [
        # A modern, fully-dated poll: named parties sum to 98 -> Övr ~2%.
        ["2026-maj", "TV4", 17.3, 2.5, 6.1, 4.5, 33.9, 8.6, 6.6, 18.3, 0.0,
         None, 4542, "2026-05-28", "2026-05-20", "2026-05-28", False, "Novus"],
        # Two OLD polls that would COLLIDE under a date+n-only id: same house,
        # missing dates, same n — distinguished only by PublYearMonth.
        ["1985-mar", "Sifo", 22.0, 12.0, 12.0, 2.0, 44.0, 5.0, 0.0, 0.0, 0.0,
         5.0, 1100, None, None, None, True, "Sifo"],
        ["1985-apr", "Sifo", 21.0, 12.5, 12.5, 2.0, 44.0, 5.0, 0.0, 0.0, 0.0,
         5.0, 1100, None, None, None, True, "Sifo"],
    ]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


def test_tidy_shape_and_normalization(tmp_path):
    tidy = swedish_polls.to_tidy(_synthetic_raw(tmp_path))

    # One row per (poll, party) across the full simplex.
    assert set(tidy["party"]) == set(SIMPLEX_PARTIES)
    counts = tidy.groupby("poll_id").size()
    assert (counts == len(SIMPLEX_PARTIES)).all()

    # Shares are fractions, and each poll's simplex sums to ~1.
    assert tidy["share"].max() <= 1.0 + 1e-9
    sums = tidy.groupby("poll_id")["share"].sum()
    assert sums.between(0.99, 1.01).all()

    # Övr residual on the modern poll = 1 - sum(named) = 1 - 0.978 = 0.022.
    modern = tidy[(tidy["pollster"] == "Novus") & (tidy["party"] == "Övr")]
    assert float(modern["share"].iloc[0]) == pytest.approx(0.022, abs=1e-9)

    # Commissioner captured when it differs from the pollster house.
    assert (tidy[tidy["pollster"] == "Novus"]["commissioner"] == "TV4").all()


def test_no_id_collision_for_undated_polls(tmp_path):
    """Regression: undated same-n polls from the same house in different
    months must get distinct poll_ids."""
    tidy = swedish_polls.to_tidy(_synthetic_raw(tmp_path))
    assert tidy["poll_id"].nunique() == 3  # exactly the three input polls


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data/processed/polls.parquet").exists(),
    reason="real poll spine not built",
)
def test_real_spine_integrity():
    path = Path(__file__).resolve().parents[1] / "data/processed/polls.parquet"
    df = pd.read_parquet(path)
    sums = df.groupby("poll_id")["share"].sum()
    assert sums.max() <= 1.1, "id collision in real spine"
    assert (df.groupby("poll_id").size() == len(SIMPLEX_PARTIES)).all()
