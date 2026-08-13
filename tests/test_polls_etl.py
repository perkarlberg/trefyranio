"""Tests for the poll-spine ETL: tidy shape, normalization, collision-proof ids.

Hermetic — builds a tiny synthetic raw CSV in the SwedishPolls format, so no
network or cached data is needed. An optional integration check runs against
the real cached parquet when it's present.
"""

from pathlib import Path

import pandas as pd
import pytest

from trefyranio.etl import manual_polls, swedish_polls
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


# --- manual supplement -----------------------------------------------------

# A poll upstream hasn't published yet: same house as no existing row's
# field-end, so it must be ADDED.
_FRESH = ["2026-aug", "Aftonbladet", 16.8, 2.4, 7.3, 7.5, 30.2, 6.5, 6.6, 20.2,
          0.0, None, 2016, "2026-08-12", "2026-07-29", "2026-08-10", False,
          "Demoskop"]


def _supplement(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=manual_polls.RAW_COLUMNS)


def test_supplement_adds_poll_missing_upstream(tmp_path):
    raw = _synthetic_raw(tmp_path)
    base = swedish_polls.to_tidy(raw)
    tidy = swedish_polls.to_tidy(raw, supplement=_supplement([_FRESH]))

    assert tidy["poll_id"].nunique() == base["poll_id"].nunique() + 1

    added = tidy[tidy["pollster"] == "Demoskop"]
    assert not added.empty
    # Normalized like any other poll: percent -> fraction, full simplex, sums ~1.
    assert float(added[added["party"] == "S"]["share"].iloc[0]) == pytest.approx(0.302)
    assert added["share"].sum() == pytest.approx(1.0, abs=1e-9)
    # Provenance is preserved rather than mislabelled as upstream.
    assert (added["source"] == manual_polls.SOURCE).all()
    assert added["poll_id"].iloc[0].startswith("ma_")
    assert (tidy[tidy["pollster"] == "Novus"]["source"] == "SwedishPolls").all()


def test_supplement_row_drops_once_upstream_has_it(tmp_path):
    """The whole point: no duplicate poll when SwedishPolls catches up."""
    raw = _synthetic_raw(tmp_path)
    # The synthetic upstream already contains a Novus poll ending 2026-05-28;
    # a supplement row for the same house+field-end must be ignored, even with
    # a different publication date and sample size than we typed in.
    stale = ["2026-maj", "TV4", 17.3, 2.5, 6.1, 4.5, 33.9, 8.6, 6.6, 18.3, 0.0,
             None, 4000, "2026-05-30", "2026-05-20", "2026-05-28", False, "Novus"]

    base = swedish_polls.to_tidy(raw)
    tidy = swedish_polls.to_tidy(raw, supplement=_supplement([stale]))

    assert tidy["poll_id"].nunique() == base["poll_id"].nunique()
    assert not (tidy["source"] == manual_polls.SOURCE).any()
    # Upstream's numbers win — n stays 4542, not the 4000 we typed.
    assert tidy[tidy["pollster"] == "Novus"]["n"].iloc[0] == 4542


def test_supplement_rejects_fractions_typed_as_percent(tmp_path):
    """A share typed as 0.302 would otherwise model S as a 0.3% party."""
    path = tmp_path / "manual_polls.csv"
    bad = list(_FRESH)
    for i in range(2, 10):  # the named-party columns
        bad[i] = bad[i] / 100.0
    _supplement([bad]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="percentages"):
        manual_polls.load(path)


def test_supplement_requires_field_end(tmp_path):
    path = tmp_path / "manual_polls.csv"
    bad = list(_FRESH)
    bad[15] = None  # collectPeriodTo — the dedupe key
    _supplement([bad]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="collectPeriodTo"):
        manual_polls.load(path)


def test_missing_supplement_file_is_fine(tmp_path):
    assert manual_polls.load(tmp_path / "nope.csv").empty


def test_shipped_supplement_file_parses():
    """The committed CSV must stay loadable — a typo here breaks every build."""
    path = Path(__file__).resolve().parents[1] / "data/manual_polls.csv"
    if path.exists():
        manual_polls.load(path)  # raises on malformed rows


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
