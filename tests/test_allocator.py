"""Validate the seat allocator against real Riksdag outcomes.

Ground truth: official 2022 Riksdag results (Valmyndigheten, via the
Wikipedia results table). If modified Sainte-Lague on national vote totals
reproduces the actual 349-seat distribution, the deterministic core that the
whole simulator rests on is proven correct.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trefyranio.allocator import (  # noqa: E402
    allocate_national,
    first_divisor_for_year,
)
from trefyranio.model import PROCESSED_DIR  # noqa: E402

# Official 2022 results: party -> (votes, actual seats won).
RESULTS_2022 = {
    "S": (1_964_474, 107),
    "SD": (1_330_325, 73),
    "M": (1_237_428, 68),
    "V": (437_050, 24),
    "C": (434_945, 24),
    "KD": (345_712, 19),
    "MP": (329_242, 18),
    "L": (298_542, 16),
    # Below 4% — must receive zero seats.
    "Nyans": (28_352, 0),
    "AfS": (16_646, 0),
}
TOTAL_VALID_2022 = 6_477_970


def test_2022_reproduces_official_seats():
    votes = {p: v for p, (v, _) in RESULTS_2022.items()}
    result = allocate_national(votes)

    expected = {p: s for p, (_, s) in RESULTS_2022.items()}
    assert result.seats == expected, (
        f"\nexpected: {expected}\n     got: {result.seats}"
    )
    assert sum(result.seats.values()) == 349


def test_2022_threshold_gate():
    votes = {p: v for p, (v, _) in RESULTS_2022.items()}
    result = allocate_national(votes)

    # The 8 parliamentary parties qualify; the two micro-parties do not.
    assert result.qualified == {"S", "SD", "M", "V", "C", "KD", "MP", "L"}
    assert result.seats["Nyans"] == 0
    assert result.seats["AfS"] == 0


def test_vote_totals_match_published():
    votes = {p: v for p, (v, _) in RESULTS_2022.items()}
    # Our subset sums close to the official total (it omits the long tail of
    # micro-parties, so it should be slightly under the full valid-vote count).
    assert sum(votes.values()) <= TOTAL_VALID_2022


def test_first_divisor_for_year():
    assert first_divisor_for_year(1973) == 1.4
    assert first_divisor_for_year(2014) == 1.4
    assert first_divisor_for_year(2018) == 1.2
    assert first_divisor_for_year(2026) == 1.2


@pytest.mark.skipif(
    not (PROCESSED_DIR / "seats_actual.parquet").exists(),
    reason="spines not built",
)
@pytest.mark.parametrize("year", [2018, 2022])
def test_national_proportional_exact_under_current_rules(year):
    """Under the post-2018 rules the leveling seats make the outcome fully
    nationally proportional among 4%+ parties, so national-proportional
    allocation reproduces the real per-party seat totals EXACTLY — no
    per-constituency allocation needed for the forecast. (Pre-2018 elections
    deviate by a few seats; that disproportionality is what the 2018 reform
    removed.) Ground truth = seats_actual VR00 (the national-total row)."""
    import pandas as pd

    res = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    seats = pd.read_parquet(PROCESSED_DIR / "seats_actual.parquet")
    votes = res[res.election_year == year].set_index("party")["votes"].to_dict()
    got = allocate_national(
        votes, first_divisor=first_divisor_for_year(year),
        ignore_parties=frozenset({"Övr"}),
    )
    actual = (seats[(seats.election_year == year) & (seats.region_code == "VR00")]
              .set_index("party")["seats"].to_dict())
    for party, s in actual.items():
        if party == "Övr":
            continue
        assert got.seats.get(party, 0) == int(s), (
            f"{year} {party}: allocator {got.seats.get(party, 0)} vs actual {s}"
        )


def test_biproportional_matches_both_margins_and_reproduces_2022():
    """Biproportional apportionment must hit BOTH margins exactly and closely
    reproduce the real 2022 per-valkrets seats (the map's fidelity guarantee)."""
    import numpy as np
    import pandas as pd
    from trefyranio.allocator import biproportional

    rv = pd.read_parquet(PROCESSED_DIR / "results_valkrets.parquet")
    sa = pd.read_parquet(PROCESSED_DIR / "seats_actual.parquet")
    RIKS = ["S", "M", "SD", "C", "V", "KD", "MP", "L"]
    base = rv[rv.election_year == 2022]
    sav = sa[(sa.election_year == 2022) & (sa.region_code != "VR00")]
    vcodes = sorted(sav.region_code.unique())
    V = np.array([[base[(base.valkrets_code == v) & (base.party == p)].votes.sum()
                   for p in RIKS] for v in vcodes], dtype=float)
    actual = np.array([[int(sav[(sav.region_code == v) & (sav.party == p)].seats.sum())
                        for p in RIKS] for v in vcodes])
    row_t = actual.sum(1)
    col_t = actual.sum(0)
    M = biproportional(V, row_t, col_t)
    assert (M.sum(1) == row_t).all()          # valkrets budgets exact
    assert (M.sum(0) == col_t).all()          # national party totals exact
    assert np.abs(M - actual).sum() <= 12     # close to the real allocation (~8/349)


def test_other_bucket_excluded_from_seats():
    """A lumped 'other' total above 4% must win no seats (it's many
    sub-threshold parties), but still counts toward the threshold denominator."""
    votes = {"S": 4_000_000, "M": 3_000_000, "SD": 2_000_000, "Övr": 500_000}
    # Övr is 500k/9.5M = 5.3% > 4%, so it would qualify if not excluded.
    res = allocate_national(votes, ignore_parties=frozenset({"Övr"}))
    assert res.seats["Övr"] == 0
    assert sum(res.seats.values()) == 349
    # The three real parties split all 349 seats.
    assert {p for p, s in res.seats.items() if s > 0} == {"S", "M", "SD"}
