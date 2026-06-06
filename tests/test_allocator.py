"""Validate the seat allocator against real Riksdag outcomes.

Ground truth: official 2022 Riksdag results (Valmyndigheten, via the
Wikipedia results table). If modified Sainte-Lague on national vote totals
reproduces the actual 349-seat distribution, the deterministic core that the
whole simulator rests on is proven correct.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trefyranio.allocator import (  # noqa: E402
    allocate_national,
    first_divisor_for_year,
)

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
