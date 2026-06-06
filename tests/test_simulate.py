"""Tests for the seat & government simulator.

Hermetic: synthetic vote shares and synthetic seat matrices, so the bloc
arithmetic and threshold logic are pinned without running the model.
"""

import numpy as np

from trefyranio import simulate
from trefyranio.etl.schema import RIKSDAG_PARTIES
from trefyranio.model import PARTY_ORDER

# A clean draw: every party ≥4% except L at 3% (must be eliminated).
SHARES = {
    "S": 0.20, "M": 0.15, "SD": 0.15, "C": 0.13, "V": 0.12,
    "KD": 0.10, "MP": 0.10, "L": 0.03, "Övr": 0.02,
}


def test_simulate_seats_totals_and_threshold():
    vec = np.array([[SHARES[p] for p in PARTY_ORDER]] * 5)  # 5 identical draws
    seats = simulate.simulate_seats(vec, PARTY_ORDER)
    assert seats.shape == (5, len(RIKSDAG_PARTIES))
    assert (seats.sum(axis=1) == 349).all()
    # L is under 4% -> always zero seats; everyone else seated.
    L = RIKSDAG_PARTIES.index("L")
    assert (seats[:, L] == 0).all()
    assert (np.delete(seats, L, axis=1) > 0).all()


def test_seat_distribution_threshold_fields():
    vec = np.array([[SHARES[p] for p in PARTY_ORDER]] * 10)
    seats = simulate.simulate_seats(vec, PARTY_ORDER)
    df = simulate.seat_distribution(seats).set_index("party")
    assert df.loc["L", "p_below_4pct"] == 1.0
    assert df.loc["L", "p_in_riksdag"] == 0.0
    # p_in_riksdag and p_below_4pct are complements.
    assert np.allclose(df["p_in_riksdag"] + df["p_below_4pct"], 1.0)


def _synthetic_seats(per_party: dict) -> np.ndarray:
    """One-draw seat matrix from a party->seats dict (fills missing with 0)."""
    row = [per_party.get(p, 0) for p in RIKSDAG_PARTIES]
    return np.array([row])


def test_government_partition_sums_to_one():
    # Right=170, Left=150, Centre=29 -> nobody ≥175 alone -> kingmaker.
    seats = _synthetic_seats({"M": 80, "SD": 60, "KD": 20, "L": 10,   # right=170
                              "S": 100, "V": 30, "MP": 20,            # left=150
                              "C": 29})                               # 349
    gov = simulate.government_outlook(seats)
    total = gov["p_right_majority"] + gov["p_left_majority"] + gov["p_centre_kingmaker"]
    assert total == 1.0
    assert gov["p_centre_kingmaker"] == 1.0


def test_coalition_majority_detection():
    seats = _synthetic_seats({"S": 120, "V": 35, "MP": 25, "C": 20,   # red-green+C=200
                              "M": 80, "SD": 49, "KD": 20})           # 349
    coal = simulate.coalition_table(seats).set_index("coalition")
    assert coal.loc["Red-green+C (S+V+MP+C)", "p_majority"] == 1.0
    assert coal.loc["Tidö (M+KD+L+SD)", "p_majority"] == 0.0
