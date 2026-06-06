"""Tests for the official-results spine (SCB ME0104).

Run against the built parquets; skipped if the spine hasn't been built
(`python -m trefyranio.etl.build_results`). The headline claim: the allocator
reproduces the 2018 and 2022 national seat distributions exactly from the
fetched SCB vote counts — the two elections held under the current rules.
"""

from pathlib import Path

import pandas as pd
import pytest

from trefyranio.allocator import allocate_national, first_divisor_for_year

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "results_national.parquet").exists(),
    reason="results spine not built (run: python -m trefyranio.etl.build_results)",
)


def _load(name):
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def test_2022_national_shares_match_official():
    nat = _load("results_national")
    s = nat[nat.election_year == 2022].set_index("party")["share"]
    assert s["S"] == pytest.approx(0.3033, abs=5e-4)
    assert s["SD"] == pytest.approx(0.2054, abs=5e-4)
    assert s["L"] == pytest.approx(0.0461, abs=5e-4)  # FP -> L mapping
    assert nat[nat.election_year == 2022]["share"].sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("year", [2018, 2022])
def test_allocator_reproduces_current_rule_elections(year):
    nat = _load("results_national")
    seats = _load("seats_actual")
    votes = dict(zip(
        nat[nat.election_year == year]["party"],
        nat[nat.election_year == year]["votes"],
    ))
    predicted = {
        p: v for p, v in allocate_national(
            votes,
            first_divisor=first_divisor_for_year(year),
            ignore_parties=frozenset({"Övr"}),
        ).seats.items() if v > 0
    }
    actual = dict(zip(
        seats[(seats.election_year == year) & (seats.region_code == "VR00")]["party"],
        seats[(seats.election_year == year) & (seats.region_code == "VR00")]["seats"],
    ))
    assert predicted == actual
    assert sum(predicted.values()) == 349


def test_valkrets_votes_reconcile_to_national():
    """From 1994 on (current 29-valkrets boundaries) the per-constituency vote
    sums must equal the national totals party by party."""
    nat = _load("results_national").set_index(["election_year", "party"])["votes"]
    vk = _load("results_valkrets")
    vk_sum = vk.groupby(["election_year", "party"])["votes"].sum()
    joined = pd.concat([vk_sum.rename("vk"), nat.rename("nat")], axis=1).dropna()
    modern = joined[joined.index.get_level_values("election_year") >= 1994]
    assert (modern["vk"] == modern["nat"]).all()
    assert vk[vk.election_year == 2022]["valkrets_code"].nunique() == 29
