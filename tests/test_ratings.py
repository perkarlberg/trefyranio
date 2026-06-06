"""Tests for the pollster-ratings layer.

Hermetic tests pin the shrinkage / weight math on a tiny synthetic error
table; an optional data smoke-test checks the real outputs are well-formed.
"""

from pathlib import Path

import pandas as pd
import pytest

from trefyranio import ratings
from trefyranio.ratings import (
    SHRINK_K,
    house_effects,
    industry_bias,
    pollster_ratings,
)


def _errs():
    # Two pollsters; "Good" is closer to actual than "Bad" in both elections.
    rows = [
        # pollster, party, election_year, error  (poll - actual, fractions)
        ("Good", "S", 2018, +0.005), ("Good", "M", 2018, -0.005),
        ("Good", "S", 2022, +0.005), ("Good", "M", 2022, -0.005),
        ("Bad",  "S", 2018, +0.020), ("Bad",  "M", 2018, -0.020),
        ("Bad",  "S", 2022, +0.020), ("Bad",  "M", 2022, -0.020),
    ]
    df = pd.DataFrame(rows, columns=["pollster", "party", "election_year", "error"])
    df["abs_error"] = df["error"].abs()
    return df


def test_house_effect_shrinkage():
    he = house_effects(_errs()).set_index(["pollster", "party"])
    # Good/S: raw mean +0.005 over n=2 elections, shrunk by K.
    raw, n = 0.005, 2
    assert he.loc[("Good", "S"), "raw"] == pytest.approx(raw)
    assert he.loc[("Good", "S"), "house_effect"] == pytest.approx(raw * n / (n + SHRINK_K))


def test_industry_bias_is_mean_of_pollster_means():
    bias = industry_bias(_errs()).set_index("party")["bias"]
    # S: Good +0.005, Bad +0.020 -> field mean +0.0125.
    assert bias["S"] == pytest.approx(0.0125)
    assert bias["M"] == pytest.approx(-0.0125)


def test_ratings_weight_centered_and_ordered():
    r = pollster_ratings(_errs()).set_index("pollster")
    assert r["weight"].mean() == pytest.approx(1.0)
    # The more accurate pollster earns the larger weight.
    assert r.loc["Good", "weight"] > r.loc["Bad", "weight"]


def test_ratings_weight_bounded_for_extreme_outlier():
    """A pollster far more accurate than the field must not produce an infinite
    or negative weight (the predictive_error floor guards the division)."""
    rows = [
        ("Perfect", "S", 2018, 0.0), ("Perfect", "M", 2018, 0.0),
        ("Perfect", "S", 2022, 0.0), ("Perfect", "M", 2022, 0.0),
        ("Awful", "S", 2018, 0.10), ("Awful", "M", 2018, -0.10),
        ("Awful", "S", 2022, 0.10), ("Awful", "M", 2022, -0.10),
    ]
    df = pd.DataFrame(rows, columns=["pollster", "party", "election_year", "error"])
    df["abs_error"] = df["error"].abs()
    r = pollster_ratings(df)
    assert r["weight"].notna().all()
    assert (r["weight"] > 0).all()
    assert r["weight"].max() < 10  # bounded, not runaway


@pytest.mark.skipif(
    not (ratings.PROCESSED_DIR / "pollster_ratings.parquet").exists(),
    reason="ratings not built",
)
def test_real_ratings_wellformed():
    r = pd.read_parquet(ratings.PROCESSED_DIR / "pollster_ratings.parquet")
    assert (r["weight"] > 0).all()
    assert r["weight"].mean() == pytest.approx(1.0, abs=1e-6)
    assert r["mean_abs_error"].between(0, 0.05).all()  # final-poll MAE < 5pp
