"""Tests for the backtest calibration math (hermetic — no model fits)."""

import numpy as np
import pandas as pd

from trefyranio import backtest
from trefyranio.model import PARTY_ORDER


def test_miss_from_moments_recovers_error_rms_when_trend_tight():
    # Trend is essentially certain → miss must absorb all the realized error.
    errs = np.array([0.1, -0.1, 0.2, -0.2])
    sd_trend = np.zeros_like(errs)
    expected = np.sqrt((errs ** 2).mean())
    assert backtest.miss_from_moments(errs, sd_trend) == expected


def test_miss_from_moments_clamps_to_zero():
    # Trend spread already exceeds the errors → no extra variance needed.
    errs = np.array([0.01, -0.01])
    sd_trend = np.array([0.5, 0.5])
    assert backtest.miss_from_moments(errs, sd_trend) == 0.0


def test_miss_subtracts_trend_variance():
    errs = np.array([0.3, -0.3])         # mean err² = 0.09
    sd_trend = np.array([0.1, 0.1])      # mean trend var = 0.01
    assert backtest.miss_from_moments(errs, sd_trend) == np.sqrt(0.08)


def test_actual_shares_normalized_and_ordered():
    results = pd.DataFrame({
        "election_year": [2022] * 3,
        "party": ["S", "M", "Övr"],
        "share": [0.3, 0.2, 0.5],
    })
    v = backtest.actual_shares(results, 2022)
    assert len(v) == len(PARTY_ORDER)
    assert np.isclose(v.sum(), 1.0)
    assert v[PARTY_ORDER.index("S")] > v[PARTY_ORDER.index("M")]
