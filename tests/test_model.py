"""Tests for the Bayesian model's pure components.

The NUTS fit is too slow for unit tests (it's exercised by `python -m
trefyranio.model`); here we pin the deterministic transforms and the data-prep,
which is where silent bugs (bad ALR, broken normalization, wrong week index)
would hide.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trefyranio import model
from trefyranio.model import (
    K,
    PARTY_ORDER,
    _alr,
    _softmax_with_ref,
    forecast_with_miss,
    miss_sigma_for_horizon,
)


def test_miss_sigma_grows_with_horizon():
    # Tighter near the election, wider far out; ~2.25pp at the 14-week calibration point.
    assert miss_sigma_for_horizon(0) < miss_sigma_for_horizon(8) < miss_sigma_for_horizon(20)
    assert miss_sigma_for_horizon(14) == pytest.approx(0.0225, abs=2e-3)
    assert miss_sigma_for_horizon(0) == pytest.approx(0.016, abs=1e-3)


def test_alr_softmax_roundtrip():
    p = np.array([0.30, 0.20, 0.18, 0.07, 0.07, 0.05, 0.05, 0.05, 0.03])
    p = p / p.sum()
    recovered = _softmax_with_ref(_alr(p)[None, :])[0]
    assert np.allclose(recovered, p, atol=1e-9)


def test_softmax_sums_to_one():
    alr = np.random.default_rng(0).normal(size=(50, K - 1))
    shares = _softmax_with_ref(alr)
    assert shares.shape == (50, K)
    assert np.allclose(shares.sum(axis=1), 1.0)


def test_miss_term_widens_spread():
    # A near-degenerate trend (tiny posterior spread) at one point.
    base = np.tile(_alr(np.full(K, 1 / K)), (500, 1))
    base += np.random.default_rng(1).normal(0, 0.002, base.shape)
    tight = _softmax_with_ref(base)
    wide = forecast_with_miss(base, sigma_miss=0.02, seed=1)  # 2pp share-space miss
    # Every party's election-day spread grows once the miss error is added.
    assert (wide.std(axis=0) > tight.std(axis=0)).all()
    # Shares still form a valid simplex after clip + renormalize.
    assert np.allclose(wide.sum(axis=1), 1.0)
    assert (wide >= 0).all()


@pytest.mark.skipif(
    not (model.PROCESSED_DIR / "polls.parquet").exists(),
    reason="spines not built",
)
def test_prepare_shapes():
    polls = pd.read_parquet(model.PROCESSED_DIR / "polls.parquet")
    results = pd.read_parquet(model.PROCESSED_DIR / "results_national.parquet")
    d = model.prepare(polls, results)
    assert d.counts.shape[1] == K
    assert (d.totals > 0).all()
    assert d.election_week == d.n_weeks - 1
    # Polls fall before election day; the forecast horizon is non-empty.
    assert d.week.max() < d.election_week
    assert np.isfinite(d.anchor_alr).all()
    assert len(d.pollsters) >= 5
