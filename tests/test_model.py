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
    KM1,
    PARTY_ORDER,
    SHRINK_IND,
    _alr,
    _build_house_priors,
    _load_pollster_weights,
    _softmax_with_ref,
    forecast_with_miss,
    miss_sigma_for_horizon,
)


def test_miss_sigma_grows_with_horizon():
    # Tighter near the election, wider far out. Coverage-calibrated on the
    # converged model across 4 cycles: ~1.50pp at H=0, ~1.65pp at the 14-week point.
    assert miss_sigma_for_horizon(0) < miss_sigma_for_horizon(8) < miss_sigma_for_horizon(20)
    assert miss_sigma_for_horizon(14) == pytest.approx(0.0165, abs=2e-3)
    assert miss_sigma_for_horizon(0) == pytest.approx(0.015, abs=1e-3)


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


def _anchor():
    p = np.array([0.30, 0.20, 0.17, 0.08, 0.08, 0.06, 0.04, 0.04, 0.03])
    return p / p.sum()


def test_house_prior_debias_to_zero():
    # A pollster whose historical lean EQUALS the field-wide industry bias has no
    # consensus-relative lean → its de-biased ALR prior should be ~0.
    parties = PARTY_ORDER[:-1]  # 8 Riksdag parties (ratings exclude Övr)
    bias = {"S": -0.02, "M": 0.01, "SD": -0.015, "C": 0.0, "V": 0.012, "KD": 0.003, "MP": 0.008, "L": 0.004}
    industry = pd.DataFrame({"party": list(bias), "bias": list(bias.values())})
    # pollster "Same" mirrors the field; "Lean" adds +1pp on SD beyond the field.
    rows = []
    for party in parties:
        rows.append(("Same", party, bias[party]))
        rows.append(("Lean", party, bias[party] + (0.01 if party == "SD" else 0.0)))
    he = pd.DataFrame(rows, columns=["pollster", "party", "house_effect"])
    prior = _build_house_priors(["Same", "Lean"], _anchor(), he, industry)
    assert prior.shape == (2, KM1)
    assert np.allclose(prior[0], 0.0, atol=1e-9)            # mirrors field → no lean
    sd_idx = PARTY_ORDER.index("SD") - 1                    # ALR drops the reference (S)
    assert prior[1, sd_idx] > 0.02                          # the +1pp SD lean shows up
    assert not np.allclose(prior[1], 0.0)


def test_house_prior_new_entrant_zero():
    industry = pd.DataFrame({"party": PARTY_ORDER[:-1], "bias": [0.0] * 8})
    he = pd.DataFrame([("Old", p, 0.005) for p in PARTY_ORDER[:-1]],
                      columns=["pollster", "party", "house_effect"])
    prior = _build_house_priors(["Old", "BrandNew"], _anchor(), he, industry)
    assert np.allclose(prior[1], 0.0)                       # no history → zero prior


def test_pollster_weights_fallback():
    ratings = pd.DataFrame({"pollster": ["A", "B"], "weight": [1.3, 0.8]})
    w = _load_pollster_weights(["A", "New", "B"], ratings)
    assert w.tolist() == [1.3, 1.0, 0.8]                    # new entrant → 1.0


def test_industry_shift_lifts_understated_party():
    # Field understates S (bias<0) → the shift must push S's election-day share UP.
    base_alr = np.tile(_alr(_anchor()), (400, 1))
    shift = np.zeros(K)
    shift[PARTY_ORDER.index("S")] = SHRINK_IND * 0.02       # +shift on S
    no_shift = forecast_with_miss(base_alr, sigma_miss=0.0, seed=3)
    with_shift = forecast_with_miss(base_alr, sigma_miss=0.0, seed=3, industry_shift=shift)
    assert with_shift[:, PARTY_ORDER.index("S")].mean() > no_shift[:, PARTY_ORDER.index("S")].mean()
    assert np.allclose(with_shift.sum(axis=1), 1.0)


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
