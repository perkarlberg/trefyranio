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
    cost_of_ruling,
    fund_weight,
    fundamentals_prior,
    miss_sigma_for_horizon,
    project_to_election,
)


def test_miss_sigma_grows_with_horizon():
    # Tighter near the election, wider far out. Model-carried calibration on the
    # 4-cycle forward projection: ~1.55pp at H=0, ~1.80pp at the 14-week point.
    assert miss_sigma_for_horizon(0) < miss_sigma_for_horizon(8) < miss_sigma_for_horizon(20)
    assert miss_sigma_for_horizon(14) == pytest.approx(0.018, abs=2e-3)
    assert miss_sigma_for_horizon(0) == pytest.approx(0.0155, abs=1e-3)


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


def test_projection_widens_and_grows_with_horizon():
    # A near-degenerate last-poll latent (tiny posterior spread); zero drift.
    last = np.tile(_alr(np.full(K, 1 / K)), (4000, 1))
    last += np.random.default_rng(1).normal(0, 0.002, last.shape)
    drift = np.zeros_like(last)
    tight = _softmax_with_ref(last)
    near = project_to_election(last, drift, horizon=0, seed=1)
    far = project_to_election(last, drift, horizon=20, seed=1)
    # The forward innovation widens the spread, and more so at a longer horizon.
    assert (far.std(axis=0) > near.std(axis=0)).all()
    assert (near.std(axis=0) > tight.std(axis=0)).all()
    # softmax keeps a valid simplex — no clipping needed.
    assert np.allclose(far.sum(axis=1), 1.0)
    assert (far >= 0).all()


def test_projection_marginal_is_near_uniform_pp():
    # The per-party logit scaling targets a ~uniform share-space sigma. Build a
    # spread of party sizes well above the floor and check marginal SDs cluster.
    shares = np.array([0.32, 0.20, 0.17, 0.09, 0.08, 0.06, 0.05, 0.02, 0.01])
    shares = shares / shares.sum()
    last = np.tile(_alr(shares), (8000, 1))
    out = project_to_election(last, np.zeros_like(last), horizon=14, rho=0.0, seed=2)
    target = miss_sigma_for_horizon(14)
    # Parties comfortably above the 4% floor should sit near the target sigma.
    big = [i for i, p in enumerate(shares) if 0.05 < p < 0.5]
    sds = out.std(axis=0)[big]
    assert np.all(np.abs(sds - target) < 0.6 * target)  # within ~60% of uniform target


def test_projection_bloc_correlation_inflates_bloc_variance():
    shares = np.array([0.30, 0.20, 0.18, 0.08, 0.08, 0.06, 0.05, 0.03, 0.02])
    shares = shares / shares.sum()
    last = np.tile(_alr(shares), (8000, 1))
    drift = np.zeros_like(last)
    right = [PARTY_ORDER.index(p) for p in ("M", "SD", "KD", "L")]  # one bloc
    iid = project_to_election(last, drift, 14, rho=0.0, seed=5)
    cor = project_to_election(last, drift, 14, rho=0.5, seed=5)
    # Correlated within-bloc misses inflate the bloc-total variance vs iid.
    assert cor[:, right].sum(1).std() > iid[:, right].sum(1).std()


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


def test_cost_of_ruling_is_negative():
    # Two synthetic cycles where the governing party (S) loses 2pp each time.
    import trefyranio.model as m
    res = pd.DataFrame({
        "election_year": [1998, 1998, 2002, 2002],
        "party": ["S", "M", "S", "M"],
        "share": [0.40, 0.30, 0.38, 0.32],   # S −2pp 1998→2002; M +2pp
    })
    old = m.GOVERNMENTS
    try:
        m.GOVERNMENTS = {2002: {"S"}}        # S governed into 2002 (prev 1998)
        assert cost_of_ruling(res) == pytest.approx(-0.02, abs=1e-9)
    finally:
        m.GOVERNMENTS = old


def test_fundamentals_prior_penalizes_governing():
    prev = np.array([0.30, 0.25, 0.20, 0.06, 0.07, 0.05, 0.04, 0.02, 0.01])
    prev = prev / prev.sum()
    gov = {"M", "KD", "L", "SD"}
    f = fundamentals_prior(prev, gov, delta=-0.014)
    gov_idx = [PARTY_ORDER.index(p) for p in gov]
    opp_idx = [i for i in range(K) if PARTY_ORDER[i] not in gov and PARTY_ORDER[i] != "Övr"]
    assert np.allclose(f.sum(), 1.0)
    assert (f[gov_idx] < prev[gov_idx]).all()          # governing pulled down
    assert (f[opp_idx] >= prev[opp_idx] - 1e-12).all()  # opposition gains the freed mass


def test_fund_weight_zero_at_election_and_monotone():
    assert fund_weight(0) == 0.0
    assert fund_weight(8) >= fund_weight(0)
    assert fund_weight(30) >= fund_weight(8)


def test_projection_blends_toward_fundamentals():
    last = np.tile(_alr(np.full(K, 1 / K)), (2000, 1))
    drift = np.zeros_like(last)
    fund = np.zeros(K); fund[PARTY_ORDER.index("S")] = 1.0  # degenerate: all mass on S
    none = project_to_election(last, drift, 14, fund_w=0.0, seed=0)
    blended = project_to_election(last, drift, 14, fundamentals=fund, fund_w=0.4, seed=0)
    si = PARTY_ORDER.index("S")
    assert blended[:, si].mean() > none[:, si].mean()   # pulled toward the fundamentals
    assert np.allclose(blended.sum(1), 1.0)


def test_industry_shift_lifts_understated_party():
    # Field understates S (bias<0) → the shift must push S's election-day share UP.
    last = np.tile(_alr(_anchor()), (400, 1))
    drift = np.zeros_like(last)
    shift = np.zeros(K)
    shift[PARTY_ORDER.index("S")] = SHRINK_IND * 0.02       # +shift on S
    no_shift = project_to_election(last, drift, horizon=14, seed=3)
    with_shift = project_to_election(last, drift, horizon=14, seed=3, industry_shift=shift)
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
