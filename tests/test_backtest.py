"""Tests for the backtest calibration seam (hermetic — no model fits)."""

import numpy as np
import pandas as pd

from trefyranio import backtest
from trefyranio.model import PARTY_ORDER, _alr


def test_project_seam_widens_with_horizon():
    # The backtest calibrates by projecting a cached (last_alr, drift) forward.
    # A near-certain last-poll latent should still gain spread from the forward
    # innovation, and more at a longer horizon — this is what _coverage tunes.
    p = np.array([0.30, 0.20, 0.18, 0.08, 0.08, 0.06, 0.05, 0.03, 0.02])
    p = p / p.sum()
    last = np.tile(_alr(p), (3000, 1))
    d = {"last_alr": last, "drift": np.zeros_like(last), "actual": p}
    near = backtest._project(d, 0, 0.015, 0.0384, 0.2)
    far = backtest._project(d, 14, 0.015, 0.0384, 0.2)
    assert np.allclose(far.sum(1), 1.0)
    assert (far.std(0) >= near.std(0) - 1e-9).all()


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
