"""Tests for the scheduled-refresh guard (ops/check_new_poll.py).

The guard decides whether the nightly job refits at all, so a false "nothing
new" is invisible: the forecast just silently stops updating. These pin the
cases that actually occur mid-campaign.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from trefyranio.model import CYCLE_2026

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_guard():
    """Import ops/check_new_poll.py (a script, not a package module)."""
    spec = importlib.util.spec_from_file_location(
        "check_new_poll", REPO_ROOT / "ops" / "check_new_poll.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spine(tmp_path: Path, polls: list[tuple[str, str, float | None]]) -> Path:
    """Write a minimal polls.parquet: [(poll_id, field_end, n), ...]."""
    rows = [{"poll_id": pid, "field_end": pd.Timestamp(end),
             "pub_date": pd.Timestamp(end), "n": n} for pid, end, n in polls]
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(processed / "polls.parquet", index=False)
    return processed


def _forecast(tmp_path: Path, counts: dict | None) -> Path:
    path = tmp_path / "forecast.json"
    body = {"updated": "2026-08-16"}
    if counts is not None:
        body["counts"] = counts
    path.write_text(json.dumps(body))
    return path


def _run(tmp_path, spine_polls, live_counts):
    guard = _load_guard()
    guard.PROCESSED_DIR = _spine(tmp_path, spine_polls)
    guard.FORECAST_JSON = _forecast(tmp_path, live_counts)
    return guard.main()


IN_CYCLE = str(CYCLE_2026.start)


def test_new_poll_sharing_a_field_end_is_detected(tmp_path):
    """Regression: Sifo and Novus both ended fieldwork 2026-08-16. The old
    date-based guard saw an unchanged max date and skipped the Novus poll
    entirely — the nightly job would never have refit."""
    exit_code = _run(
        tmp_path,
        [("sifo", "2026-08-16", 3043.0), ("novus", "2026-08-16", 5617.0)],
        {"cycle_polls": 1},   # live forecast was published with Sifo only
    )
    assert exit_code == 0


def test_backfilled_older_poll_is_detected(tmp_path):
    """A lagging house publishing a poll with an OLDER field-end than the
    newest one already in the fit also left the max date untouched."""
    exit_code = _run(
        tmp_path,
        [("newest", "2026-08-16", 3000.0), ("late_arrival", "2026-08-02", 1800.0)],
        {"cycle_polls": 1},
    )
    assert exit_code == 0


def test_nothing_new_skips(tmp_path):
    exit_code = _run(tmp_path, [("a", "2026-08-16", 3000.0)], {"cycle_polls": 1})
    assert exit_code == 10


def test_polls_without_sample_size_are_not_counted(tmp_path):
    """model.py drops polls with no n, so the guard must not count them —
    otherwise it triggers a refit that changes nothing."""
    exit_code = _run(
        tmp_path,
        [("a", "2026-08-16", 3000.0), ("no_n", "2026-08-18", None)],
        {"cycle_polls": 1},
    )
    assert exit_code == 10


def test_polls_before_the_cycle_are_not_counted(tmp_path):
    exit_code = _run(
        tmp_path,
        [("a", "2026-08-16", 3000.0), ("old", "2019-05-05", 1000.0)],
        {"cycle_polls": 1},
    )
    assert exit_code == 10


def test_shrinking_spine_skips_rather_than_refitting(tmp_path):
    """Upstream revision/retraction: fewer polls than published. Not new data,
    and not something to publish automatically."""
    exit_code = _run(tmp_path, [("a", "2026-08-16", 3000.0)], {"cycle_polls": 5})
    assert exit_code == 10


def test_forecast_without_cycle_polls_is_undetermined(tmp_path):
    """A forecast.json predating the count must not be guessed at."""
    assert _run(tmp_path, [("a", "2026-08-16", 3000.0)], {"polls": 2636}) == 2
    assert _run(tmp_path, [("a", "2026-08-16", 3000.0)], None) == 2


def test_missing_spine_is_undetermined(tmp_path):
    guard = _load_guard()
    guard.PROCESSED_DIR = tmp_path / "empty"
    (tmp_path / "empty").mkdir()
    guard.FORECAST_JSON = _forecast(tmp_path, {"cycle_polls": 1})
    assert guard.main() == 2


def test_live_forecast_records_the_count_the_model_fits():
    """Integration: the published forecast's cycle_polls must equal what the
    guard computes from the real spine, or the nightly job mis-triggers."""
    parquet = REPO_ROOT / "data" / "processed" / "polls.parquet"
    forecast = REPO_ROOT / "web" / "src" / "data" / "forecast.json"
    if not (parquet.exists() and forecast.exists()):
        pytest.skip("real spine / forecast not built")
    guard = _load_guard()
    live = (json.loads(forecast.read_text()).get("counts") or {}).get("cycle_polls")
    if live is None:
        pytest.skip("forecast.json predates cycle_polls")
    assert guard.spine_cycle_polls() == live
