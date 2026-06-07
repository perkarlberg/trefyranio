"""Per-valkrets seat origins for the seat-origins map (visualization only).

National-proportional allocation is already EXACT (see allocator + the regression
test): the leveling seats make the outcome nationally proportional, so the headline
seat totals need no per-constituency step. This module is purely for the *map* —
showing WHERE each party's and bloc's seats come from across the 29 constituencies.

Method: take the live central forecast's exact national seat totals, then distribute
each party's seats across constituencies by **biproportional apportionment** with
weights from a **uniform-swing** projection of each valkrets's 2022 result. Two
honest caveats, both documented on the page:
  * uniform swing — regional swings aren't uniform, so per-valkrets detail is an
    approximation (the national per-party totals are exact);
  * biproportional ≠ Sweden's exact fixed+leveling rule, but reproduces real 2022
    per-valkrets seats to within ~8 of 349 (validated in tests).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trefyranio.allocator import allocate_national, biproportional, first_divisor_for_year
from trefyranio.etl.schema import RIKSDAG_PARTIES

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
BASELINE_YEAR = 2022

# Blocs mirror simulate.py / model._MISS_GROUP.
BLOC = {"S": "left", "V": "left", "MP": "left",
        "M": "right", "SD": "right", "KD": "right", "L": "right",
        "C": "centre"}


def uniform_swing(base_share: np.ndarray, nat_base: np.ndarray, nat_now: np.ndarray) -> np.ndarray:
    """Apply the national swing (nat_now − nat_base) uniformly to each valkrets's
    baseline shares; clip at 0 and renormalize per valkrets. base_share is
    (V, P)."""
    swung = np.clip(base_share + (nat_now - nat_base)[None, :], 0.0, None)
    return swung / swung.sum(axis=1, keepdims=True)


def valkrets_seat_matrix(national_shares: dict[str, float], year: int = 2026):
    """Return (valkrets_codes, names, parties, seat_matrix V×P) for the central
    forecast, via uniform swing + biproportional apportionment.

    ``national_shares`` is the central forecast's national vote share per party
    (the 8 Riksdag parties + Övr). Column targets are the EXACT national seat
    totals from allocate_national, so per-party map totals match the headline."""
    rv = pd.read_parquet(PROCESSED_DIR / "results_valkrets.parquet")
    sa = pd.read_parquet(PROCESSED_DIR / "seats_actual.parquet")
    res = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    parties = list(RIKSDAG_PARTIES)  # 8 Riksdag parties

    base = rv[rv.election_year == BASELINE_YEAR]
    sav = sa[(sa.election_year == BASELINE_YEAR) & (sa.region_code != "VR00")]
    vcodes = sorted(sav.region_code.unique())
    names = (base.drop_duplicates("valkrets_code")
             .set_index("valkrets_code")["valkrets_name"].to_dict())

    # Per-valkrets 2022 baseline: shares (V×P) and electorate size (≈ valid votes).
    pivot_v = (base[base.party.isin(parties)]
               .pivot_table(index="valkrets_code", columns="party", values="votes",
                            aggfunc="sum").reindex(index=vcodes, columns=parties).fillna(0.0))
    size = pivot_v.to_numpy().sum(axis=1)                      # (V,) valkrets size
    base_share = pivot_v.to_numpy() / size[:, None]

    nat_base = np.array([res[(res.election_year == BASELINE_YEAR) & (res.party == p)]
                         .share.iloc[0] for p in parties])
    nat_now = np.array([national_shares.get(p, 0.0) for p in parties])

    swung = uniform_swing(base_share, nat_base, nat_now)        # (V, P)
    weights = swung * size[:, None]                            # 2026 vote-count estimate

    # Column targets: exact national seats from the central forecast.
    nat = allocate_national({**{p: national_shares.get(p, 0.0) for p in parties},
                             "Övr": national_shares.get("Övr", 0.0)},
                            first_divisor=first_divisor_for_year(year),
                            ignore_parties=frozenset({"Övr"}))
    col_t = np.array([nat.seats.get(p, 0) for p in parties])
    # Row targets: valkrets seat budgets (2022 totals ≈ 2026; sizes change slowly).
    row_t = np.array([int(sav[sav.region_code == v].seats.sum()) for v in vcodes])

    M = biproportional(weights, row_t, col_t)
    return vcodes, names, parties, M


def build(national_shares: dict[str, float]) -> dict:
    """Assemble the seat-origins payload (per-party + per-bloc seats per valkrets)
    for the web map."""
    vcodes, names, parties, M = valkrets_seat_matrix(national_shares)
    out = []
    for i, v in enumerate(vcodes):
        per_party = {p: int(M[i, j]) for j, p in enumerate(parties) if M[i, j] > 0}
        per_bloc = {"left": 0, "right": 0, "centre": 0}
        for p, s in per_party.items():
            per_bloc[BLOC[p]] += s
        out.append({"code": v, "name": names.get(v, v),
                    "seats": int(M[i].sum()), "parties": per_party, "blocs": per_bloc})
    totals = {p: int(M[:, j].sum()) for j, p in enumerate(parties)}
    return {"valkrets": out, "party_totals": {p: s for p, s in totals.items() if s > 0}}
