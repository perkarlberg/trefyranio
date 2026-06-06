"""Export the forecast artifacts to web-ready JSON (Phase 6).

Reads the pipeline outputs in data/processed/ and writes the JSON contract the
Astro site consumes into web/src/data/. Run after `simulate`:

    python -m trefyranio.web_export

Produces:
  forecast.json    — government outlook, per-party shares+seats+threshold, coalitions
  trend.json       — weekly poll/model trend per party (the momentum chart)
  seat_draws.json  — per-draw seats (for the client-side interactive coalition builder)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trefyranio.allocator import allocate_national, first_divisor_for_year
from trefyranio.etl.schema import OTHER, RIKSDAG_PARTIES
from trefyranio.model import CYCLE_2026, ELECTION_DATES, PARTY_ORDER
from trefyranio.parties import PARTY_HEX, PARTY_NAME_SV, POLLSTER_METHOD
from trefyranio.simulate import CENTRE, COALITIONS, ELECTION_YEAR, LEADERS, LEFT, MAJORITY, RIGHT

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
WEB_DATA_DIR = REPO_ROOT / "web" / "src" / "data"

_BLOC = {**{p: "right" for p in RIGHT}, **{p: "left" for p in LEFT},
         **{p: "centre" for p in CENTRE}}


def _round(x, n=4):
    return round(float(x), n)


def build_forecast(as_of: str, weeks_to_go: int, counts: dict, latest_poll: dict) -> dict:
    npz = np.load(PROCESSED_DIR / "forecast_samples.npz", allow_pickle=True)
    shares = npz["shares"]                       # (draws, K) in PARTY_ORDER
    seat_df = pd.read_parquet(PROCESSED_DIR / "seat_forecast.parquet").set_index("party")
    coal = pd.read_parquet(PROCESSED_DIR / "coalition_forecast.parquet")
    gov = json.loads((PROCESSED_DIR / "government_forecast.json").read_text())

    # Central parliament: allocate the MEAN forecast shares through the real
    # system ONCE. This respects the 4% gate (a party below 4% gets 0), unlike
    # averaging per-draw seats — which would show a coherent-looking but
    # impossible allocation for a bimodal near-threshold party (e.g. L "4 seats").
    central_votes = {PARTY_ORDER[k]: float(shares[:, k].mean()) for k in range(len(PARTY_ORDER))}
    central = allocate_national(
        central_votes, first_divisor=first_divisor_for_year(ELECTION_YEAR),
        ignore_parties=frozenset({OTHER}),
    ).seats

    parties = []
    for k, code in enumerate(PARTY_ORDER):
        s = shares[:, k]
        row = {
            "code": code,
            "name": PARTY_NAME_SV[code],
            "color": PARTY_HEX[code],
            "bloc": _BLOC.get(code),
            "share_mean": _round(s.mean()),
            "share_lo": _round(np.quantile(s, 0.1)),
            "share_hi": _round(np.quantile(s, 0.9)),
            "central_seats": int(central.get(code, 0)),  # coherent central parliament
        }
        if code in seat_df.index:
            r = seat_df.loc[code]
            row.update(
                seats_mean=_round(r["mean_seats"], 1),
                seats_lo=int(r["lo"]), seats_hi=int(r["hi"]),
                p_in_riksdag=_round(r["p_in_riksdag"], 3),
                p_below_4pct=_round(r["p_below_4pct"], 3),
            )
        parties.append(row)
    parties.sort(key=lambda p: p.get("seats_mean", 0), reverse=True)

    coalitions = [
        {"name": r["coalition"], "parties": r["parties"].split("+"),
         "p_majority": _round(r["p_majority"], 3), "mean_seats": _round(r["mean_seats"], 1),
         "lo": int(r["lo"]), "hi": int(r["hi"])}
        for _, r in coal.iterrows()
    ]
    return {
        "updated": as_of,
        "election_day": gov.get("election_day", "2026-09-13"),
        "weeks_to_go": weeks_to_go,
        "majority": MAJORITY,
        "counts": counts,            # {polls, elections, sims}
        "latest_poll": latest_poll,  # {pollster, date}
        "government": {
            "right": gov["p_right_majority"],
            "kingmaker": gov["p_centre_kingmaker"],
            "left": gov["p_left_majority"],
            "leaders": LEADERS,
            "blocs": {"right": RIGHT, "left": LEFT, "centre": CENTRE},
        },
        "parties": parties,
        "coalitions": coalitions,
        "swing_drama": gov["swing_drama"],
    }


def build_trend() -> dict:
    df = pd.read_parquet(PROCESSED_DIR / "model_trend.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    dates = sorted(df["date"].unique())
    series = {}
    for code in PARTY_ORDER:
        sub = df[df["party"] == code].set_index("date").reindex(dates)
        series[code] = {
            "color": PARTY_HEX[code], "name": PARTY_NAME_SV[code],
            "mean": [_round(v, 4) for v in sub["mean"]],
            "lo": [_round(v, 4) for v in sub["lo"]],
            "hi": [_round(v, 4) for v in sub["hi"]],
        }
    return {"dates": dates, "series": series}


# Methodology-based quality notes for pollsters with no election-eve accuracy data.
QUALITY_NOTE = {
    "SCB": "Officiell statistik – stort slumpmässigt urval, högsta metodkvalitet",
    "Indikator": "Postal slumpmässig enkät – god metodkvalitet",
}


def _tier(mae_pp: float) -> str:
    if mae_pp < 1.0:
        return "Hög träffsäkerhet"
    if mae_pp < 1.3:
        return "God träffsäkerhet"
    return "Lägre träffsäkerhet"


def build_polls_detail() -> dict:
    """Per-poll detail for the trend overlay, recent-polls table and poll modal:
    raw vs house-effect-adjusted shares + per-pollster quality/bias."""
    polls = pd.read_parquet(PROCESSED_DIR / "polls.parquet")
    polls = polls[polls["party"].isin(RIKSDAG_PARTIES)].copy()
    polls["date"] = polls["field_end"].fillna(polls["pub_date"])
    polls = polls[polls["date"].dt.date >= CYCLE_2026.start]

    # House effect = consensus-relative lean: each poll vs the model trend on that
    # date, averaged per pollster per party (the standard 538 definition). Works
    # for EVERY pollster that polls at any time — including SCB (May PSU) and
    # Indikator, which the election-eve-error method can't score.
    trend = pd.read_parquet(PROCESSED_DIR / "model_trend.parquet")
    trend["date"] = pd.to_datetime(trend["date"])
    twide = trend.pivot_table(index="date", columns="party", values="mean").sort_index()
    devs = {}  # (pollster, party) -> list of raw - trend
    for pid, g in polls.groupby("poll_id"):
        meta = g.iloc[0]
        ti = twide.index.get_indexer([meta["date"]], method="nearest")[0]
        trow = twide.iloc[ti]
        for _, r in g.iterrows():
            devs.setdefault((meta["pollster"], r["party"]), []).append(r["share"] - trow[r["party"]])
    house = {k: float(np.mean(v)) for k, v in devs.items()}
    ratings = pd.read_parquet(PROCESSED_DIR / "pollster_ratings.parquet").set_index("pollster")

    records = []
    for pid, g in polls.groupby("poll_id"):
        meta = g.iloc[0]
        raw = {r["party"]: _round(r["share"]) for _, r in g.iterrows()}
        adj = {p: _round(raw[p] - house.get((meta["pollster"], p), 0.0)) for p in raw}
        records.append({
            "date": meta["date"].strftime("%Y-%m-%d"),
            "pollster": meta["pollster"],
            "n": None if pd.isna(meta["n"]) else int(meta["n"]),
            "method": POLLSTER_METHOD.get(meta["pollster"], "—"),
            "raw": raw, "adjusted": adj,
        })
    records.sort(key=lambda r: r["date"])

    pollsters = {}
    for name in sorted({r["pollster"] for r in records}):
        info = {"method": POLLSTER_METHOD.get(name, "—"),
                "house_bias": {p: _round(house.get((name, p), 0.0)) for p in RIKSDAG_PARTIES}}
        if name in ratings.index:
            mae_pp = float(ratings.loc[name, "mean_abs_error"]) * 100
            info.update(mae_pp=round(mae_pp, 2), weight=_round(ratings.loc[name, "weight"], 2),
                        tier=_tier(mae_pp), n_elections=int(ratings.loc[name, "n_elections"]))
        else:
            # No final-30d-before-election polls to score (e.g. SCB's May PSU,
            # Indikator) — give a methodology-based quality note instead.
            info["note"] = QUALITY_NOTE.get(
                name, "Träffsäkerhet ej beräknad – för få mätningar nära tidigare val.")
        pollsters[name] = info
    return {"polls": records, "pollsters": pollsters}


def build_seat_draws() -> dict:
    npz = np.load(PROCESSED_DIR / "seat_draws.npz", allow_pickle=True)
    return {
        "parties": list(RIKSDAG_PARTIES),
        "colors": {p: PARTY_HEX[p] for p in RIKSDAG_PARTIES},
        "majority": MAJORITY,
        "draws": npz["seats"].astype(int).tolist(),   # (draws, 8)
    }


def build() -> None:
    polls = pd.read_parquet(PROCESSED_DIR / "polls.parquet")
    polls["date"] = polls["field_end"].fillna(polls["pub_date"])
    as_of_ts = polls["date"].max()
    weeks_to_go = int((pd.Timestamp("2026-09-13") - as_of_ts).days // 7)
    as_of = as_of_ts.strftime("%Y-%m-%d")

    # Exact provenance counts for the homepage line. n_polls = the full poll
    # archive the project is built on (the model aggregates the current cycle;
    # the full history feeds the pollster ratings).
    cycle = polls[(polls["date"].dt.date >= CYCLE_2026.start) & polls["n"].notna()]
    n_polls = int(polls["poll_id"].nunique())
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")
    n_elections = int(results["election_year"].nunique())
    gov = json.loads((PROCESSED_DIR / "government_forecast.json").read_text())
    n_sim = int(gov.get("n_sim", 0))
    latest = cycle.loc[cycle["date"].idxmax()]
    counts = {"polls": n_polls, "elections": n_elections, "sims": n_sim}
    latest_poll = {"pollster": str(latest["pollster"]), "date": latest["date"].strftime("%Y-%m-%d")}

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forecast.json": build_forecast(as_of, weeks_to_go, counts, latest_poll),
        "trend.json": build_trend(),
        "polls.json": build_polls_detail(),
        "seat_draws.json": build_seat_draws(),
    }
    for name, obj in outputs.items():
        path = WEB_DATA_DIR / name
        path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        print(f"wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
