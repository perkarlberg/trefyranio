"""Ingest official Riksdag election results from SCB (Statistics Sweden).

Source: SCB PXWeb API, subject ME0104 ("Allmänna val, valresultat"), CC0.
Three tables under ME0104C (Riksdagsval):

* ``ME0104T3``     — votes per region & party, 1973-2022 (ground-truth labels)
* ``Riksdagsmandat`` — seats won per region & party, 1973-2022 (allocator check)
* (turnout is derived from the OGILTIGA / VALSKOLKARE rows of ME0104T3)

Region ``VR00`` is the national total; ``VR1``..``VR29`` are the 29 current
constituencies. These results are the labels we score polls against (Phase 2)
and backtest the model on (Phase 5).

API docs: https://www.scb.se/vara-tjanster/oppna-data/api-for-statistikdatabasen/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from trefyranio.etl.schema import (
    CURRENT_VALKRETSAR,
    NATIONAL_REGION,
    RESULT_PARTIES,
    SCB_INVALID,
    SCB_NONVOTERS,
    SCB_PARTY_MAP,
)

BASE = "https://api.scb.se/OV0104/v1/doris/sv/ssd/ME/ME0104/ME0104C"
SOURCE = "SCB:ME0104"
_UA = {"User-Agent": "trefyranio/0.1 (election-model research)"}

# Each table names its party variable differently.
RESULTS_TABLE = "ME0104T3"
RESULTS_PARTY_VAR = "Partimm"
RESULTS_CONTENT = "ME0104B6"  # Antal röster (vote counts)
MANDAT_TABLE = "Riksdagsmandat"
MANDAT_PARTY_VAR = "Parti"
MANDAT_CONTENT = "ME0104C3"  # Mandat i riksdagen


def _post(table: str, party_var: str, content: str, regions: list[str]) -> pd.DataFrame:
    """Query a ME0104C table for the given regions, all parties, all years.

    Returns a frame with columns [region, scb_party, year, value].
    """
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "item", "values": regions}},
            {"code": party_var, "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": [content]}},
            {"code": "Tid", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json"},
    }
    resp = requests.post(f"{BASE}/{table}", json=query, timeout=60, headers=_UA)
    resp.raise_for_status()
    rows = []
    for rec in resp.json()["data"]:
        region, party, year = rec["key"]
        raw = rec["values"][0]
        value = pd.NA if raw in ("..", ".", "") else int(raw)
        rows.append((region, party, int(year), value))
    return pd.DataFrame(rows, columns=["region", "scb_party", "year", "value"])


def _region_names() -> dict[str, str]:
    meta = requests.get(f"{BASE}/{RESULTS_TABLE}", timeout=30, headers=_UA).json()
    region = next(v for v in meta["variables"] if v["code"] == "Region")
    return dict(zip(region["values"], region["valueTexts"]))


def fetch_national() -> tuple[pd.DataFrame, pd.DataFrame]:
    """National results -> (party_results, turnout_meta).

    ``party_results``: election_year, party, votes, share (of valid votes).
    ``turnout_meta``:   election_year, valid_votes, invalid_votes, non_voters,
                        eligible, turnout.
    """
    raw = _post(RESULTS_TABLE, RESULTS_PARTY_VAR, RESULTS_CONTENT, [NATIONAL_REGION])

    parties = raw[raw["scb_party"].isin(SCB_PARTY_MAP)].copy()
    parties["party"] = parties["scb_party"].map(SCB_PARTY_MAP)
    parties = (
        parties.groupby(["year", "party"], as_index=False)["value"].sum()
        .rename(columns={"year": "election_year", "value": "votes"})
    )
    valid = parties.groupby("election_year")["votes"].sum().rename("valid_votes")
    parties = parties.merge(valid, on="election_year")
    parties["share"] = parties["votes"] / parties["valid_votes"]
    parties = parties.drop(columns="valid_votes")

    pivot = raw.pivot_table(index="year", columns="scb_party", values="value", aggfunc="sum")
    meta = pd.DataFrame({
        "election_year": pivot.index,
        "valid_votes": valid.reindex(pivot.index).values,
        "invalid_votes": pivot.get(SCB_INVALID),
        "non_voters": pivot.get(SCB_NONVOTERS),
    })
    cast = (
        meta["valid_votes"] + meta["invalid_votes"] + meta["non_voters"]
    )
    meta["eligible"] = cast
    meta["turnout"] = (meta["valid_votes"] + meta["invalid_votes"]) / meta["eligible"]

    parties = _order(parties, ["election_year", "party"])
    return parties.reset_index(drop=True), meta.reset_index(drop=True)


def fetch_valkrets() -> pd.DataFrame:
    """Per-constituency results for the 29 current valkretsar:
    election_year, valkrets_code, valkrets_name, party, votes, share."""
    raw = _post(RESULTS_TABLE, RESULTS_PARTY_VAR, RESULTS_CONTENT, CURRENT_VALKRETSAR)
    raw = raw[raw["scb_party"].isin(SCB_PARTY_MAP) & raw["value"].notna()].copy()
    raw["party"] = raw["scb_party"].map(SCB_PARTY_MAP)
    df = (
        raw.groupby(["year", "region", "party"], as_index=False)["value"].sum()
        .rename(columns={"year": "election_year", "region": "valkrets_code", "value": "votes"})
    )
    valid = (
        df.groupby(["election_year", "valkrets_code"])["votes"].sum().rename("valid_votes")
    )
    df = df.merge(valid, on=["election_year", "valkrets_code"])
    df["share"] = df["votes"] / df["valid_votes"]
    df = df.drop(columns="valid_votes")
    names = _region_names()
    df["valkrets_name"] = df["valkrets_code"].map(names)
    df = df[["election_year", "valkrets_code", "valkrets_name", "party", "votes", "share"]]
    return _order(df, ["election_year", "valkrets_code", "party"]).reset_index(drop=True)


def fetch_seats() -> pd.DataFrame:
    """Actual seats won: election_year, region_code, region_name, party, seats."""
    regions = [NATIONAL_REGION] + CURRENT_VALKRETSAR
    raw = _post(MANDAT_TABLE, MANDAT_PARTY_VAR, MANDAT_CONTENT, regions)
    raw = raw[raw["scb_party"].isin(SCB_PARTY_MAP) & raw["value"].notna()].copy()
    raw["party"] = raw["scb_party"].map(SCB_PARTY_MAP)
    df = (
        raw.groupby(["year", "region", "party"], as_index=False)["value"].sum()
        .rename(columns={"year": "election_year", "region": "region_code", "value": "seats"})
    )
    df = df[df["seats"] > 0]
    names = _region_names()
    df["region_name"] = df["region_code"].map(names)
    df = df[["election_year", "region_code", "region_name", "party", "seats"]]
    return _order(df, ["election_year", "region_code", "party"]).reset_index(drop=True)


def _order(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    """Sort with party in canonical order rather than alphabetical."""
    if "party" in df.columns:
        df = df.copy()
        df["__p"] = pd.Categorical(df["party"], categories=RESULT_PARTIES, ordered=True)
        sort_cols = [c if c != "party" else "__p" for c in sort_cols]
        df = df.sort_values(sort_cols).drop(columns="__p")
    else:
        df = df.sort_values(sort_cols)
    return df
