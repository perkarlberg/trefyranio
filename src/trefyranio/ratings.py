"""Pollster ratings, house effects, and industry bias.

Joins the two data spines — the poll history (`polls.parquet`) and official
results (`results_national.parquet`) — to answer three questions from each
pollster's *final* pre-election polls vs the actual outcome:

* **House effect**  — does a pollster systematically lean toward/against a party?
  (signed error, shrunk toward zero). Subtracted from polls in aggregation.
* **Industry bias** — does the *whole field* miss a party in one direction?
  (e.g. the persistent underestimate of SD). A correction the model applies on
  top of per-pollster house effects.
* **Accuracy / weight** — how close is each pollster, relative to the field in
  the same elections (a "plus-minus", shrunk by how many elections they cover),
  turned into an aggregation weight.

This is the descriptive, out-of-model layer: it produces priors and weights and
a human-readable scorecard. The Bayesian model (Phase 3) re-estimates house
effects *jointly* against the consensus trend; these are the sanity-check and
the warm-start.

Scope: elections 2010-2022 — the current eight-party system with dense polling.
Earlier elections (sparse SD polling, a different party landscape) are excluded
on purpose; widen ELECTION_DATES to extend coverage.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from trefyranio.etl.schema import RIKSDAG_PARTIES

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Riksdagsval election days (second Sunday of September).
ELECTION_DATES = {
    2010: dt.date(2010, 9, 19),
    2014: dt.date(2014, 9, 14),
    2018: dt.date(2018, 9, 9),
    2022: dt.date(2022, 9, 11),
}
FINAL_WINDOW_DAYS = 30   # a pollster's "final" polls = those within 30d of E-day
SHRINK_K = 1.5           # shrinkage strength (in "elections covered")


def final_poll_errors(polls: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per (pollster, election, party): the pollster's averaged final
    poll share, the actual result, and the signed/absolute error."""
    polls = polls.copy()
    polls["date"] = polls["field_end"].fillna(polls["pub_date"])
    polls = polls[polls["party"].isin(RIKSDAG_PARTIES)]

    rows = []
    for year, eday in ELECTION_DATES.items():
        eday = pd.Timestamp(eday)
        window = polls[
            (polls["date"] <= eday)
            & (polls["date"] >= eday - pd.Timedelta(days=FINAL_WINDOW_DAYS))
        ]
        if window.empty:
            continue
        avg = (
            window.groupby(["pollster", "party"])
            .agg(poll_share=("share", "mean"), n_polls=("poll_id", "nunique"))
            .reset_index()
        )
        avg["election_year"] = year
        rows.append(avg)

    errs = pd.concat(rows, ignore_index=True)
    actual = (
        results[results["election_year"].isin(ELECTION_DATES)]
        [["election_year", "party", "share"]]
        .rename(columns={"share": "actual_share"})
    )
    errs = errs.merge(actual, on=["election_year", "party"], how="inner")
    errs["error"] = errs["poll_share"] - errs["actual_share"]      # + => overstates
    errs["abs_error"] = errs["error"].abs()
    return errs


def house_effects(errs: pd.DataFrame) -> pd.DataFrame:
    """Per (pollster, party) signed bias, shrunk toward zero by elections seen."""
    g = errs.groupby(["pollster", "party"]).agg(
        raw=("error", "mean"), n_elections=("election_year", "nunique")
    ).reset_index()
    g["house_effect"] = g["raw"] * g["n_elections"] / (g["n_elections"] + SHRINK_K)
    return g[["pollster", "party", "house_effect", "n_elections", "raw"]]


def industry_bias(errs: pd.DataFrame) -> pd.DataFrame:
    """Field-wide signed bias per party (negative => the field understates it).

    Averaged over pollster-means first so prolific pollsters don't dominate."""
    per_pollster = errs.groupby(["party", "pollster"])["error"].mean().reset_index()
    bias = per_pollster.groupby("party")["error"].mean().rename("bias").reset_index()
    bias = bias.set_index("party").reindex(RIKSDAG_PARTIES).reset_index()
    return bias


def pollster_ratings(errs: pd.DataFrame) -> pd.DataFrame:
    """Accuracy relative to the field, shrunk, turned into a weight.

    For each (pollster, election) the error is the mean absolute party error.
    Its "plus-minus" is that minus the field's mean for the same election (so a
    hard election doesn't penalize anyone). Averaged over elections and shrunk
    toward the field (0), then mapped to a weight centered on 1.0.
    """
    pe = errs.groupby(["pollster", "election_year"])["abs_error"].mean().rename(
        "poll_err"
    ).reset_index()
    field = pe.groupby("election_year")["poll_err"].mean().rename("field_err")
    pe = pe.merge(field, on="election_year")
    pe["plus_minus"] = pe["poll_err"] - pe["field_err"]  # negative => better

    g = pe.groupby("pollster").agg(
        n_elections=("election_year", "nunique"),
        mean_abs_error=("poll_err", "mean"),
        raw_plus_minus=("plus_minus", "mean"),
    ).reset_index()
    g["plus_minus"] = g["raw_plus_minus"] * g["n_elections"] / (g["n_elections"] + SHRINK_K)

    field_global = pe["poll_err"].mean()
    g["predictive_error"] = field_global + g["plus_minus"]
    g["weight"] = field_global / g["predictive_error"]
    g["weight"] = g["weight"] / g["weight"].mean()  # center on 1.0
    return g.sort_values("predictive_error").reset_index(drop=True)


def build() -> dict[str, pd.DataFrame]:
    polls = pd.read_parquet(PROCESSED_DIR / "polls.parquet")
    results = pd.read_parquet(PROCESSED_DIR / "results_national.parquet")

    errs = final_poll_errors(polls, results)
    outputs = {
        "pollster_house_effects": house_effects(errs),
        "industry_bias": industry_bias(errs),
        "pollster_ratings": pollster_ratings(errs),
    }
    for name, df in outputs.items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    _scorecard(errs, outputs)
    return outputs


def _scorecard(errs: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> None:
    yrs = sorted(errs["election_year"].unique())
    print(f"pollster ratings from final-{FINAL_WINDOW_DAYS}d polls, elections {yrs}")

    print("\n=== accuracy (lower error = better; weight centered on 1.0) ===")
    print(f"{'pollster':12} {'elec':>4} {'MAE pp':>7} {'+/- pp':>7} {'weight':>7}")
    for _, r in outputs["pollster_ratings"].iterrows():
        print(f"{r['pollster']:12} {r['n_elections']:>4.0f} "
              f"{r['mean_abs_error']*100:>7.2f} {r['plus_minus']*100:>+7.2f} "
              f"{r['weight']:>7.2f}")

    print("\n=== industry bias (negative pp => field UNDERSTATES the party) ===")
    for _, r in outputs["industry_bias"].iterrows():
        if pd.notna(r["bias"]):
            print(f"  {r['party']:3} {r['bias']*100:>+6.2f} pp")


if __name__ == "__main__":
    build()
