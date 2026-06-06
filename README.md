# trefyranio

A [FiveThirtyEight](https://en.wikipedia.org/wiki/FiveThirtyEight)-style
forecasting model and webapp for the Swedish parliamentary election
(riksdagsval).

**The name:** `trefyran.io` reads as **tre · fyra · n-io** → "tre fyra nio" →
**3 · 4 · 9** → the **349 seats** of the Riksdag. The `.io` TLD *is* the final
syllable. Where FiveThirtyEight encodes 538 US electoral votes, trefyranio
encodes 349 Swedish seats.

## What it does

Aggregates Swedish opinion polls into a dynamic Bayesian estimate of party
vote shares, then runs tens of thousands of Monte Carlo simulations — each one
passed through the real Swedish electoral system (4% / 12% threshold → modified
Sainte-Laguë seat allocation) — to produce probabilistic forecasts of:

- seats per party (with the **4% cliff** producing bimodal distributions for
  parties near threshold — the central drama of a Swedish forecast),
- bloc majorities and **government-formation probabilities** (the real prize:
  who can govern, not just who gets the most votes),
- pivotal-party analysis (which small party crossing 4% flips the majority).

## Methodology

We port the **Economist / Gelman dynamic Bayesian model** (the only major
forecaster with fully published, implementable formulas — Silver's exact
weights and correlation matrix are proprietary), adapted for proportional
representation:

- **Top-down** national 8-party vote-share random walk in additive-log-ratio
  space, plus correlated valkrets (constituency) deviations. Top-down because
  Swedish swing is national-dominated and constituency polling is thin.
- **Beta-Binomial** poll likelihood (absorbs nonsampling error).
- Per-pollster **house effects** (zero-sum across parties), shrunk to zero.
- **Pollster ratings** computed from scratch: historical error vs official
  results, shrunk toward a global mean. SCB/PSU weighted as the benchmark.
- Weak **fundamentals prior** on the election-day node, dominated by the
  lagged result (Swedish vote shares are persistent).

**Swedish-specific tweaks:** the 4% national / 12% constituency threshold;
government-formation as the headline output; correction for the persistent
~2-3pp industry-wide **underestimate of SD**; the Novus 2023 phone→web
methodology break; and flagging *stödröstning* (vote-lending to keep small
allies above 4%), which polls capture poorly.

## Uncertainty model (and an honest limitation)

The forecast's spread is a **calibrated, horizon-dependent add-on — not emergent
from the Bayesian latent walk.** Swedish national vote shares move slowly between
elections, so the random walk's innovation variance is estimated near zero; left
alone it would project to election day with false certainty (the Taleb/martingale
trap — ±0.1pp a year out). So the real election-day error is injected as an
explicit **polling-miss term** in share space, then propagated through the simulator.

- **Calibrated, not guessed** — sized by backtesting 2018 & 2022 from polls-only
  so interval coverage matches realized poll-vs-result error (≈88%).
- **Horizon-dependent** — calibrated at two horizons (election-eve and 14 weeks
  out): `σ(H) = √(1.6² + 0.18·H) pp`, H = weeks from the last poll to election. The
  forecast is wider far out and **tightens as the election nears** (`MISS_SIGMA_*`,
  `miss_sigma_for_horizon` in `model.py`).
- **Correlated within blocs** — a factor model gives within-bloc co-movement
  (`MISS_RHO`≈0.2) so bloc-total / government-formation variance isn't understated.

**The limitation, owned:** a hand-fit term carries the forecast error rather than
the generative model itself. The principled fix — a backward-from-election-day
random walk with horizon-accumulating innovations + an explicit election-day
fundamentals prior (the Economist approach), so the spread *emerges* from the
model — is on the roadmap. Until then the add-on is calibrated, horizon-aware, and
transparent rather than hidden.

## Electoral system (encoded in `allocator.py`)

- 349 seats = 310 fixed constituency seats + 39 leveling seats
  (utjamningsmandat), making the result nationally proportional above threshold.
- 29 constituencies (valkretsar).
- Threshold: ≥4% nationally **or** ≥12% in one constituency.
- Modified Sainte-Laguë (jämkade uddatalsmetoden), **first divisor 1.2** (since
  2018; was 1.4), then 3, 5, 7, …

## Stack

- **Model / ETL:** Python + NumPyro (Bayesian model), pandas (ETL).
- **Web:** Astro + islands (static, SEO-first), D3 / Observable Plot charts.
- **Compute:** runs **locally** (no cloud compute). The pipeline produces static
  JSON artifacts; refresh on demand with `./daily_update.sh`.
- **Hosting:** **Firebase Hosting** (free Spark tier), deployed via the Hosting
  REST API with a `gcloud` token (`deploy.sh` → `deploy_hosting.py`, no firebase
  CLI / no service-account keys). Live at `trefyran.io`. The webapp computes
  nothing on page load — it serves precomputed forecasts.

## Data spine

| Layer | Source | License |
|---|---|---|
| Results (valdistrikt, 2006–2022) | Valmyndigheten `data.val.se` | free + attribution |
| Results (kommun/valkrets, 1973+) | SCB PXWeb API (ME0104) | CC0 |
| Polls (per-poll, 2008+ clean) | `MansMeg/SwedishPolls` | CC0 |
| Polls (current cycle, all 8 pollsters) | Wikipedia 2026 polling table | CC-BY-SA |
| Polls (benchmark) | SCB PSU (ME0201), direct API | CC0 |
| Demographics | SCB DeSO/RegSO (PXWeb + WFS) | CC0 |
| Exit-poll crosstabs | VALU (SVT/GU) — **PDF open, microdata gated at SND** | restricted |
| Geo (map) | Valmyndigheten valdistrikt GeoJSON | free + attribution |

Gated pieces (VALU microdata, commercial pollster crosstabs) are phase-2
enrichments requiring outreach to GU / pollsters — not blockers for v1.

## Roadmap

- [x] **Phase 0** — scaffold + deterministic seat allocator, validated against
      2022 actuals.
- [~] **Phase 1** — data spine ETL into parquet.
  - [x] Poll spine: SwedishPolls → `polls.parquet` (2,625 polls, 1980–2026, all 8
        current pollsters + SCB PSU; tidy long, simplex-normalized).
  - [x] Official results: SCB ME0104 → `results_national`, `results_valkrets`,
        `results_meta` (turnout), `seats_actual` (1973–2022). National + 29
        valkretsar + actual seats. FP→L mapping; year-aware first divisor
        (1.4 pre-2018, 1.2 from 2018). Allocator reproduces 2018 & 2022 exactly.
  - [ ] Per-constituency allocator (310 fixed per valkrets + 39 leveling) — for
        exact historical backtests; pure national-proportional already nails the
        current-rule elections, deviates ≤3 seats on 1988/2010/2014.
  - [ ] val.se valdistrikt-level results + demographics — Phase-2 enrichment.
  - [ ] Geo (valkrets GeoJSON) — deferred to Phase 6.
- [x] **Phase 2** — pollster ratings + house effects (`ratings.py` →
      `pollster_house_effects`, `industry_bias`, `pollster_ratings`). Built from
      final-30d poll error vs actual results, 2010–2022.
- [~] **Phase 3** — Bayesian model (NumPyro), `model.py`. Top-down national
      8-party ALR model with a **damped local-linear-trend** (level + velocity)
      latent state — momentum is estimated and projected, the deliberate fix for
      aggregator inertia. Dirichlet-Multinomial likelihood (over-dispersion),
      per-pollster house effects (centered), anchored at the 2022 result. Fits
      the cycle (~4 min on the laptop) → `model_trend.parquet` + election-day
      `forecast_samples.npz`. Election-day uncertainty is a **share-space**
      polling-miss term — calibrated (Phase 5), **horizon-dependent**, and
      **within-bloc correlated** (see "Uncertainty model").
      - [ ] **Model-carried error** — make the spread emerge from the latent
            (backward-from-election-day random walk with horizon-accumulating
            innovations + explicit election-day fundamentals prior, Economist-style)
            instead of the calibrated add-on. The deeper fix.
      - [ ] Feed Phase-2 house effects / industry bias as priors.
- [x] **Phase 4** — simulator (`simulate.py`): forecast draws → 4%/12% gate →
      allocator → seat distributions → bloc & government-formation probabilities,
      coalition table, threshold survival, kingmaker drama. Config-driven blocs
      (Right/Tidö, Left, C as unaligned kingmaker). Runs in <1s →
      `seat_forecast`, `coalition_forecast`, `government_forecast.json`,
      `seat_draws.npz`. Conditionals guarded against tiny-subsample noise.
- [x] **Phase 5** — backtest & calibration (`backtest.py`). Refits 2018 & 2022
      from polls-only (velocity on/off, horizon-matched to the live ~14-week
      gap). **Calibrated `MISS_SIGMA` = 2.25pp** (share-space, ~88% coverage on
      backtests). **Momentum thesis confirmed**: velocity cuts rising-party
      under-prediction 0.60→0.52pp. Point MAE 0.9–1.5pp. Found & fixed: the miss
      belongs in share space, not ALR (poll errors are ~uniform in pp across
      party sizes). Within-bloc miss correlation modelled as a factor
      (`MISS_RHO`, deliberately modest at 0.2 — see model.py).
      - [ ] **Longer backtest: add 2010 & 2014 (4 cycles)** — to calibrate
            `MISS_SIGMA` more tightly and estimate `MISS_RHO` (within-bloc miss
            correlation) from data instead of assuming it; plus horizon-dependent
            `MISS_SIGMA`.
- [~] **Phase 6** — Astro webapp (Swedish, prime-era 538 look) + Cloudflare
      Pages + local recompute/publish.
  - [x] `parties.py` (verified palette), `web_export.py` (→ `web/src/data/*.json`:
        forecast, trend, seat_draws), Astro scaffold, Base layout (GA4 prod-only,
        fonts, header/footer), **Hero** ("Vem styr Sverige?" 3-way) + **Hemicycle**
        (349-seat arc, logo motif). Builds clean.
  - [x] All sections: "På vippen" threshold gauges, momentum trend chart,
        interactive coalition builder (uses `seat_draws.json`), Metod page.
  - [x] **Deployed live: https://trefyranio.web.app** (Firebase Hosting, project
        `trefyranio`, free Spark tier). `deploy.sh` (export→build→deploy via
        Hosting REST API + gcloud token, mirrors fifa-2026), `daily_update.sh`
        (recompute→publish), `ops/launchd-daily.plist.example`.
  - [x] **Live on https://trefyran.io** (apex + cert). Central-parliament
        hemicycle (4% gate applied to mean shares → near-threshold parties show 0,
        not a misleading average). Interactive trend chart: range toggle, raw-poll
        overlay, recent-polls table + poll modal (raw vs house-adjusted, pollster
        quality/bias). www = CNAME → trefyranio.web.app.
  - [ ] Dynamic OG image per deploy, seat-outcome histogram, further polish.
- [~] **Phase 7** — SEO/awareness. Done: robots.txt, sitemap (`@astrojs/sitemap`),
      OG share image, IndexNow (key + per-deploy ping), GA4 (prod-only).
      TODO: Google Search Console verification + submit sitemap; traffic strategy.
- [ ] **Phase 7** — domain/SEO/analytics + traffic strategy.

## Headline outputs (the product)

1. **Who governs** — P(statsminister / each viable regering).
2. **Mandate per party** — vote-share posterior → seats, with uncertainty.
3. **Momentum** — how the wind is turning, over the full term and recent months.
4. **Coalitions** — which reach ≥175, who's pivotal, P(each government).
5. **Threshold survival** — "is this the year a party gets eliminated?!"
   P(below 4%) per near-threshold party (L, KD, MP, C historically).

## Design

Prime-era FiveThirtyEight (~2016–2020): clean white canvas, sparse gridlines,
data-forward, bold headline numerals. **Swedish blue/yellow in the logo only**;
charts use **party colours**. Key chart types (refs in `img-references/`): poll
trend line (dashed raw + solid model), seat-outcome histogram with dual
"X in 100" headlines, beeswarm, snake distribution bar.

## Pipeline

Local, in order (Python 3.12 venv):

```sh
python -m trefyranio.etl.build_polls      # poll spine
python -m trefyranio.etl.build_results    # results spine (SCB)
python -m trefyranio.ratings              # pollster ratings / house effects
python -m trefyranio.model                # Bayesian model fit (~4 min)
python -m trefyranio.simulate             # seats → government → coalitions
python -m trefyranio.web_export           # → web/src/data/*.json
./deploy.sh                               # export + astro build + Firebase deploy
# full refresh + publish in one go:  ./daily_update.sh
# calibration (occasional):          python -m trefyranio.backtest all   # ~18 min
```

**Live:** https://trefyranio.web.app (Firebase Hosting, project `trefyranio`).
Custom domain `trefyran.io` pending DNS.

## Develop

```sh
python -m pytest tests/ -v
```

## License & disclaimer

MIT (see `LICENSE`) — © 2026 Krukis Data Science. Open source on purpose:
transparency is the point of a forecast.

This is a statistical forecast built on public data (Valmyndigheten, SCB,
SwedishPolls) — **not a prediction, and not affiliated with any party**. No
restricted data (e.g. VALU microdata) is redistributed here; the ETL fetches
open sources directly.
