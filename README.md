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

- **Top-down** national 8-party vote-share **local-level random walk** in
  additive-log-ratio space, with a per-party **drift** (momentum) and
  **fixed walk scales** (for convergence — see below). Anchored at the previous
  election result.
- **Dirichlet-Multinomial** poll likelihood (absorbs nonsampling error via fixed
  overdispersion).
- Per-pollster **house effects** (centered across parties), **warm-started from
  each pollster's historical lean** (Phase-2 ratings, de-biased of the field-wide
  component); the current cycle's data overrides as it accumulates.
- **Accuracy-weighted likelihood**: pollsters that historically hit closest get
  higher Dirichlet concentration (tighter constraint on the latent). New entrants
  start neutral.
- **Field-bias correction**: a heavily-shrunk (30%) adjustment for the industry's
  persistent per-party miss, applied to the election-day forecast (see below).
  *Identification note:* centering house effects to zero-sum identifies the latent
  as the **pollster consensus**, which de-biases relative to the *average pollster*,
  not relative to *truth* — a field-wide lean is invisible to a poll-vs-poll
  comparison. We break that degeneracy with an external truth anchor: `industry_bias`
  is measured poll-vs-**actual result** (2010–2022) and applied as the field-bias
  correction. The house priors are de-biased of this same field component first, so
  the two corrections compose without double-counting. (The displayed *trend* line is
  the raw consensus; the *forecast* carries the truth correction — the 538-style split
  of "poll average" vs "forecast".)
- Fit with **multiple chains** + an r-hat convergence check (see "Convergence").

(The constituency-level piece of the full Economist model is not yet implemented.
A **cost-of-ruling fundamentals prior** *is* implemented but ships at **weight 0** —
the backtest rejects it (see "Fundamentals prior" below).)

**Swedish-specific tweaks:** the 4% national / 12% constituency threshold;
government-formation as the headline output; a **field-bias correction** from the
2010–2022 final-poll record — which shows the largest systematic industry misses
are an **understated S (~2.2pp)** and **overstated V/MP (~1pp)**, with SD's
final-poll bias actually small (~0.2pp) once pollsters adjusted (correction applied
at 30% strength — 4 elections is noisy, pollsters partly adapt); and the Novus 2023
phone→web methodology break.

**On *stödröstning* (tactical vote-lending).** A reasonable worry is that lending
votes to keep a small ally above 4% systematically rescues near-threshold parties
beyond their polls — the exact quantity we headline. We checked: among the near-
threshold cases (final-poll share 3–5.5%) in 2010–2022 there is **no systematic
overperformance** (mean actual−poll ≈ −0.25pp; the canonical L-2022 case got 4.6%
vs polling 5.5% — it *underperformed* its final poll and survived only because it
already polled above 4%). The reading: by the final stretch, lending intentions are
**already priced into the polls**, so there's no robust residual mean effect to add
(adding one would fit noise, like the rejected momentum/fundamentals terms). What
matters — threshold *uncertainty* — is already carried: near-threshold parties get
the full election-day spread (the `FWD_FLOOR_PQ` hybrid), so their 4%-survival is an
honest probability, not a point call.

## Uncertainty model (model-carried)

The forecast's spread **emerges from a forward projection of the latent**, not a
post-hoc add-on. The in-sample latent walk tracks the polls tightly (its own
innovation variance is small — Swedish shares move slowly), so the election-day
uncertainty is carried by a separate, calibrated **forward projection** from the
last poll to election day (`project_to_election` in `model.py`): the latent is
projected forward and a multiplicative (logit-space) random-walk innovation,
accumulating over the H-week gap, supplies the realized poll-miss spread.

- **Model-carried** — the spread is the latent process projected forward, so it
  *emerges* from a projection rather than being sprinkled on the shares. softmax
  keeps the simplex (no clipping). The simulator resamples the last-poll latent +
  drift posterior and draws a fresh forward innovation per draw → a 10k predictive
  ensemble.
- **Calibrated, not guessed** — the per-party innovation is sized so the induced
  election-day **share-space** spread matches realized poll error (errors are
  ~uniform in pp across party sizes). Coverage-calibrated across **four cycles
  (2010, 2014, 2018, 2022)** to 85% of the 80% interval: `σ(H) = √(1.55² + 0.060·H) pp`
  → ~1.55pp at H=0, ~1.80pp at H=14 (the *total* election-day spread; `MISS_SIGMA_*`).
  A 6-cycle test (adding 2002/2006) was **rejected**: those pre-2010 cycles had
  bigger misses (thinner polling, SD's emergence), and their wider σ *over-covers*
  the 2010+ regime the 2026 forecast lives in (91% vs the 85% target) — a regime
  mismatch, so we calibrate σ on the current party system only.
- **Hybrid per-party scaling** — in logit space share spread is `p(1−p)·σ_logit`,
  so we set `σ_logit = σ_share / (p(1−p))` (floored, `FWD_FLOOR_PQ`) to realize a
  ~uniform-pp band while protecting the 4%-threshold survival probabilities.
- **Correlated within blocs** — a factor model on the forward innovation gives
  within-bloc co-movement so bloc-total / government-formation variance isn't
  understated. `MISS_RHO`=0.12 is **estimated from data** (bloc-total coverage:
  iid under-covers at 75–83%, ρ≈0.12 reaches 85%, consistent across the 4- and
  6-cycle backtests) rather than assumed.

### Fundamentals prior (implemented, gated to 0)

The remaining structural ingredient of the Economist model is a
**fundamentals prior** — an election-day expectation from structure, not polls. The
robust, leak-free Swedish fundamental is the **cost of ruling**: governing parties
lose vote share in 11 of 14 elections (mean −1.4pp). It's implemented end-to-end
(`GOVERNMENTS` table, `cost_of_ruling`, `fundamentals_prior`, a horizon-weighted
blend in `project_to_election`) and **backtest-gated** — but the gate **rejects** it:
blending toward fundamentals improves point-forecast MAE only ~1.5% at H=14 (noise
on 4 cycles) and **0%** at H=30. A genuine fundamentals signal would help *more*
further out; that it doesn't is the tell. The reason is structural: **Sweden polls
densely even 7 months out, so the cost of ruling is already priced into the polls**
by the time we forecast. So `FUND_WEIGHT_PER_WEEK = 0` — the machinery stays
(re-testable with `python -m trefyranio.backtest fundamentals`), shipped inert, like
the rejected momentum term. We don't ship a prior the backtest can't justify.

### Convergence (fixed)

An earlier version did **not** converge — 4-chain diagnostics showed **r-hat
15–34, ESS≈2**: the level+velocity walk with *sampled* scale parameters had a
Neal's-funnel geometry NUTS couldn't navigate, so the posterior mean was
seed-sensitive (SD ±1.5pp across seeds). Fixed by:

- **fixing the walk scales** (`SIGMA_LVL`, `SIGMA_HOUSE`, `KAPPA`) instead of
  sampling them — removes the funnel;
- **dropping the per-week velocity** (the main mixing culprit) for a single
  per-party **drift** (`DRIFT_SIGMA`, 8 well-identified params) — convergence-safe;
- **multiple chains** (`num_chains=4`, vectorized) with an r-hat check baked into
  `fit()`.

Result: **worst election-day r-hat ≈ 1.00, ESS ~4,000.** The forecast is now
stable across seeds (and converges just as cleanly on all four backtest cycles,
r-hat ≤ 1.004). Trade-off: momentum is minimal — the per-party drift captures
trend, but the global drift comes out ≈0 this cycle. We tested a separate
damped recent-momentum term (Gardner & McKenzie) against the 4-cycle backtest:
the effect was **noise-level** (ALR RMSE 0.306 → 0.302 across all damping
factors) and the unregularized argmax was *undamped* — an overfitting signature.
So the live model ships **no separate momentum term**; the per-party drift is the
only trend mechanism. A term the backtest can't justify doesn't go in.

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
  - [x] **National-proportional allocation is EXACT under post-2018 rules** —
        verified against `seats_actual`: it reproduces 2018 & 2022 to the seat
        (`test_national_proportional_exact_under_current_rules`). The leveling
        seats make the outcome fully nationally proportional among 4%+ parties, so
        no per-constituency allocation is needed for forecast accuracy. (Pre-2018
        elections deviate a few seats — that disproportionality is exactly what the
        2018 reform removed; not relevant to 2026.)
  - [~] Per-constituency allocator (310 fixed + 39 leveling) — **not for accuracy**
        (national-proportional is exact) but to *visualize* where each party's/bloc's
        seats come from (uniform-swing per-valkrets allocation → seat-origins map).
  - [ ] val.se valdistrikt-level results + demographics — Phase-2 enrichment.
  - [~] Geo (valkrets GeoJSON) — for the seat-origins map (dedicated page).
- [x] **Phase 2** — pollster ratings + house effects (`ratings.py` →
      `pollster_house_effects`, `industry_bias`, `pollster_ratings`). Built from
      final-30d poll error vs actual results, 2010–2022.
- [x] **Phase 3** — Bayesian model (NumPyro), `model.py`. Top-down national
      8-party ALR **local-level random walk + per-party drift, fixed scales**,
      Dirichlet-Multinomial likelihood, per-pollster house effects (centered),
      anchored at the 2022 result. **Converges** (multi-chain, r-hat ≈ 1.0 — see
      Convergence) after the funnel reparam; velocity dropped (didn't converge),
      and a separate damped recent-momentum term was tested on the 4-cycle
      backtest and rejected as noise-level — drift only. Fits the cycle (~10 min,
      4 chains) →
      `model_trend.parquet` + election-day `forecast_samples.npz`. Election-day
      uncertainty is **model-carried** — a forward projection from the last poll
      (`project_to_election`) with a calibrated, **horizon-dependent**,
      **within-bloc correlated** multiplicative innovation (see "Uncertainty model").
      - [x] **Model-carried error** — the spread emerges from a forward projection
            of the latent (last-poll latent + calibrated logit-space innovation),
            retiring the share-space bolt-on. Calibrated to 85% coverage on 4 cycles.
      - [x] **Phase-2 ratings wired into the fit** — house-effect priors (de-biased,
            warm-start), accuracy weights (per-poll concentration), and a shrunk
            field-bias correction at the election-day forecast.
- [x] **Phase 4** — simulator (`simulate.py`): forecast draws → 4%/12% gate →
      allocator → seat distributions → bloc & government-formation probabilities,
      coalition table, threshold survival, kingmaker drama. Config-driven blocs
      (Right/Tidö, Left, C as unaligned kingmaker). Runs in <1s →
      `seat_forecast`, `coalition_forecast`, `government_forecast.json`,
      `seat_draws.npz`. Conditionals guarded against tiny-subsample noise.
- [x] **Phase 5** — backtest & calibration (`backtest.py`). Refits the converged
      model across **four cycles (2010, 2014, 2018, 2022)** from polls-only,
      horizon-matched at H=0 and H=14, and calibrates the **model-carried forward
      projection** (coverage-cal to 85%, σ(H) ~1.55→1.80pp; the induced share-space
      spread is ~uniform-pp). Point MAE 0.6–1.5pp. Damped recent-momentum tested on
      all 4 cycles and rejected (noise-level, overfit argmax) → drift-only. ⚠️ The
      earlier "momentum thesis confirmed (velocity)" result was on the
      **non-converged** model and is superseded.
      - [x] **6-cycle A/B (2002, 2006)** — tested adding the two pre-2010 cycles.
            **Rejected for σ**: their bigger misses over-cover the 2010+ regime
            (91% vs 85%) — a regime mismatch, so σ stays 4-cycle. **`MISS_RHO`
            estimated from data** at 0.12 (consistent 4- vs 6-cycle), replacing the
            assumed 0.2. (FP=L and SD-as-own-category are already handled in the
            data spine, so no special pre-2010 munging was needed.)
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
# calibration (occasional):          python -m trefyranio.backtest all   # ~25 min (8 fits, cached)
```

**Live:** https://trefyranio.web.app (Firebase Hosting, project `trefyranio`).
Custom domain `trefyran.io` pending DNS.

## References

The model stands on established work, not a bespoke recipe:

**Dynamic Bayesian poll aggregation & house effects**
- Jackman (2005), [*Pooling the Polls Over an Election Campaign*](https://www.tandfonline.com/doi/abs/10.1080/10361140500302472), *Australian Journal of Political Science* 40(4) — latent vote intention + per-pollster house effects.
- Linzer (2013), [*Dynamic Bayesian Forecasting of Presidential Elections in the States*](https://doi.org/10.1080/01621459.2012.737735), *JASA* 108(501).
- Heidemanns, Gelman & Morris (2020), [*An Updated Dynamic Bayesian Forecasting Model for the U.S. Presidential Election*](https://hdsr.mitpress.mit.edu/pub/nw1dzd02), *Harvard Data Science Review* 2(4) — the Economist model.
- Stoetzer, Neunhoeffer, Gschwend, Munzert & Sternberg (2019), [*Forecasting Elections in Multiparty Systems*](https://doi.org/10.1017/pan.2018.49), *Political Analysis* 27(2) — the closest analog (multiparty, polls + fundamentals, coalition probabilities).

**Vote shares as compositional data (the ALR/simplex geometry)**
- Aitchison (1986), *The Statistical Analysis of Compositional Data* — the additive-log-ratio transform.
- Bergman & Holmquist (2014), [*Poll of Polls: A Compositional Loess Model*](https://onlinelibrary.wiley.com/doi/10.1111/sjos.12023), *Scandinavian Journal of Statistics* 41(2); Bergman (2015), [*Are there house effects in Swedish polls?*](http://lup.lub.lu.se/search/record/c66befbe-deaa-4b4d-852f-9bb56a6107c5) (Lund) — Swedish poll-of-polls + house effects, compositional.

**Time series & inference**
- Harvey (1989), *Forecasting, Structural Time Series Models and the Kalman Filter* — the local-level random walk.
- Hoffman & Gelman (2014), [*The No-U-Turn Sampler*](https://jmlr.org/papers/v15/hoffman14a.html), *JMLR* 15 — the NUTS sampler (via NumPyro).

**Inspiration**
- Nate Silver / FiveThirtyEight: pollster ratings, house-effect adjustment, probabilistic simulation ([how 538's forecast works](https://abcnews.go.com/538/538s-2024-presidential-election-forecast-works/story?id=113068753); Silver, *The Signal and the Noise*, 2012). A Sweden-specific divergence: 538/Economist lean on fundamentals early-cycle, but Swedish polling is dense enough that our backtest finds fundamentals add nothing even ~7 months out (see "Fundamentals prior").

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
