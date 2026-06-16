# trefyranio — operating guide (for Claude)

Probabilistic forecast for the 2026 Swedish riksdagsval, live at **https://trefyran.io**
(Firebase Hosting, project `trefyranio`). **Methodology, model, and references: see
`README.md`** — this file is the operational runbook, not the explainer.

## Environment
- Python venv at `.venv` (3.12). Run modules as `.venv/bin/python -m trefyranio.<x>`.
- Tests: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` (should be all green).
- `data/` (parquet, npz, backtests/) is **gitignored** — built locally, never committed.
- Web build needs `npm` (in `web/`). The seat-map geometry ETL needs the optional
  `geo` extra (`shapely`); it's a rare rebuild, not part of a normal refresh.

## Recurring task — refresh the forecast when new polls land
1. **Check first** (don't refit blindly — a refit with no new poll just reproduces the
   same numbers and wastes ~10 min):
   ```sh
   .venv/bin/python -m trefyranio.etl.build_polls   # fetch SwedishPolls + 2026 Wikipedia table → polls.parquet
   ```
   It prints the latest poll (`latest poll: <pollster> field-end <date>`). Compare to
   the live forecast's `as_of` (in `web/src/data/forecast.json`) or the prior run. Count
   check:
   ```sh
   PYTHONPATH=src .venv/bin/python -c "import pandas as pd; from trefyranio.model import PROCESSED_DIR, CYCLE_2026; p=pd.read_parquet(PROCESSED_DIR/'polls.parquet'); p['d']=p.field_end.fillna(p.pub_date); c=p[p.d.dt.date>=CYCLE_2026.start]; print('cycle polls', c.poll_id.nunique(), '| latest', c.loc[c.d.idxmax(),'pollster'], c.d.max().date())"
   ```
   **No new poll → stop. Do not redeploy.**
2. **New poll(s) → refit, simulate, deploy:**
   ```sh
   .venv/bin/python -m trefyranio.model      # ~10 min (4 chains); prints worst election-day r-hat — want <1.05
   .venv/bin/python -m trefyranio.simulate
   ./deploy.sh                               # web_export → make_og → astro build → Firebase deploy → IndexNow
   ```
   `./daily_update.sh` does the poll-fetch + all three in one shot.
3. **Commit the regenerated web data** so the repo matches the live site:
   `web/src/data/*.json` + `web/public/{seat_draws.json,og.png}` are **tracked** and
   change each forecast (the `data/` artifacts are gitignored and stay local). Commit
   them with a short message. (If you forget, the repo silently drifts from production.)

## Deploy prerequisites
- `deploy.sh` mints a token via `gcloud auth print-access-token --account=perkarlberg@gmail.com`
  (override with `TREFYRANIO_GCLOUD_ACCT`). If it errors, gcloud isn't authed for that
  account — Claude can't do interactive login; tell the user to run `! gcloud auth login`.
- Firebase project + site are both `trefyranio`. No service-account keys, no firebase-CLI login.

## Conventions & gotchas
- Commit messages end with the `Co-Authored-By: Claude …` trailer (match `git log`).
- **Occasional, not per-update:** `python -m trefyranio.backtest all` recalibrates the
  uncertainty (`MISS_SIGMA`/`MISS_RHO`, ~25 min, cached fits); `make_valkrets_geo.py`
  rebuilds the map geometry; the results/ratings spines change only at a new election.
- Tested-but-rejected terms ship **inert** and are re-testable, not deleted: recent
  momentum (`use_velocity`/drift, φ=0) and the cost-of-ruling fundamentals prior
  (`FUND_WEIGHT_PER_WEEK=0`). Re-enable only if a backtest justifies it.
- The map/forecast totals are intentionally the **central scenario** (`allocate_national`
  on mean shares), which can differ from the mean-of-draws seat averages — don't "fix"
  this; it's documented.
