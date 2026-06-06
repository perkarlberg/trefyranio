#!/usr/bin/env bash
# Full local refresh → publish. Pulls the latest polls, refits the model,
# re-simulates seats/government, then builds + deploys the site.
#
# Compute runs locally (no cloud compute). The model fit takes a few minutes.
# Historical results, pollster ratings and MISS_SIGMA calibration change rarely,
# so they are NOT rerun here — refresh them manually when needed
# (build_results / ratings / backtest).
#
#   ./daily_update.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"

echo "==> [$(date -u +%FT%TZ)] Refreshing polls (SwedishPolls)"
"$PY" -m trefyranio.etl.build_polls

echo "==> Refitting model (~4 min)"
"$PY" -m trefyranio.model

echo "==> Simulating seats & government"
"$PY" -m trefyranio.simulate

echo "==> Build + deploy"
"$ROOT/deploy.sh"

echo "==> [$(date -u +%FT%TZ)] Daily update complete."
