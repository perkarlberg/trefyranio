#!/usr/bin/env bash
# Guarded daily refresh — the entrypoint the Ubuntu systemd timer runs at 07:00.
#
# Unlike daily_update.sh (which refits unconditionally), this fetches the latest
# polls FIRST and only refits/simulates/deploys when a poll newer than the live
# forecast has landed. A refit with no new poll just reproduces the same numbers
# and wastes ~10 min, so on a "nothing new" day we stop after the cheap fetch.
#
#   ops/daily_refresh.sh
# Logs go to stdout/stderr (journald when run under systemd; see ops/ubuntu-setup.md).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "==> [$(date -u +%FT%TZ)] $*"; }

cd "$ROOT"

log "Fetching latest polls (SwedishPolls + Wikipedia 2026)"
"$PY" -m trefyranio.etl.build_polls

# Branch on the guard's exit code: 0 = new poll, 10 = nothing new, 2 = undetermined.
set +e
"$PY" "$ROOT/ops/check_new_poll.py"
guard=$?
set -e

case "$guard" in
  0)  log "New poll detected — refitting" ;;
  10) log "No new poll since the live forecast — nothing to do."; exit 0 ;;
  *)  log "Guard could not determine poll state (exit $guard) — aborting to be safe."; exit 1 ;;
esac

log "Refitting model (~10 min, 4 chains)"
"$PY" -m trefyranio.model

log "Simulating seats & government"
"$PY" -m trefyranio.simulate

log "Building + deploying to Firebase Hosting"
"$ROOT/deploy.sh"

log "Daily refresh complete — published."
