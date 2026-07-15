# Running the daily forecast refresh on Ubuntu 24.04

This sets up an unattended job that, **every day at 07:00 Europe/Stockholm**, fetches
the latest opinionsmätningar and — *only if a poll newer than the live forecast has
landed* — refits the model, re-simulates, and publishes to Firebase Hosting. On a day
with no new poll it stops after the cheap fetch (~seconds) instead of wasting ~10 min
on a refit that reproduces the same numbers.

Pieces (all in `ops/`):
- `daily_refresh.sh` — the guarded entrypoint (fetch → guard → refit/simulate/deploy).
- `check_new_poll.py` — the guard: exit 0 = new poll, 10 = nothing new.
- `trefyranio-daily.service` / `.timer` — systemd **user** units that run it on schedule.

This is the Linux counterpart to the macOS `ops/launchd-daily.plist.example`.

## 1. System prerequisites

```sh
sudo apt update
sudo apt install -y python3.12 python3.12-venv git curl
# Node 20+ for the Astro build (NodeSource; Ubuntu's own nodejs is older):
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

## 2. Clone and build the Python env

The units assume the repo lives at `~/trefyranio` (they use `%h/trefyranio`). If you
put it elsewhere, edit the `%h/trefyranio` paths in both unit files.

```sh
git clone <your-remote> ~/trefyranio
cd ~/trefyranio
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[model]'          # pulls numpyro/jax/numpy for the fit
cd web && npm ci && cd ..                      # Astro deps for the build
```

Smoke-test the pipeline once by hand (this WILL refit + deploy if a new poll exists):

```sh
.venv/bin/python -m trefyranio.etl.build_polls
PYTHONPATH=src .venv/bin/python ops/check_new_poll.py; echo "guard exit=$?"
```

## 3. gcloud auth (needed by deploy.sh)

`deploy.sh` mints a Firebase Hosting token via
`gcloud auth print-access-token --account=perkarlberg@gmail.com`.

```sh
# Install the CLI (snap is simplest on Ubuntu):
sudo snap install google-cloud-cli --classic

# Authenticate ONCE as the deploy account. On a headless box use --no-launch-browser
# and paste the code back. The refresh token persists in ~/.config/gcloud, and
# `print-access-token` refreshes silently on every run thereafter.
gcloud auth login --no-launch-browser perkarlberg@gmail.com
```

> If you'd rather not keep user credentials on the server, create a service account
> with the *Firebase Hosting Admin* role, download its key, and either point
> `GOOGLE_APPLICATION_CREDENTIALS` at it or override `TREFYRANIO_GCLOUD_ACCT` with the
> activated SA — `gcloud auth print-access-token` works the same way. User creds are
> fine here; they just need that one interactive login.

Verify: `gcloud auth print-access-token --account=perkarlberg@gmail.com >/dev/null && echo ok`

## 4. Install the timer

```sh
mkdir -p ~/.config/systemd/user
cp ~/trefyranio/ops/trefyranio-daily.service ~/.config/systemd/user/
cp ~/trefyranio/ops/trefyranio-daily.timer   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now trefyranio-daily.timer

# Let user services run without an active login session (survives logout/reboot):
sudo loginctl enable-linger "$USER"
```

## 5. Verify & operate

```sh
systemctl --user list-timers trefyranio-daily.timer   # next scheduled run
systemctl --user status trefyranio-daily.service      # last run result

# Trigger a run right now (respects the guard — deploys only if a new poll exists):
systemctl --user start trefyranio-daily.service

# Logs (stdout/stderr land in journald):
journalctl --user -u trefyranio-daily.service -n 100 --no-pager
journalctl --user -u trefyranio-daily.service -f        # follow live
```

## Notes

- **No catch-up.** `Persistent=false` — if the machine is off at 07:00, the run is
  skipped and the next attempt is the following day's 07:00. (Change to `true` if you
  ever want a missed run to fire on the next boot.)
- **Timezone.** The `Europe/Stockholm` suffix on `OnCalendar` needs systemd ≥ 252;
  24.04 ships 255, so DST is handled automatically. On older systemd, drop the suffix
  and set the box's timezone with `timedatectl set-timezone Europe/Stockholm`.
- **Deploy only, no git.** This job publishes to Firebase but does not commit or push
  the regenerated `web/src/data/*.json`. Commit the refreshed forecast from your Mac as
  usual (see the root `CLAUDE.md`), or the repo will drift behind the live site.
- **Occasional recalibration** (backtest / ratings / map geometry) is *not* part of this
  job — run it by hand as documented in `CLAUDE.md`.
