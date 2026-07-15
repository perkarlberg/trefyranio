# Deploying the daily refresh on the home Home Assistant box — plan & findings

**Status:** planned, not yet built. Picked up when next on the home LAN
(`10.0.0.0/24`). Captured 2026-07-15 from an on-net investigation session.

Goal: run the trefyranio forecast refresh **nightly (~03:00)** on the always-on
home server — fetch new polls, and *only if a poll newer than the live forecast
landed* refit → simulate → deploy to Firebase. Deploy-only; no git push from the
box (commit the refreshed forecast from the Mac, as per the root `CLAUDE.md`).

> The target turned out **not** to be a plain "Ubuntu 24" machine — it's
> **Home Assistant OS** running on an old NUC. That changes the deployment
> mechanism entirely (a local HA add-on, not systemd). The generic-Ubuntu
> `systemd` bundle in this dir (`daily_refresh.sh`, `check_new_poll.py`,
> `trefyranio-daily.{service,timer}`, `ubuntu-setup.md`) is still valid **for a
> normal Linux host / cloud VM** — keep it for that option — but it is *not* the
> path for the HAOS box. `daily_refresh.sh` + `check_new_poll.py` are reused
> unchanged by the add-on below.

## The target box (measured on-net)

| | |
|---|---|
| OS | **Home Assistant OS 18.1**, HA Core 2026.7.1, Supervisor 2026.06.2, Docker 29.5.3, `machine=generic-x86-64`, hostname `homeassistant` |
| Hardware | Intel **Core i3-3217U @ 1.80 GHz** (Ivy Bridge, 2 physical / 4 logical cores) |
| RAM | 3381 MB total, **~2270 MB available**, 141 MB "free" (rest reclaimable cache) |
| Swap | **1117 MB already configured** at `/mnt/data/swapfile` (priority -2) |
| Disk | `/share` (`/dev/sda8`) 109 GB, **98 GB free** |
| **CPU flags** | `avx sse4_1 sse4_2` — **AVX yes, AVX2/FMA no** |

### Access (from the Mac, on the home LAN only)
- `ssh root@10.0.0.210` — **public-key only; the Mac's default `~/.ssh/id_ed25519`
  is authorized.** Lands in the `core-ssh` add-on container (**Alpine Linux v3.23**).
- `10.0.0.210` is the **WiFi** link (`wlp2s0`, SSID `LillaEssingen4Ever`, hidden).
  The primary **ethernet** `10.0.0.209` (USB adapter `enp0s29u1u1`) was **timing
  out** during the session — use `.210`, or check the USB ethernet next time.
- Debug SSH port 22222 is **closed**. The `ha` CLI is at `/usr/bin/ha`.
- No Docker binary / socket exposed inside `core-ssh`. `/addons`, `/addon_configs`,
  `/share` **are** mounted — the supported route is a **local add-on** under `/addons/`.
- See the `homeassistant` repo's memory `ha-server-access.md` for the fuller picture
  (backups, network admin, etc.).

## Measurements (fit on the Mac, Apple Silicon)

Real `python -m trefyranio.model` run, `/usr/bin/time -l`:
- **Peak RAM ~611 MB** (max RSS 640,909,312 B; footprint 680,691,248 B) — this is
  the current **vectorized, 4-chain** default.
- **Wall time 4.9 min** (291.95 s real / 677 s user → ~2.3× core parallelism).
  The "~10 min" in `daily_update.sh`'s comment is stale.
- Fit shape: 243 polls, 8 pollsters, 209 weeks (election week 208), worst
  election-day r-hat 1.000.

Model config (`src/trefyranio/model.py:418-426`): `fit(warmup=600, samples=500,
num_chains=4, chain_method="vectorized")`, NUTS `target_accept=0.9`,
`init_to_median`. `build()` calls it with `samples=600`. **No `jax_enable_x64`
anywhere → float32** (memory already as lean as it gets).

## OOM verdict: essentially a non-issue

611 MB peak against **2.27 GB available + 1.1 GB swap** ≈ 3× headroom before the
OOM-killer is in play. Conclusions:
- **Keep `chain_method="vectorized"`** — no need to switch to `"sequential"` for
  memory. (Sequential would cut peak to ~250 MB but ~4× the wall time; only worth
  it if we want to be gentler on CPU, and thread-capping does that more cheaply.)
- Levers actually worth applying, both light-touch and mainly to keep Home
  Assistant responsive during the run (not for RAM):
  - Cap threads: `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
    `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`.
  - Run at night when HA is idle.
- Float32 already in effect — do **not** enable x64.
- The 1.1 GB swap is the seatbelt; `/share` has 98 GB to grow it if ever needed.

## The real risks (verify before building the full job)

1. **jax on an AVX-only (no AVX2/FMA) 2012 CPU.** `jaxlib` should *import* (AVX is
   present, so no `Illegal instruction` on load), but XLA will JIT non-AVX2 code →
   expect the 4.9-min Mac fit to become **~30–60 min** here. Fine at night — *if it
   runs at all*. **Must be smoke-tested on the box.**
2. **glibc vs musl.** The `core-ssh` container is **Alpine/musl**; `jaxlib` wheels
   are **glibc-only**. So the job **must** run in a **Debian-glibc** container
   (`python:3.12-slim`), i.e. a local add-on with its own base image — never in the
   SSH container directly.
3. **gcloud auth inside a container.** `deploy.sh` needs a token from
   `gcloud auth print-access-token --account=perkarlberg@gmail.com`. In a container,
   point `CLOUDSDK_CONFIG` at the add-on's persistent `/data` (or a `/addon_configs`
   path) so creds survive restarts, and do the one-time
   `gcloud auth login --no-launch-browser` into that volume — **or** mount a service
   account key (Firebase Hosting Admin) and activate it. Decide at build time.
4. **Astro/node build** also runs in the same container (`deploy.sh` → `npm run
   build`) — needs node 20 + `web/` deps; modest RAM, but include it and time it.

## Proposed build (HAOS local add-on)

1. **Smoke test first** — minimal `/addons/trefyranio/` add-on, `python:3.12-slim`
   base, `pip install jax numpyro`, run a 30-second toy NUTS. Build via
   `ha addons reload && ha addons install local_trefyranio && ha addons start …`;
   read logs. Confirms jax executes on this CPU + gives a real speed read. Abort/rethink
   if it SIGILLs or is unusably slow.
2. **Full add-on** (same skeleton): Debian base with python 3.12 + the repo
   (`pip install -e '.[model]'`), node 20 + `web/` deps, gcloud CLI. Clones/mounts
   the repo (or bakes it in and `git pull`s). Runs the guarded pipeline via the
   existing **`ops/daily_refresh.sh`** (which already fetches → `ops/check_new_poll.py`
   guard → refit/simulate → `deploy.sh`).
3. **Schedule:** long-lived add-on with an internal `crond` entry at ~03:00
   (add-ons have no native scheduler). Alternative: an HA automation firing
   `hassio.addon_start` nightly — but internal cron is simpler and self-contained.
4. **Persist** `CLOUDSDK_CONFIG` + the built venv/repo on the add-on's `/data` so
   restarts don't re-download everything.

## Open decisions for next session
- gcloud: persisted user creds vs service-account key (see risk #3).
- Bake the repo into the image vs clone-and-`git pull` at runtime.
- Whether to also fold in the occasional recalibration (backtest/ratings/geo) — **no**,
  keep those manual (per root `CLAUDE.md`); this job is polls → refit → deploy only.
