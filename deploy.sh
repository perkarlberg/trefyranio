#!/usr/bin/env bash
# Build the site from the latest forecast artifacts and deploy to Firebase
# Hosting (project + site "trefyranio", https://trefyranio.web.app).
#
# Deploys via the Hosting REST API with a gcloud access token — no firebase
# CLI login, no service-account keys (same approach as fifa-2026/probaball).
#
#   ./deploy.sh        export web data → build Astro → deploy
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="trefyranio"
SITE="trefyranio"
GCLOUD_ACCT="${TREFYRANIO_GCLOUD_ACCT:-perkarlberg@gmail.com}"
PY="$ROOT/.venv/bin/python"

echo "==> Exporting web data from forecast artifacts"
"$PY" -m trefyranio.web_export

echo "==> Building Astro site"
cd "$ROOT/web" && npm run build

echo "==> Deploying to Firebase Hosting ($SITE)"
TOKEN=$(gcloud auth print-access-token --account="$GCLOUD_ACCT")
"$PY" "$ROOT/deploy_hosting.py" "$SITE" "$ROOT/web/dist" "$TOKEN" "$PROJECT"

# Tell IndexNow (Bing/Yandex) the pages changed so they re-crawl promptly.
KEY="9d4c1f7a8b2e4a6c9f0d3b5e7a1c2d48"
curl -s -X POST "https://api.indexnow.org/indexnow" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "{\"host\":\"trefyran.io\",\"key\":\"$KEY\",\"keyLocation\":\"https://trefyran.io/$KEY.txt\",\"urlList\":[\"https://trefyran.io/\",\"https://trefyran.io/metod\"]}" \
  -o /dev/null -w "==> IndexNow ping: HTTP %{http_code}\n" || true

echo "==> Live: https://$SITE.web.app"
