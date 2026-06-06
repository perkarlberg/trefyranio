#!/usr/bin/env python3
"""Deploy a static dist/ folder to Firebase Hosting via the REST API.

Same approach as the fifa-2026/probaball deploy: a Google OAuth access token
(`gcloud auth print-access-token`) + quota-project header. Avoids firebase-tools
interactive login and service-account keys.

Usage: deploy_hosting.py <site> <dist_dir> <access_token> <quota_project>
"""
import concurrent.futures
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://firebasehosting.googleapis.com/v1beta1"


def call(method, url, token, project, data=None, ctype="application/json",
         raw=False, timeout=60, retries=4):
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Goog-User-Project": project,
        "Content-Type": ctype,
    }
    body = data if raw else (data.encode() if data is not None else None)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last


def main():
    site, dist, token, project = sys.argv[1:5]

    # Static multipage site: clean URLs (/metod → /metod/index.html), strip
    # trailing slashes. No SPA fallback — unknown paths 404.
    version_config = json.dumps({
        "config": {
            "cleanUrls": True,
            "trailingSlashBehavior": "REMOVE",
            # Firebase applies headers last-match-wins, so the broad short-cache
            # rule goes first and the immutable hashed-asset rule goes last.
            "headers": [
                {"glob": "**",
                 "headers": {"Cache-Control": "public, max-age=600, must-revalidate"}},
                {"glob": "/_astro/**",
                 "headers": {"Cache-Control": "public, max-age=31536000, immutable"}},
            ],
        }
    })
    version = json.loads(call("POST", f"{API}/sites/{site}/versions",
                              token, project, version_config))["name"]
    print("version:", version)

    manifest, blobs = {}, {}
    for root, _, files in os.walk(dist):
        for fn in files:
            full = os.path.join(root, fn)
            rel = "/" + os.path.relpath(full, dist).replace(os.sep, "/")
            with open(full, "rb") as f:
                gz = gzip.compress(f.read(), 9)
            h = hashlib.sha256(gz).hexdigest()
            manifest[rel] = h
            blobs[h] = gz
    print("files:", len(manifest))

    pop = json.loads(call("POST", f"{API}/{version}:populateFiles",
                          token, project, json.dumps({"files": manifest})))
    upload_url = pop["uploadUrl"]
    required = pop.get("uploadRequiredHashes", [])
    print("upload required:", len(required))

    def _put(h):
        call("PUT", f"{upload_url}/{h}", token, project,
             data=blobs[h], ctype="application/octet-stream", raw=True)
        return h

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(_put, h) for h in required]
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            fut.result()
            if done % 50 == 0 or done == len(required):
                print(f"uploaded {done}/{len(required)}")

    call("PATCH", f"{API}/{version}?update_mask=status",
         token, project, json.dumps({"status": "FINALIZED"}))
    print("finalized")

    rel = json.loads(call("POST",
                          f"{API}/sites/{site}/releases?version_name={version}",
                          token, project, "{}"))
    print("released:", rel.get("name"))


if __name__ == "__main__":
    main()
