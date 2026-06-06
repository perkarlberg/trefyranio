#!/usr/bin/env python3
"""Generate the social share image (web/public/og.png, 1200×630) from the live
forecast, so link previews show the current numbers — not a stale snapshot.

Run after web_export, before the Astro build (deploy.sh does this). Needs a local
headless Chrome; if absent it keeps the existing og.png and exits 0 (never breaks
a deploy).
"""
import base64
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main() -> None:
    fc = json.loads((ROOT / "web/src/data/forecast.json").read_text())
    g = fc["government"]
    pct = lambda x: round(x * 100)
    cols = [
        ("Rödgrönt", pct(g["left"]), "#c8102e", g["left"]),
        ("Centern avgör", pct(g["kingmaker"]), "#009933", g["kingmaker"]),
        ("Tidö", pct(g["right"]), "#1b4f8a", g["right"]),
    ]
    logo = base64.b64encode((ROOT / "web/public/logo.png").read_bytes()).decode()

    cards = "".join(
        f'<div class="c"><div class="n" style="color:{c}">{p}%</div><div class="l">{lbl}</div></div>'
        for lbl, p, c, _ in cols)
    bar = "".join(f'<i style="width:{f*100:.1f}%;background:{c}"></i>' for _, _, c, f in cols)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@600;800;900&display=swap" rel="stylesheet">
<style>
 *{{margin:0;box-sizing:border-box}}
 body{{width:1200px;height:630px;font-family:"Libre Franklin",sans-serif;color:#2b2b2b;background:#fff;
   padding:64px 72px;display:flex;flex-direction:column;justify-content:space-between}}
 .top{{display:flex;align-items:center;gap:20px}} .top img{{width:84px;height:84px}}
 .wm{{font-weight:900;font-size:52px;letter-spacing:-.02em}} .wm .io{{color:#6b6b6b;font-weight:800}}
 h1{{font-size:84px;font-weight:900;letter-spacing:-.03em;line-height:1.05;margin-top:18px}}
 .sub{{font-size:30px;color:#6b6b6b;font-weight:600;margin-top:14px}}
 .cols{{display:flex;gap:48px}} .n{{font-size:74px;font-weight:900;line-height:1}}
 .l{{font-size:24px;font-weight:700;color:#6b6b6b;margin-top:4px}}
 .bar{{display:flex;height:18px;border-radius:9px;overflow:hidden;margin-top:8px}} .bar i{{display:block;height:100%}}
</style></head><body>
 <div>
  <div class="top"><img src="data:image/png;base64,{logo}"><span class="wm">trefyran<span class="io">io</span></span></div>
  <h1>Vem styr Sverige efter valet?</h1>
  <div class="sub">Valprognos för riksdagsvalet 2026 — uppdaterad {fc['updated']}</div>
 </div>
 <div>
  <div class="cols">{cards}</div>
  <div class="bar">{bar}</div>
 </div>
</body></html>"""

    if not Path(CHROME).exists():
        print("make_og: Chrome not found — keeping existing og.png")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp = f.name
    out = ROOT / "web/public/og.png"
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--window-size=1200,630",
         "--default-background-color=FFFFFFFF", f"--screenshot={out}", f"file://{tmp}"],
        check=True, capture_output=True,
    )
    Path(tmp).unlink()
    print(f"make_og: wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB) — {cols[0][1]}/{cols[1][1]}/{cols[2][1]}")


if __name__ == "__main__":
    main()
