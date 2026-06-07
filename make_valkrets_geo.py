"""Build simplified per-valkrets SVG paths for the seat map (occasional ETL).

There is no off-the-shelf GeoJSON for the 29 riksdag valkretsar, so we assemble
them by dissolving the 290 municipality polygons (okfse/sweden-geojson) via a
municipality -> valkrets mapping, then project to SVG paths for a dependency-free
inline choropleth.

  .venv/bin/python make_valkrets_geo.py   # → web/src/data/valkrets_geo.json

The 18 single-county valkretsar + Stockholm map by county/kommun code; the Skåne
and Västra Götaland sub-valkretsar use the municipality groupings defined by
Valmyndigheten (sourced from the sv.wikipedia valkrets articles; verified to cover
all 33 Skåne and 49 VG municipalities exactly).
"""
import json
import math
import urllib.request
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "web" / "src" / "data" / "valkrets_geo.json"
SRC = "https://raw.githubusercontent.com/okfse/sweden-geojson/master/swedish_municipalities.geojson"

# County (lan_code) -> valkrets for the 18 single-county valkretsar.
LAN_TO_VK = {
    "03": ("VR3", "Uppsala läns"), "04": ("VR4", "Södermanlands läns"),
    "05": ("VR5", "Östergötlands läns"), "06": ("VR6", "Jönköpings läns"),
    "07": ("VR7", "Kronobergs läns"), "08": ("VR8", "Kalmar läns"),
    "09": ("VR9", "Gotlands läns"), "10": ("VR10", "Blekinge läns"),
    "13": ("VR15", "Hallands läns"), "17": ("VR21", "Värmlands läns"),
    "18": ("VR22", "Örebro läns"), "19": ("VR23", "Västmanlands läns"),
    "20": ("VR24", "Dalarnas läns"), "21": ("VR25", "Gävleborgs läns"),
    "22": ("VR26", "Västernorrlands läns"), "23": ("VR27", "Jämtlands läns"),
    "24": ("VR28", "Västerbottens läns"), "25": ("VR29", "Norrbottens läns"),
}
VK_NAME = {
    "VR1": "Stockholms kommun", "VR2": "Stockholms län", "VR11": "Malmö kommun",
    "VR12": "Skåne läns västra", "VR13": "Skåne läns södra", "VR14": "Skåne läns norra och östra",
    "VR16": "Göteborgs kommun", "VR17": "Västra Götalands läns västra",
    "VR18": "Västra Götalands läns norra", "VR19": "Västra Götalands läns södra",
    "VR20": "Västra Götalands läns östra",
}
# Skåne (lan 12) sub-valkrets municipality groupings (Malmö = VR11 by kommun code).
SKANE = {
    "VR12": {"Bjuv", "Eslöv", "Helsingborg", "Höganäs", "Hörby", "Höör", "Landskrona", "Svalöv"},
    "VR13": {"Burlöv", "Kävlinge", "Lomma", "Lund", "Sjöbo", "Skurup", "Staffanstorp",
             "Svedala", "Trelleborg", "Vellinge", "Ystad"},
    "VR14": {"Bromölla", "Båstad", "Hässleholm", "Klippan", "Kristianstad", "Osby", "Perstorp",
             "Simrishamn", "Tomelilla", "Åstorp", "Ängelholm", "Örkelljunga", "Östra Göinge"},
}
# Västra Götaland (lan 14) sub-valkrets groupings (Göteborg = VR16 by kommun code).
VG = {
    "VR17": {"Ale", "Alingsås", "Härryda", "Kungälv", "Lerum", "Lilla Edet", "Mölndal",
             "Partille", "Stenungsund", "Tjörn", "Öckerö"},
    "VR18": {"Bengtsfors", "Dals-Ed", "Färgelanda", "Lysekil", "Mellerud", "Munkedal", "Orust",
             "Sotenäs", "Strömstad", "Tanum", "Trollhättan", "Uddevalla", "Vänersborg", "Åmål"},
    "VR19": {"Bollebygd", "Borås", "Herrljunga", "Mark", "Svenljunga", "Tranemo", "Ulricehamn", "Vårgårda"},
    "VR20": {"Essunga", "Falköping", "Grästorp", "Gullspång", "Götene", "Hjo", "Karlsborg",
             "Lidköping", "Mariestad", "Skara", "Skövde", "Tibro", "Tidaholm", "Töreboda", "Vara"},
}


def _norm(name: str) -> str:
    n = name.strip().lower()
    return n[:-1] if n.endswith("s") else n   # drop genitive 's' (Bjuvs -> bjuv)


def _vk_for(lan: str, code: str, name: str) -> str:
    if lan == "01":
        return "VR1" if code == "0180" else "VR2"
    if lan == "12":
        if code == "1280":
            return "VR11"
        for vk, members in SKANE.items():
            if _norm(name) in {_norm(m) for m in members}:
                return vk
        raise KeyError(f"unmapped Skåne municipality {name} ({code})")
    if lan == "14":
        if code == "1480":
            return "VR16"
        for vk, members in VG.items():
            if _norm(name) in {_norm(m) for m in members}:
                return vk
        raise KeyError(f"unmapped VG municipality {name} ({code})")
    return LAN_TO_VK[lan][0]


def main():
    raw = json.loads(urllib.request.urlopen(SRC).read())
    groups: dict[str, list] = {}
    for f in raw["features"]:
        p = f["properties"]
        vk = _vk_for(p["lan_code"], p["id"], p["kom_namn"])
        groups.setdefault(vk, []).append(shape(f["geometry"]))
    assert len(groups) == 29, f"expected 29 valkretsar, got {len(groups)}"

    # Dissolve + simplify each valkrets.
    geoms = {vk: unary_union(g).simplify(0.012, preserve_topology=True) for vk, g in groups.items()}

    # Project lon/lat -> SVG units (equirectangular with latitude correction).
    lats = [pt[1] for geo in geoms.values() for poly in _polys(geo) for pt in poly.exterior.coords]
    lons = [pt[0] for geo in geoms.values() for poly in _polys(geo) for pt in poly.exterior.coords]
    lat0 = math.radians((min(lats) + max(lats)) / 2)
    k = math.cos(lat0)
    xs = [(lo - min(lons)) * k for lo in lons]
    W = 1000.0
    sx = W / max(xs)
    ymax = max(lats)
    H = (max(lats) - min(lats)) * sx

    def proj(lon, lat):
        return ((lon - min(lons)) * k * sx, (ymax - lat) * sx)

    out = []
    for vk, geo in geoms.items():
        d = []
        cx = cy = n = 0
        for poly in _polys(geo):
            pts = [proj(x, y) for x, y in poly.exterior.coords]
            d.append("M" + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z")
            for x, y in pts:
                cx += x; cy += y; n += 1
        out.append({"code": vk, "name": _county_name(vk),
                    "d": " ".join(d), "cx": round(cx / n, 1), "cy": round(cy / n, 1)})

    payload = {"viewBox": f"0 0 {W:.0f} {H:.0f}", "valkrets": sorted(out, key=lambda r: r["code"])}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB, {len(out)} valkretsar, "
          f"viewBox {payload['viewBox']})")


def _polys(geo):
    return list(geo.geoms) if geo.geom_type == "MultiPolygon" else [geo]


def _county_name(vk):
    for lan, (code, name) in LAN_TO_VK.items():
        if code == vk:
            return name
    return VK_NAME.get(vk, vk)


if __name__ == "__main__":
    main()
