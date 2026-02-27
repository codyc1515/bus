#!/usr/bin/env python3
"""Generate a cycleway accessibility map for addresses.

This script mirrors the map-style workflow of ``main.py`` but replaces bus routes/
stops with cycleways. It computes each address's nearest-cycleway distance and
builds an interactive HTML map with:
- cycleway polylines,
- address markers coloured by distance,
- a 400-800m threshold slider for live recolouring,
- summary counts for <=400m, <=800m, and 400-800m.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Optional

import folium
import pandas as pd
from folium import Element
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dist_to_green_red_hex(dist_m: Optional[float], max_m: float = 400.0) -> str:
    if dist_m is None or not math.isfinite(dist_m):
        return "#ff0000"
    if dist_m > max_m:
        return "#ff0000"
    t = clamp(dist_m / max_m, 0.0, 1.0)
    return f"#{int(round(255 * t)):02x}ff00"


def parse_bbox(vals: Optional[List[float]]) -> Optional[BBox]:
    if not vals:
        return None
    if len(vals) == 3:
        min_lon, min_lat, max_lon = vals
        max_lat = min_lat + (max_lon - min_lon)
    elif len(vals) == 4:
        min_lon, min_lat, max_lon, max_lat = vals
    else:
        raise ValueError("--bbox must be 3 or 4 numbers: min_lon min_lat max_lon [max_lat]")
    if min_lon > max_lon:
        raise ValueError("bbox invalid: min_lon > max_lon")
    if min_lat > max_lat:
        raise ValueError("bbox invalid: min_lat > max_lat")
    return BBox(min_lon, min_lat, max_lon, max_lat)


def load_ward_geometry(ward_geojson: str, ward_name: str):
    with open(ward_geojson, "r", encoding="utf-8") as f:
        data = json.load(f)

    ward_name_lc = ward_name.strip().lower()
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        name = str(props.get("WardNameDescription", "")).strip()
        if name.lower() == ward_name_lc:
            geom = feat.get("geometry")
            if geom:
                return shape(geom)
    raise ValueError(f"Ward {ward_name!r} not found in {ward_geojson}")


def load_addresses(
    addresses_csv: str,
    town: Optional[str],
    ta: Optional[str],
    bbox: Optional[BBox],
    ward_geom,
    max_addresses: int,
) -> pd.DataFrame:
    usecols = {"WKT", "address_id", "full_address", "territorial_authority", "town_city", "shape_X", "shape_Y"}
    addr = pd.read_csv(addresses_csv, usecols=lambda c: c in usecols)

    if town and "town_city" in addr.columns:
        addr = addr[addr["town_city"].astype(str) == town]
    if ta and "territorial_authority" in addr.columns:
        addr = addr[addr["territorial_authority"].astype(str) == ta]

    if "shape_X" in addr.columns and "shape_Y" in addr.columns:
        addr["lon"] = pd.to_numeric(addr["shape_X"], errors="coerce")
        addr["lat"] = pd.to_numeric(addr["shape_Y"], errors="coerce")
    else:
        pts = addr["WKT"].astype(str).apply(wkt.loads)
        addr["lon"] = pts.apply(lambda p: float(p.x))
        addr["lat"] = pts.apply(lambda p: float(p.y))

    addr = addr.dropna(subset=["lon", "lat"]).copy()

    if bbox:
        addr = addr[
            (addr["lon"] >= bbox.min_lon)
            & (addr["lon"] <= bbox.max_lon)
            & (addr["lat"] >= bbox.min_lat)
            & (addr["lat"] <= bbox.max_lat)
        ]
    if ward_geom is not None:
        addr = addr[addr.apply(lambda r: ward_geom.contains(Point(float(r["lon"]), float(r["lat"]))), axis=1)]

    if len(addr) > max_addresses:
        addr = addr.iloc[:max_addresses].copy()
    if addr.empty:
        raise SystemExit("No addresses left after filtering. Adjust --town/--ta/--bbox/--ward.")
    return addr


def load_cycleways(cycleways_geojson: str, bbox: Optional[BBox], ward_geom) -> List[LineString]:
    with open(cycleways_geojson, "r", encoding="utf-8") as f:
        data = json.load(f)

    bbox_poly = None
    if bbox:
        bbox_poly = shape(
            {
                "type": "Polygon",
                "coordinates": [[
                    [bbox.min_lon, bbox.min_lat], [bbox.max_lon, bbox.min_lat],
                    [bbox.max_lon, bbox.max_lat], [bbox.min_lon, bbox.max_lat],
                    [bbox.min_lon, bbox.min_lat],
                ]],
            }
        )

    lines: List[LineString] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        service_status = str(props.get("ServiceStatus", "")).strip().casefold()
        public_relevance = str(props.get("PublicRelevance", "")).strip().casefold()
        if service_status != "in service" or public_relevance != "public":
            continue

        geom_data = feat.get("geometry")
        if not geom_data:
            continue
        geom = shape(geom_data)
        if geom.is_empty:
            continue
        if bbox_poly is not None and not geom.intersects(bbox_poly):
            continue
        if ward_geom is not None and not geom.intersects(ward_geom):
            continue

        if geom.geom_type == "LineString":
            lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            lines.extend(list(geom.geoms))

    if not lines:
        raise SystemExit("No cycleway geometries left after filtering.")
    return lines


def compute_distances(addr: pd.DataFrame, cycle_lines: List[LineString]) -> pd.DataFrame:
    tf_to_m = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    cycle_lines_m = [transform(tf_to_m.transform, line) for line in cycle_lines]

    def nearest_dist(lon: float, lat: float) -> float:
        p_m = transform(tf_to_m.transform, Point(lon, lat))
        return float(min(line.distance(p_m) for line in cycle_lines_m))

    addr = addr.copy()
    addr["nearest_cycleway_m"] = [nearest_dist(float(lon), float(lat)) for lon, lat in zip(addr["lon"], addr["lat"])]
    return addr


def build_map(addr: pd.DataFrame, cycle_lines: List[LineString], out_html: str) -> None:
    centre_lat = float(addr["lat"].mean())
    centre_lon = float(addr["lon"].mean())

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13, control_scale=True, tiles="CartoDB positron")

    cycle_fg = folium.FeatureGroup(name="Cycleways", show=True)
    for line in cycle_lines:
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in line.coords],
            weight=3,
            color="#0a66c2",
            opacity=0.8,
        ).add_to(cycle_fg)
    cycle_fg.add_to(m)

    points = []
    for _, r in addr.iterrows():
        dist = float(r["nearest_cycleway_m"])
        color = dist_to_green_red_hex(dist, 400.0)
        popup = folium.Popup(
            f"<b>{r.get('full_address', 'Address')}</b><br/>"
            f"Nearest cycleway: {dist:.1f} m",
            max_width=300,
        )
        marker = folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=1,
            popup=popup,
        )
        marker.add_to(m)
        points.append({"dist": dist, "marker": marker.get_name()})

    within_400 = int((addr["nearest_cycleway_m"] <= 400).sum())
    within_800 = int((addr["nearest_cycleway_m"] <= 800).sum())
    between_400_800 = int(((addr["nearest_cycleway_m"] >= 400) & (addr["nearest_cycleway_m"] <= 800)).sum())

    panel_html = f"""
    <div id="cycling-controls" style="
      position: fixed; top: 12px; right: 12px; z-index: 9999;
      background: white; border: 1px solid #999; border-radius: 6px;
      padding: 10px; width: 300px; font: 13px/1.35 sans-serif;
      box-shadow: 0 1px 8px rgba(0,0,0,0.25);">
      <b>Cycleway accessibility</b><br/>
      Total addresses: {len(addr)}<br/>
      ≤400m: <span id="sum400">{within_400}</span><br/>
      ≤800m: <span id="sum800">{within_800}</span><br/>
      400–800m: <span id="sum400800">{between_400_800}</span>
      <hr style="margin:8px 0;"/>
      Colour max distance: <span id="thrLabel">400</span>m
      <input id="thrSlider" type="range" min="400" max="800" step="10" value="400" style="width:100%;"/>
      <div style="margin-top:4px;font-size:12px;color:#555;">Green=near, Yellow=at threshold, Red=over</div>
    </div>
    """

    points_json = json.dumps(points)
    script = f"""
    <script>
    (function() {{
      const points = {points_json};

      function colorForDist(dist, maxM) {{
        if (!Number.isFinite(dist) || dist > maxM) return '#ff0000';
        const t = Math.max(0, Math.min(1, dist / maxM));
        const r = Math.round(255 * t);
        return '#' + r.toString(16).padStart(2, '0') + 'ff00';
      }}

      function recolor(maxM) {{
        document.getElementById('thrLabel').textContent = String(maxM);
        for (const p of points) {{
          const c = colorForDist(p.dist, maxM);
          const mk = window[p.marker];
          if (!mk) continue;
          mk.setStyle({{color: c, fillColor: c}});
        }}
      }}

      document.getElementById('thrSlider').addEventListener('input', (e) => recolor(Number(e.target.value)));
      recolor(400);
    }})();
    </script>
    """

    m.get_root().html.add_child(Element(panel_html + script))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cycleway distance map and counts for addresses.")
    ap.add_argument("--addresses", default="lds-nz-addresses-CSV/nz-addresses.csv")
    ap.add_argument("--cycleways", default="Cycleway_(OpenData).geojson")
    ap.add_argument("--ward-geojson", default="Ward_(OpenData).geojson")
    ap.add_argument("--out", default="cycling.html")
    ap.add_argument("--bbox", nargs="+", type=float, default=None)
    ap.add_argument("--ward", default=None)
    ap.add_argument("--town", default=None)
    ap.add_argument("--ta", default=None)
    ap.add_argument("--max-addresses", type=int, default=10000)
    args = ap.parse_args()

    bbox = parse_bbox(args.bbox)
    ward_geom = load_ward_geometry(args.ward_geojson, args.ward) if args.ward else None

    addr = load_addresses(
        addresses_csv=args.addresses,
        town=args.town,
        ta=args.ta,
        bbox=bbox,
        ward_geom=ward_geom,
        max_addresses=args.max_addresses,
    )
    cycle_lines = load_cycleways(args.cycleways, bbox=bbox, ward_geom=ward_geom)
    addr = compute_distances(addr, cycle_lines)

    build_map(addr, cycle_lines, args.out)

    within_400 = int((addr["nearest_cycleway_m"] <= 400).sum())
    within_800 = int((addr["nearest_cycleway_m"] <= 800).sum())
    between_400_800 = int(((addr["nearest_cycleway_m"] >= 400) & (addr["nearest_cycleway_m"] <= 800)).sum())

    print(f"Wrote {args.out}")
    print(f"Total addresses analysed: {len(addr)}")
    print(f"Addresses within 400m of a cycleway: {within_400}")
    print(f"Addresses within 800m of a cycleway: {within_800}")
    print(f"Addresses between 400m and 800m of a cycleway: {between_400_800}")


if __name__ == "__main__":
    main()
