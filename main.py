#!/usr/bin/env python3
"""main.py

Build a standalone HTML map showing:
- NZ addresses (points) from lds-nz-addresses-CSV/nz-addresses.csv (WKT and/or shape_X/shape_Y)
- Roads (lines) from lds-nz-roads-addressing-CSV/nz-roads-addressing.csv (WKT)
- GTFS routes/shapes/stops from gtfs/*

Click an address marker:
- Popup shows nearest stop + distance in metres.
- Distances are computed in-browser using roads + tracks routing.
- Highlights the route taken on the map.

NEW:
- Roads are drawn thicker and coloured green→red by distance to nearest stop.
- Address markers are coloured the same way.
- UI slider lets you change the colour-scale max from 400→800m (live recolour).
- Bus stops are draggable: on drag-end, nearest-stop distances are recalculated
  in the browser via the routable graph for consistency.

Dependencies:
  pip install pandas numpy shapely pyproj folium networkx scipy

Usage example:
  python3 main.py \
    --addresses lds-nz-addresses-CSV/nz-addresses.csv \
    --roads lds-nz-roads-road-section-geometry-CSV/nz-roads-road-section-geometry.csv \
    --gtfs gtfs \
    --out map.html

Notes:
- Plotting all NZ addresses in one HTML is not practical. Use --bbox/--ward/--town/--ta and/or --max-addresses.
- Road network is derived from addressing roads; it may detour where footpaths/accessways exist.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import json

import numpy as np
import pandas as pd
import folium
import networkx as nx
from shapely import wkt
from shapely.geometry import LineString, Point, shape
from pyproj import Transformer
from scipy.spatial import cKDTree
from folium import Element


# ----------------------------
# Geometry / distance helpers
# ----------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def natural_route_sort_key(label: str) -> Tuple[int, int, str]:
    """Sort labels by leading route number when present, then text."""
    s = (label or "").strip()
    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
        elif not ch.isspace():
            break
    if digits:
        return (0, int("".join(digits)), s.lower())
    return (1, 0, s.lower())


def dist_to_green_red_hex(dist_m: Optional[float], max_m: float = 400.0) -> str:
    """Map distance to hex color from green (0) to red (max_m).

    If dist_m is None, treat as max.
    Values > max are clamped.
    """
    if dist_m is None or not math.isfinite(dist_m):
        dist_m = max_m
    t = clamp(dist_m / max_m, 0.0, 1.0)
    r = int(round(255 * t))
    g = int(round(255 * (1.0 - t)))
    b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres for two WGS84 coordinates."""
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains_lonlat(self, lon: float, lat: float) -> bool:
        return (self.min_lon <= lon <= self.max_lon) and (self.min_lat <= lat <= self.max_lat)


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
        raise ValueError("bbox invalid: min_lat > max_lat (remember: in NZ, more negative is south)")
    return BBox(min_lon, min_lat, max_lon, max_lat)


def load_ward_geometry(ward_geojson: str, ward_name: str):
    """Load a ward geometry from Ward_(OpenData).geojson by WardNameDescription."""
    with open(ward_geojson, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    ward_name_lc = ward_name.strip().lower()
    for feat in features:
        props = feat.get("properties", {}) or {}
        name = str(props.get("WardNameDescription", "")).strip()
        if name.lower() == ward_name_lc:
            geom = feat.get("geometry")
            if not geom:
                raise ValueError(f"Ward {ward_name!r} has no geometry in {ward_geojson}")
            return shape(geom)

    raise ValueError(f"Ward {ward_name!r} not found in {ward_geojson}")


# ----------------------------
# GTFS loading
# ----------------------------

def load_gtfs(gtfs_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns routes, stops, shapes, trips."""

    def r(path: str) -> str:
        p = os.path.join(gtfs_dir, path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing GTFS file: {p}")
        return p

    routes = pd.read_csv(r("routes.txt"))
    stops = pd.read_csv(r("stops.txt"))
    shapes = pd.read_csv(r("shapes.txt"))
    trips = pd.read_csv(r("trips.txt"))

    # Normalise route_color
    if "route_color" in routes.columns:
        routes["route_color"] = routes["route_color"].fillna("").astype(str).str.strip()
    else:
        routes["route_color"] = ""

    return routes, stops, shapes, trips


def build_shapes_by_route(
    routes: pd.DataFrame, shapes: pd.DataFrame, trips: pd.DataFrame
) -> Dict[str, List[List[Tuple[float, float]]]]:
    """Map route_id -> list of polylines (each polyline is list of (lat, lon))."""

    if "shape_id" not in trips.columns:
        raise ValueError("GTFS trips.txt must include shape_id to draw route shapes.")
    if "route_id" not in trips.columns:
        raise ValueError("GTFS trips.txt must include route_id to draw route shapes.")

    route_to_shapes: Dict[str, List[str]] = (
        trips.dropna(subset=["route_id", "shape_id"])
        .drop_duplicates(subset=["route_id", "shape_id"])
        .groupby("route_id")["shape_id"]
        .apply(lambda s: [str(x) for x in s.tolist()])
        .to_dict()
    )

    if not {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}.issubset(set(shapes.columns)):
        raise ValueError(
            "GTFS shapes.txt missing required columns: shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence"
        )

    shapes_sorted = shapes.sort_values(["shape_id", "shape_pt_sequence"])

    shape_lines: Dict[str, List[Tuple[float, float]]] = {}
    for sid, grp in shapes_sorted.groupby("shape_id"):
        pts = list(zip(grp["shape_pt_lat"].astype(float).tolist(), grp["shape_pt_lon"].astype(float).tolist()))
        cleaned: List[Tuple[float, float]] = []
        last = None
        for p in pts:
            if last is None or p != last:
                cleaned.append(p)
            last = p
        if len(cleaned) >= 2:
            shape_lines[str(sid)] = cleaned

    out: Dict[str, List[List[Tuple[float, float]]]] = {}
    for rid, sids in route_to_shapes.items():
        polylines = []
        for sid in sids:
            if sid in shape_lines:
                polylines.append(shape_lines[sid])
        if polylines:
            out[str(rid)] = polylines

    return out


def build_stop_distance_details(
    trips: pd.DataFrame,
    stops: pd.DataFrame,
    stop_times: Optional[pd.DataFrame],
) -> Dict[str, List[dict]]:
    """Build per-stop neighbour-distance details for popup display.

    Returns stop_id -> list of records with keys:
      route_id, direction ("to_next"/"from_prev"), other_stop_name, distance_m.
    Uses one representative trip per route (most stops) to avoid excessive overlap.
    """
    if stop_times is None or stop_times.empty:
        return {}
    if not {"trip_id", "route_id"}.issubset(trips.columns):
        return {}
    needed = {"trip_id", "stop_id", "stop_sequence"}
    if not needed.issubset(stop_times.columns):
        return {}

    trip_routes = trips[["trip_id", "route_id"]].dropna().astype(str)
    merged = stop_times[["trip_id", "stop_id", "stop_sequence"]].dropna().copy()
    merged["trip_id"] = merged["trip_id"].astype(str)
    merged["stop_id"] = merged["stop_id"].astype(str)
    merged["stop_sequence"] = pd.to_numeric(merged["stop_sequence"], errors="coerce")
    merged = merged.dropna(subset=["stop_sequence"])
    merged = merged.merge(trip_routes, on="trip_id", how="inner")
    if merged.empty:
        return {}

    stop_coords = (
        stops[["stop_id", "stop_lat", "stop_lon", "stop_name"]]
        .dropna(subset=["stop_id", "stop_lat", "stop_lon"])
        .copy()
    )
    stop_coords["stop_id"] = stop_coords["stop_id"].astype(str)
    stop_coords["stop_lat"] = pd.to_numeric(stop_coords["stop_lat"], errors="coerce")
    stop_coords["stop_lon"] = pd.to_numeric(stop_coords["stop_lon"], errors="coerce")
    stop_coords["stop_name"] = stop_coords["stop_name"].fillna("").astype(str)
    stop_coords = stop_coords.dropna(subset=["stop_lat", "stop_lon"]).drop_duplicates(subset=["stop_id"])
    stop_coord_name_map: Dict[str, Tuple[float, float, str]] = {
        str(r["stop_id"]): (float(r["stop_lat"]), float(r["stop_lon"]), str(r["stop_name"]).strip())
        for _, r in stop_coords.iterrows()
    }

    out: Dict[str, List[dict]] = {}
    for route_id, route_grp in merged.groupby("route_id"):
        best_trip_stops: Optional[pd.DataFrame] = None
        for _, trip_grp in route_grp.groupby("trip_id"):
            ordered = trip_grp.sort_values("stop_sequence")
            if best_trip_stops is None or len(ordered) > len(best_trip_stops):
                best_trip_stops = ordered
        if best_trip_stops is None or len(best_trip_stops) < 2:
            continue

        stop_ids = best_trip_stops["stop_id"].astype(str).tolist()
        for a, b in zip(stop_ids[:-1], stop_ids[1:]):
            if a == b:
                continue
            if a not in stop_coord_name_map or b not in stop_coord_name_map:
                continue
            lat_a, lon_a, name_a = stop_coord_name_map[a]
            lat_b, lon_b, name_b = stop_coord_name_map[b]
            dist_m = float(haversine_m(lat_a, lon_a, lat_b, lon_b))

            out.setdefault(a, []).append({
                "route_id": str(route_id),
                "direction": "to_next",
                "other_stop_name": name_b or b,
                "distance_m": dist_m,
            })
            out.setdefault(b, []).append({
                "route_id": str(route_id),
                "direction": "from_prev",
                "other_stop_name": name_a or a,
                "distance_m": dist_m,
            })

    for sid in list(out.keys()):
        out[sid].sort(key=lambda x: (str(x.get("route_id", "")), str(x.get("direction", "")), str(x.get("other_stop_name", ""))))

    return out


# ----------------------------
# Roads graph building
# ----------------------------

def densify_linestring(ls: LineString, max_segment_m: float, tf_to_m: Transformer) -> List[Tuple[float, float]]:
    """Densify a lon/lat LineString so no segment is longer than max_segment_m.

    Returns list of (lon, lat) points along the line.
    """
    coords = list(ls.coords)
    if len(coords) < 2:
        return coords

    out: List[Tuple[float, float]] = [coords[0]]
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        # coords are (lon, lat)
        X1, Y1 = tf_to_m.transform(y1, x1)  # lat, lon -> metres
        X2, Y2 = tf_to_m.transform(y2, x2)
        seg = math.hypot(X2 - X1, Y2 - Y1)

        if seg <= max_segment_m:
            out.append((x2, y2))
            continue

        n = max(2, int(math.ceil(seg / max_segment_m)) + 1)
        for i in range(1, n):
            t = i / (n - 1)
            out.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))

    return out


def build_road_graph(
    roads_df: pd.DataFrame,
    bbox: Optional[BBox],
    ward_geom,
    max_roads: int,
    densify_m: float,
) -> Tuple[nx.Graph, np.ndarray]:
    """Build undirected weighted graph from road WKT MULTILINESTRING/LINESTRING.

    Source: lds-nz-roads-road-section-geometry-CSV

    Nodes: (lon, lat) tuples.
    Edge weight: metres (NZTM2000 EPSG:2193).

    Edge attr: road=<best-effort label derived from road_section_id/road_type/geometry_class>.
    """

    # GD2000 lon/lat -> NZTM2000 metres
    tf_to_m = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=False)

    g = nx.Graph()
    used = 0

    for _, row in roads_df.iterrows():
        if used >= max_roads:
            break

        geom = row.get("WKT")
        if not isinstance(geom, str) or not geom:
            continue

        # The road-section-geometry dataset doesn't include full_road_name.
        # Use a best-effort label for segment/run counting & tooltips.
        rsid = str(row.get("road_section_id", "")).strip()
        rtype = str(row.get("road_type", "")).strip()
        gcls = str(row.get("geometry_class", "")).strip()
        rid = str(row.get("road_section_geometry_id", "")).strip()
        parts = [p for p in [rtype, gcls] if p and p.lower() != "nan"]
        base = " / ".join(parts) if parts else "road"
        if rsid and rsid.lower() != "nan":
            road_name = f"{base} (section {rsid})"
        elif rid and rid.lower() != "nan":
            road_name = f"{base} (geom {rid})"
        else:
            road_name = base

        try:
            sh = wkt.loads(geom)
        except Exception:
            continue

        if sh.geom_type == "LineString":
            lines = [sh]
        elif sh.geom_type == "MultiLineString":
            lines = list(sh.geoms)
        else:
            continue

        for ls in lines:
            coords = list(ls.coords)
            if len(coords) < 2:
                continue

            if bbox and not any(bbox.contains_lonlat(x, y) for (x, y) in coords):
                continue
            if ward_geom is not None and not ward_geom.intersects(ls):
                continue

            dense = densify_linestring(ls, max_segment_m=densify_m, tf_to_m=tf_to_m)
            if len(dense) < 2:
                continue

            for (lon1, lat1), (lon2, lat2) in zip(dense[:-1], dense[1:]):
                x1, y1 = tf_to_m.transform(lat1, lon1)
                x2, y2 = tf_to_m.transform(lat2, lon2)
                w = float(math.hypot(x2 - x1, y2 - y1))

                n1 = (float(lon1), float(lat1))
                n2 = (float(lon2), float(lat2))
                if n1 == n2:
                    continue

                if g.has_edge(n1, n2):
                    if w < g[n1][n2].get("weight", w):
                        g[n1][n2]["weight"] = w
                        g[n1][n2]["road"] = road_name
                else:
                    g.add_edge(n1, n2, weight=w, road=road_name)

        used += 1

    nodes = np.array([[n[0], n[1]] for n in g.nodes()], dtype=np.float64)
    return g, nodes


def load_track_lines_geojson(path: str, bbox: Optional[BBox]) -> List[Tuple[LineString, str]]:
    """Load track geometries from GeoJSON and return routable linework.

    For polygonal tracks, we use the polygon boundary to approximate walkable
    track paths.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    feats = data.get("features", []) if isinstance(data, dict) else []
    out: List[Tuple[LineString, str]] = []

    for ft in feats:
        if not isinstance(ft, dict):
            continue
        geom = ft.get("geometry")
        if not geom:
            continue
        props = ft.get("properties") or {}

        track_id = str(props.get("TrackID", "")).strip()
        site_name = str(props.get("SiteName", "")).strip()
        name = site_name if site_name else f"track {track_id}" if track_id and track_id.lower() != "nan" else "track"

        try:
            sh = shape(geom)
        except Exception:
            continue

        geoms: List[LineString] = []
        if sh.geom_type == "LineString":
            geoms = [sh]
        elif sh.geom_type == "MultiLineString":
            geoms = [g for g in sh.geoms if g.geom_type == "LineString"]
        elif sh.geom_type in {"Polygon", "MultiPolygon"}:
            b = sh.boundary
            if b.geom_type == "LineString":
                geoms = [b]
            elif b.geom_type == "MultiLineString":
                geoms = [g for g in b.geoms if g.geom_type == "LineString"]

        for ls in geoms:
            coords = list(ls.coords)
            if len(coords) < 2:
                continue
            if bbox and not any(bbox.contains_lonlat(x, y) for (x, y) in coords):
                continue
            out.append((ls, name))

    return out


def add_lines_to_graph(
    g: nx.Graph,
    lines: List[Tuple[LineString, str]],
    densify_m: float,
    tf_to_m: Transformer,
    kind: str,
) -> Set[Tuple[float, float]]:
    """Add named linework to graph and return added nodes."""
    added_nodes: Set[Tuple[float, float]] = set()
    for ls, name in lines:
        dense = densify_linestring(ls, max_segment_m=densify_m, tf_to_m=tf_to_m)
        if len(dense) < 2:
            continue

        for (lon1, lat1), (lon2, lat2) in zip(dense[:-1], dense[1:]):
            x1, y1 = tf_to_m.transform(lat1, lon1)
            x2, y2 = tf_to_m.transform(lat2, lon2)
            w = float(math.hypot(x2 - x1, y2 - y1))

            n1 = (float(lon1), float(lat1))
            n2 = (float(lon2), float(lat2))
            if n1 == n2:
                continue

            added_nodes.add(n1)
            added_nodes.add(n2)

            if g.has_edge(n1, n2):
                if w < g[n1][n2].get("weight", w):
                    g[n1][n2]["weight"] = w
                    g[n1][n2]["road"] = name
                    g[n1][n2]["kind"] = kind
            else:
                g.add_edge(n1, n2, weight=w, road=name, kind=kind)

    return added_nodes


def connect_track_nodes_to_graph(
    g: nx.Graph,
    track_nodes: Set[Tuple[float, float]],
    base_nodes: np.ndarray,
    tf_to_m: Transformer,
    max_link_m: float = 20.0,
) -> None:
    """Stitch track nodes to nearest existing graph nodes when very close.

    Many park tracks are digitised as polygons that stop just short of roads.
    Without a short connector edge, routing cannot traverse tracks even when
    they are effectively connected in reality.
    """
    if not track_nodes or base_nodes.size == 0:
        return

    base_kd = cKDTree(base_nodes)

    for lon, lat in track_nodes:
        _, idx = base_kd.query([lon, lat], k=1)
        n2 = (float(base_nodes[int(idx), 0]), float(base_nodes[int(idx), 1]))
        n1 = (float(lon), float(lat))
        if n1 == n2:
            continue

        x1, y1 = tf_to_m.transform(lat, lon)
        x2, y2 = tf_to_m.transform(n2[1], n2[0])
        w = float(math.hypot(x2 - x1, y2 - y1))
        if w > max_link_m:
            continue

        if g.has_edge(n1, n2):
            continue

        g.add_edge(n1, n2, weight=w, road="track connector", kind="track")


def snap_to_graph_node(kdtree: cKDTree, nodes_arr: np.ndarray, lon: float, lat: float) -> Tuple[Tuple[float, float], float]:
    """Snap a lon/lat point to nearest graph node in lon/lat space."""
    dist, idx = kdtree.query([lon, lat], k=1)
    node = (float(nodes_arr[idx, 0]), float(nodes_arr[idx, 1]))
    return node, float(dist)


def shortest_path_details(
    g: nx.Graph, start: Tuple[float, float], end: Tuple[float, float]
) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]], Optional[int], bool]:
    """Compute shortest path."""
    try:
        node_path = nx.shortest_path(g, start, end, weight="weight")
    except Exception:
        return None, None, None, False

    dist = 0.0
    segs = 0
    last_road = None
    uses_track = False

    for u, v in zip(node_path[:-1], node_path[1:]):
        data = g.get_edge_data(u, v) or {}
        dist += float(data.get("weight", 0.0))
        road = str(data.get("road", "")).strip()
        if str(data.get("kind", "road")) == "track":
            uses_track = True
        if road != last_road:
            segs += 1
            last_road = road

    return float(dist), node_path, int(segs), uses_track


def linestring_sample_points_latlon(
    ls: LineString,
    tf_to_m: Transformer,
    sample_every_m: float = 60.0,
) -> List[List[float]]:
    """Sample a LineString into [lat, lon] points for browser-side distance updates."""
    dense = densify_linestring(ls, max_segment_m=sample_every_m, tf_to_m=tf_to_m)
    return [[float(lat), float(lon)] for lon, lat in dense]


def add_ui_and_interaction_js(m: folium.Map) -> None:
    """Inject UI + client-side distance/colour recompute on stop drag."""
    js = r"""
<script>
(function() {
  // ---------- UI state ----------
  window.__gradMaxM = 400;
  window.__activeRouteFilter = new Set();
  window.__showChangesOnly = false;

  // ---------- helpers ----------
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function distToColorHex(distM, maxM) {
    if (distM == null || !isFinite(distM)) distM = maxM;
    const t = clamp(distM / maxM, 0.0, 1.0);
    const r = Math.round(255 * t);
    const g = Math.round(255 * (1.0 - t));
    const b = 0;
    const hex = (n) => n.toString(16).padStart(2, '0');
    return `#${hex(r)}${hex(g)}${hex(b)}`;
  }

  // Haversine metres
  function haversineM(lat1, lon1, lat2, lon2) {
    const R = 6371000.0;
    const toRad = (d) => d * Math.PI / 180.0;
    const phi1 = toRad(lat1);
    const phi2 = toRad(lat2);
    const dphi = toRad(lat2 - lat1);
    const dl = toRad(lon2 - lon1);
    const a = Math.sin(dphi/2)**2 + Math.cos(phi1)*Math.cos(phi2)*Math.sin(dl/2)**2;
    return 2 * R * Math.asin(Math.sqrt(a));
  }

  // ---------- simple grid index ----------
  // Cell size in degrees (roughly ~200m lat-wise)
  const CELL = 0.002;

  function keyFor(lat, lon) {
    const i = Math.floor(lat / CELL);
    const j = Math.floor(lon / CELL);
    return `${i},${j}`;
  }

  function stopAllowedByFilter(s) {
    if (!window.__activeRouteFilter || window.__activeRouteFilter.size === 0) return true;
    const routeIds = Array.isArray(s.routeIds) ? s.routeIds : [];
    for (const rid of routeIds) {
      if (window.__activeRouteFilter.has(String(rid))) return true;
    }
    return false;
  }

  function ensureGraphRuntime() {
    if (window.__graphRt) return window.__graphRt;
    const gd = window.__graphData;
    if (!gd || !Array.isArray(gd.nodes) || !Array.isArray(gd.edges)) return null;
    const n = gd.nodes.length;
    const adj = Array.from({ length: n }, () => []);
    for (const e of gd.edges) {
      if (!Array.isArray(e) || e.length < 3) continue;
      const u = e[0] | 0;
      const v = e[1] | 0;
      const w = +e[2];
      if (!isFinite(w)) continue;
      if (u < 0 || u >= n || v < 0 || v >= n) continue;
      adj[u].push([v, w]);
      adj[v].push([u, w]);
    }
    const grid = new Map();
    for (let i = 0; i < n; i++) {
      const nd = gd.nodes[i];
      const k = keyFor(nd[0], nd[1]);
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(i);
    }
    window.__graphRt = { nodes: gd.nodes, adj, grid };
    return window.__graphRt;
  }

  function nearestGraphNode(lat, lon) {
    const rt = ensureGraphRuntime();
    if (!rt || !rt.nodes.length) return { idx: null, snapM: null };
    const ci = Math.floor(lat / CELL);
    const cj = Math.floor(lon / CELL);
    let bestIdx = null;
    let bestD = Infinity;
    for (let ring = 0; ring <= 15; ring++) {
      let found = false;
      for (let di = -ring; di <= ring; di++) {
        for (let dj = -ring; dj <= ring; dj++) {
          if (Math.abs(di) !== ring && Math.abs(dj) !== ring) continue;
          const list = rt.grid.get(`${ci + di},${cj + dj}`);
          if (!list) continue;
          found = true;
          for (const idx of list) {
            const nd = rt.nodes[idx];
            const d = haversineM(lat, lon, nd[0], nd[1]);
            if (d < bestD) { bestD = d; bestIdx = idx; }
          }
        }
      }
      if (found && ring >= 1) break;
    }
    if (bestIdx == null) {
      for (let i = 0; i < rt.nodes.length; i++) {
        const nd = rt.nodes[i];
        const d = haversineM(lat, lon, nd[0], nd[1]);
        if (d < bestD) { bestD = d; bestIdx = i; }
      }
    }
    return { idx: bestIdx, snapM: isFinite(bestD) ? bestD : null };
  }

  function recomputeStopField() {
    const rt = ensureGraphRuntime();
    if (!rt || !window.__stopData) return null;
    const EPS = 1e-6;
    const n = rt.nodes.length;
    const dist = new Array(n).fill(Infinity);
    const parent = new Array(n).fill(-1);
    const sourceStopIdx = new Array(n).fill(-1);
    const sourceStopKey = new Array(n).fill('');
    const heap = [];

    function stableStopKey(si, s) {
      if (s && s.id != null && String(s.id).trim() !== '' && String(s.id) !== 'nan') return `id:${String(s.id)}`;
      if (s && s.stopCode != null && String(s.stopCode).trim() !== '' && String(s.stopCode) !== 'nan') return `code:${String(s.stopCode)}`;
      if (s && s.name != null && String(s.name).trim() !== '' && String(s.name) !== 'nan') return `name:${String(s.name)}`;
      return `idx:${si}`;
    }

    function shouldPrefer(nextDist, nextKey, curDist, curKey) {
      if (nextDist < (curDist - EPS)) return true;
      if (Math.abs(nextDist - curDist) <= EPS) {
        if (!curKey) return true;
        if (!nextKey) return false;
        return nextKey < curKey;
      }
      return false;
    }

    function push(item) {
      heap.push(item);
      let i = heap.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heap[p][0] <= heap[i][0]) break;
        [heap[p], heap[i]] = [heap[i], heap[p]];
        i = p;
      }
    }
    function pop() {
      if (!heap.length) return null;
      const out = heap[0];
      const last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        let i = 0;
        while (true) {
          const l = i * 2 + 1;
          const r = l + 1;
          let m = i;
          if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
          if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
          if (m === i) break;
          [heap[m], heap[i]] = [heap[i], heap[m]];
          i = m;
        }
      }
      return out;
    }

    for (let si = 0; si < window.__stopData.length; si++) {
      const s = window.__stopData[si];
      if (!stopAllowedByFilter(s)) continue;
      const snap = nearestGraphNode(s.lat, s.lon);
      if (snap.idx == null || snap.snapM == null) continue;
      const key = stableStopKey(si, s);
      s.__snapNode = snap.idx;
      s.__snapM = snap.snapM;
      if (shouldPrefer(snap.snapM, key, dist[snap.idx], sourceStopKey[snap.idx])) {
        dist[snap.idx] = snap.snapM;
        sourceStopIdx[snap.idx] = si;
        sourceStopKey[snap.idx] = key;
        parent[snap.idx] = -1;
        push([snap.snapM, snap.idx]);
      }
    }

    while (heap.length) {
      const item = pop();
      if (!item) break;
      const d = item[0];
      const u = item[1];
      if (d !== dist[u]) continue;
      for (const [v, w] of rt.adj[u]) {
        const nd = d + w;
        const srcIdx = sourceStopIdx[u];
        const src = (srcIdx != null && srcIdx >= 0 && srcIdx < window.__stopData.length) ? window.__stopData[srcIdx] : null;
        const srcKey = stableStopKey(srcIdx, src);
        if (shouldPrefer(nd, srcKey, dist[v], sourceStopKey[v])) {
          dist[v] = nd;
          parent[v] = u;
          sourceStopIdx[v] = srcIdx;
          sourceStopKey[v] = srcKey;
          push([nd, v]);
        }
      }
    }

    window.__stopField = { dist, parent, sourceStopIdx };
    return window.__stopField;
  }

  // ---------- global registries populated by Python ----------
  // window.__stopData = [{id, lat, lon, name, routeIds}]
  // window.__routeOptions = [{id, label}]
  // window.__addrData = [{id, lat, lon, markerName, basePopupHtml, markerRefName, distM}]
  // window.__roadData = [{polyRefName, dM, samples}]
  // window.__trackData = [{polyRefName, dM, samples}]

  function getByName(name) {
    try { return window[name]; } catch { return null; }
  }

  function updateLegendMax(maxM) {
    const lbl = document.getElementById('__gradMaxLabel');
    if (lbl) lbl.textContent = `${maxM} m`;
  }

  function recalcLineDistancesFromStops(lineData) {
    if (!lineData || !window.__stopData || !Array.isArray(window.__stopData)) return;
    const sf = window.__stopField || recomputeStopField();
    if (!sf) return;
    for (const line of lineData) {
      const pts = Array.isArray(line.samples) ? line.samples : [];
      if (!pts.length) {
        line.dM = null;
        continue;
      }
      let best = Infinity;
      for (const pt of pts) {
        if (!Array.isArray(pt) || pt.length < 2) continue;
        const snap = nearestGraphNode(pt[0], pt[1]);
        if (snap.idx == null || snap.snapM == null) continue;
        const graphD = sf.dist[snap.idx];
        if (!isFinite(graphD)) continue;
        const d = snap.snapM + graphD;
        if (d < best) best = d;
      }
      line.dM = isFinite(best) ? best : null;
    }
  }

  function recolorAll(maxM) {
    // Addresses
    if (window.__addrData) {
      for (const a of window.__addrData) {
        const marker = getByName(a.markerRefName);
        if (!marker) continue;
        const baseServed = !!a.baseServed;
        const currentServed = (a.distM != null && isFinite(a.distM) && a.distM <= maxM);
        let c = distToColorHex(a.distM, maxM);

        let style = { color: c, fillColor: c, opacity: 1.0, fillOpacity: 0.95 };
        if (window.__showChangesOnly) {
          const changed = (!baseServed && currentServed) || (baseServed && !currentServed);
          if (!changed) {
            // Hide unchanged markers entirely so changed ones stand out.
            style = { color: c, fillColor: c, opacity: 0.0, fillOpacity: 0.0 };
          } else if (!baseServed && currentServed) {
            style = { color: '#2e7d32', fillColor: '#2e7d32', opacity: 1.0, fillOpacity: 0.95 };
          } else {
            style = { color: '#c62828', fillColor: '#c62828', opacity: 1.0, fillOpacity: 0.95 };
          }
        }

        if (marker.setStyle) {
          marker.setStyle(style);
        }
      }
    }

    // Roads (we store per-poly distance as dM)
    if (window.__roadData) {
      for (const r of window.__roadData) {
        const poly = getByName(r.polyRefName);
        if (!poly) continue;
        const c = distToColorHex(r.dM, maxM);
        if (poly.setStyle) {
          const nextOpacity = window.__showChangesOnly ? 0.0 : 0.75;
          poly.setStyle({ color: c, opacity: nextOpacity });
        }
      }
    }

    // Tracks
    if (window.__trackData) {
      for (const t of window.__trackData) {
        const poly = getByName(t.polyRefName);
        if (!poly) continue;
        const c = distToColorHex(t.dM, maxM);
        if (poly.setStyle) {
          const nextOpacity = window.__showChangesOnly ? 0.0 : 0.8;
          poly.setStyle({ color: c, opacity: nextOpacity });
        }
      }
    }
  }

  // ---------- serviceability stats ----------
  function countFilteredStops() {
    if (!window.__stopData || !Array.isArray(window.__stopData)) return 0;
    if (!window.__activeRouteFilter || window.__activeRouteFilter.size === 0) return window.__stopData.length;
    let n = 0;
    for (const s of window.__stopData) {
      const routeIds = Array.isArray(s.routeIds) ? s.routeIds : [];
      let matches = false;
      for (const rid of routeIds) {
        if (window.__activeRouteFilter.has(String(rid))) { matches = true; break; }
      }
      if (matches) n++;
    }
    return n;
  }

  function computeServedCounts(thresholdM) {
    const total = (window.__addrData && window.__addrData.length) ? window.__addrData.length : 0;
    let served = 0;
    if (window.__addrData) {
      for (const a of window.__addrData) {
        if (a.distM != null && isFinite(a.distM) && a.distM <= thresholdM) served++;
      }
    }
    const unserved = total - served;
    const stops = countFilteredStops();
    return { total, served, unserved, stops, thresholdM };
  }

  function ensureStatsPanel() {
    let panel = document.getElementById('__svcPanel');
    if (panel) return panel;

    panel = document.createElement('div');
    panel.id = '__svcPanel';
    panel.style.position = 'fixed';
    panel.style.bottom = '18px';
    panel.style.right = '18px';
    panel.style.zIndex = '9999';
    panel.style.width = '320px';
    panel.style.maxWidth = 'calc(100vw - 40px)';
    panel.style.background = 'rgba(255,255,255,0.92)';
    panel.style.padding = '10px 12px';
    panel.style.borderRadius = '10px';
    panel.style.boxShadow = '0 2px 10px rgba(0,0,0,0.12)';
    panel.style.fontFamily = 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
    panel.style.fontSize = '12px';
    panel.style.lineHeight = '1.25';

    panel.innerHTML = `
      <div style="display:flex; align-items:baseline; justify-content:space-between; gap:10px;">
        <div style="font-weight:700;">Starting metrics</div>
        <div style="color:#555; opacity:0.9;">≤ <span id="__svcStartThreshold">—</span> m</div>
      </div>
      <div style="margin-top:8px; display:grid; grid-template-columns: 1fr auto; gap:6px 10px;">
        <div>Bus stops in area</div><div id="__svcStartStops" style="font-weight:700; text-align:right;">—</div>
        <div>Addresses served</div><div id="__svcStartServed" style="font-weight:700; text-align:right;">—</div>
        <div>Addresses not served</div><div id="__svcStartUnserved" style="font-weight:700; text-align:right;">—</div>
      </div>
      <div style="margin-top:10px; font-weight:700;">Change log</div>
      <div id="__svcLog" style="margin-top:6px; max-height:260px; overflow:auto; padding-right:4px;"></div>
      <div style="margin-top:10px; font-weight:700;">Totals after changes</div>
      <div style="display:flex; align-items:baseline; justify-content:space-between; gap:10px; margin-top:6px;">
        <div style="font-weight:700;">Serviceable area</div>
        <div style="color:#555; opacity:0.9;">≤ <span id="__svcThreshold">—</span> m</div>
      </div>
      <div style="margin-top:8px; display:grid; grid-template-columns: 1fr auto; gap:6px 10px;">
        <div>Bus stops in area</div><div id="__svcStops" style="font-weight:700; text-align:right;">—</div>
        <div>Addresses served</div><div id="__svcServed" style="font-weight:700; text-align:right;">—</div>
        <div>Addresses not served</div><div id="__svcUnserved" style="font-weight:700; text-align:right;">—</div>
      </div>
    `;

    document.body.appendChild(panel);
    return panel;
  }

  function fmtDelta(n) {
    if (!isFinite(n) || n === 0) return '±0';
    return (n > 0 ? `+${n}` : `${n}`);
  }

  function appendLogLine(key, reason, baseStats, nextStats) {
    const log = document.getElementById('__svcLog');
    if (!log) return;

    if (!window.__logEntryEls) window.__logEntryEls = {};

    const dServed = nextStats.served - (baseStats ? baseStats.served : nextStats.served);

    let line = window.__logEntryEls[key] || null;

    // If the action's net effect returned to zero, remove it from the log.
    if (isFinite(dServed) && dServed === 0) {
      if (line && line.parentNode) line.parentNode.removeChild(line);
      delete window.__logEntryEls[key];
      return;
    }

    const isNew = !line;
    if (!line) {
      line = document.createElement('div');
      line.style.borderTop = '1px solid rgba(0,0,0,0.08)';
      line.style.paddingTop = '6px';
      line.style.marginTop = '6px';
      window.__logEntryEls[key] = line;
    }

    // Show only net change in served, no timestamp.
    const deltaTxt = (isFinite(dServed) ? `${fmtDelta(dServed)} served` : '±0 served');

    line.innerHTML = `
      <div style="display:flex; justify-content:space-between; gap:10px; align-items:baseline;">
        <div style="font-weight:600;">${reason}</div>
        <div style="color:#555; font-weight:600;">${deltaTxt}</div>
      </div>
    `;

    // Only add to DOM once. Keep the existing ordering after that.
    if (isNew) {
      // newest at top
      log.prepend(line);
    }
  }

  function renderStats(nextStats) {
    ensureStatsPanel();
    const th = document.getElementById('__svcThreshold');
    const st = document.getElementById('__svcStops');
    const se = document.getElementById('__svcServed');
    const un = document.getElementById('__svcUnserved');

    if (th) th.textContent = `${Math.round(nextStats.thresholdM)}`;
    if (st) st.textContent = nextStats.stops.toLocaleString();
    if (se) se.textContent = nextStats.served.toLocaleString();
    if (un) un.textContent = nextStats.unserved.toLocaleString();
  }

  function renderStartingStats(startStats) {
    ensureStatsPanel();
    const th = document.getElementById('__svcStartThreshold');
    const st = document.getElementById('__svcStartStops');
    const se = document.getElementById('__svcStartServed');
    const un = document.getElementById('__svcStartUnserved');

    if (th) th.textContent = `${Math.round(startStats.thresholdM)}`;
    if (st) st.textContent = startStats.stops.toLocaleString();
    if (se) se.textContent = startStats.served.toLocaleString();
    if (un) un.textContent = startStats.unserved.toLocaleString();
  }

  function updateSummary(reason, logKey) {
    const nextStats = computeServedCounts(window.__gradMaxM);

    if (!window.__startingSvcStats) {
      window.__startingSvcStats = nextStats;
      renderStartingStats(nextStats);
    }

    // For each log key, keep a stable baseline so entries show net change from
    // the original state and disappear when they return to zero.
    if (!window.__baseSvcStatsByKey) window.__baseSvcStatsByKey = {};

    renderStats(nextStats);

    if (reason) {
      const key = logKey || reason;
      if (!window.__baseSvcStatsByKey[key]) {
        window.__baseSvcStatsByKey[key] = window.__startingSvcStats || nextStats;
      }
      appendLogLine(key, reason, window.__baseSvcStatsByKey[key], nextStats);
    }

    window.__lastSvcStats = nextStats;
  }

  function updateAddressPopups() {
    if (!window.__addrData) return;
    for (const a of window.__addrData) {
      const marker = getByName(a.markerRefName);
      if (!marker) continue;
      // Replace only the fields we control. Keep the rest of the HTML.
      const dTxt = (a.distM == null) ? 'N/A' : `${Math.round(a.distM).toLocaleString()} m`;
      const nTxt = a.nearestStopName || 'N/A';
      const methodTxt = a.methodology || 'N/A';
      const html = a.basePopupHtml
        .replace('__NEAREST__', nTxt)
        .replace('__DIST__', dTxt)
        .replace('__METHOD__', methodTxt);

      if (marker.getPopup && marker.getPopup()) {
        marker.getPopup().setContent(html);
      }
    }
  }

  function recalcAddressesFromStops(reason, logKey, methodOverride) {
    if (!window.__stopData || !window.__addrData) return;
    const sf = recomputeStopField();
    const rt = ensureGraphRuntime();
    if (!sf || !rt) return;

    for (const a of window.__addrData) {
      const asnap = nearestGraphNode(a.lat, a.lon);
      let bestStop = null;
      let bestDist = null;

      if (asnap.idx != null && asnap.snapM != null) {
        const core = sf.dist[asnap.idx];
        if (isFinite(core)) {
          bestDist = asnap.snapM + core;
          const si = sf.sourceStopIdx[asnap.idx];
          if (si != null && si >= 0 && si < window.__stopData.length) {
            bestStop = window.__stopData[si];
          }
        }
      }

      a.distM = bestDist;
      a.nearestStopName = (bestStop && bestStop.name) ? bestStop.name : 'N/A';
      if (methodOverride !== undefined && methodOverride !== null) {
        a.methodology = methodOverride;
      } else {
        a.methodology = bestDist != null ? 'roads + tracks' : 'N/A';
      }

      // Update highlight route on the graph.
      if (window.__routeStore && a.id && bestStop && asnap.idx != null) {
        const coords = [[a.lat, a.lon]];
        let cur = asnap.idx;
        const seen = new Set();
        while (cur != null && cur >= 0 && !seen.has(cur)) {
          seen.add(cur);
          const nd = rt.nodes[cur];
          if (Array.isArray(nd) && nd.length >= 2) coords.push([nd[0], nd[1]]);
          const nx = sf.parent[cur];
          if (nx == null || nx < 0) break;
          cur = nx;
        }
        coords.push([bestStop.lat, bestStop.lon]);
        window.__routeStore[a.id] = coords;
      }
    }

    recalcLineDistancesFromStops(window.__roadData);
    recalcLineDistancesFromStops(window.__trackData);

    recolorAll(window.__gradMaxM);
    updateAddressPopups();
    const summaryReason = (reason === undefined) ? 'bus stop moved' : reason;
    const summaryKey = (logKey === undefined) ? 'stop:unknown' : logKey;
    updateSummary(summaryReason, summaryKey);
  }

  function populateRouteFilterOptions() {
    const sel = document.getElementById('__routeFilter');
    if (!sel) return;
    sel.innerHTML = '';
    const opts = Array.isArray(window.__routeOptions) ? window.__routeOptions : [];
    for (const r of opts) {
      const opt = document.createElement('option');
      opt.value = String(r.value);
      opt.textContent = `${r.label}`;
      sel.appendChild(opt);
    }
  }

  function setStartingStatsToCurrent() {
    const nextStats = computeServedCounts(window.__gradMaxM);
    window.__startingSvcStats = nextStats;
    if (window.__addrData) {
      for (const a of window.__addrData) {
        a.baseServed = (a.distM != null && isFinite(a.distM) && a.distM <= window.__gradMaxM);
      }
    }
    recolorAll(window.__gradMaxM);
    renderStartingStats(nextStats);
  }

  function installRouteFilter() {
    const sel = document.getElementById('__routeFilter');
    if (!sel) return;
    populateRouteFilterOptions();
    sel.addEventListener('change', function() {
      const picked = Array.from(sel.selectedOptions).map(o => String(o.value));
      const idSet = new Set();
      const valueToIds = window.__routeValueToIds || {};
      for (const v of picked) {
        const ids = Array.isArray(valueToIds[v]) ? valueToIds[v] : [];
        for (const rid of ids) idSet.add(String(rid));
      }
      window.__activeRouteFilter = idSet;
      recalcAddressesFromStops(null, null, null);
      setStartingStatsToCurrent();
    });
  }

  function installTopRightRouteFilterControl(map) {
    if (!map || !L || !L.control) return;

    const RouteFilterControl = L.Control.extend({
      options: { position: 'topright' },
      onAdd: function() {
        const wrap = L.DomUtil.create('div', 'leaflet-bar');
        wrap.style.background = 'rgba(255,255,255,0.95)';
        wrap.style.padding = '8px';
        wrap.style.width = '250px';
        wrap.style.boxShadow = '0 2px 10px rgba(0,0,0,0.12)';
        wrap.style.borderRadius = '6px';
        wrap.innerHTML = `
          <div style="font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size:12px; font-weight:600; margin-bottom:4px;">
            Routes to include
          </div>
          <select id="__routeFilter" multiple size="8" style="width:100%; font-size:12px;"></select>
        `;

        L.DomEvent.disableClickPropagation(wrap);
        L.DomEvent.disableScrollPropagation(wrap);
        return wrap;
      }
    });

    map.addControl(new RouteFilterControl());
  }

  function installSlider(map) {
    const wrap = document.createElement('div');
    wrap.style.position = 'fixed';
    wrap.style.bottom = '18px';
    wrap.style.left = '18px';
    wrap.style.zIndex = '9999';
    wrap.style.background = 'rgba(255,255,255,0.92)';
    wrap.style.padding = '10px 12px';
    wrap.style.borderRadius = '10px';
    wrap.style.boxShadow = '0 2px 10px rgba(0,0,0,0.12)';
    wrap.style.fontFamily = 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
    wrap.style.fontSize = '12px';
    wrap.style.lineHeight = '1.25';

    wrap.innerHTML = `
      <div style="font-weight:700; margin-bottom:6px;">Nearest stop distance</div>
      <div style="width:220px; height:12px; border-radius:8px; background: linear-gradient(90deg, #00ff00 0%, #ff0000 100%);"></div>
      <div style="display:flex; justify-content: space-between; margin-top: 4px; color:#333;">
        <span>0 m</span>
        <span id="__gradMaxLabel">400 m</span>
      </div>
      <div style="margin-top:8px;">
        <input id="__gradMaxSlider" type="range" min="400" max="800" step="10" value="400" style="width:220px;" />
      </div>
      <div style="margin-top:8px; display:flex; align-items:center; gap:8px;">
        <input id="__changesOnlyToggle" type="checkbox" />
        <label for="__changesOnlyToggle" style="cursor:pointer;">Show only gained/lost changes</label>
      </div>
      <div style="margin-top:6px; color:#555; opacity:0.85;">
        Drag bus stops to update nearest-stop distance for addresses.
      </div>
    `;

    document.body.appendChild(wrap);

    const slider = document.getElementById('__gradMaxSlider');
    const changesOnlyToggle = document.getElementById('__changesOnlyToggle');

    if (changesOnlyToggle) {
      changesOnlyToggle.checked = !!window.__showChangesOnly;
      changesOnlyToggle.addEventListener('change', function() {
        window.__showChangesOnly = !!changesOnlyToggle.checked;
        recolorAll(window.__gradMaxM);
      });
    }

    slider.addEventListener('input', function() {
      const v = parseFloat(slider.value);
      window.__gradMaxM = v;
      updateLegendMax(v);
      recolorAll(v);
      // no log spam while dragging
      renderStats(computeServedCounts(window.__gradMaxM));
    });

    // Only log once the user finishes interaction (mouse up / touch end)
    slider.addEventListener('change', function() {
      const v = parseFloat(slider.value);
      window.__gradMaxM = v;
      updateLegendMax(v);
      recolorAll(v);
      updateSummary(`range set to ${Math.round(v)} m`, 'range');
    });

    updateLegendMax(window.__gradMaxM);
  }

  function flattenLatLngs(latlngs, out) {
    if (!Array.isArray(latlngs)) return;
    for (const item of latlngs) {
      if (!item) continue;
      if (Array.isArray(item)) {
        flattenLatLngs(item, out);
      } else if (typeof item.lat === 'number' && typeof item.lng === 'number') {
        out.push(item);
      }
    }
  }

  function buildRouteSnapIndex() {
    const refs = Array.isArray(window.__routeShapeRefs) ? window.__routeShapeRefs : [];
    const refMeta = Array.isArray(window.__routeShapeMeta) ? window.__routeShapeMeta : [];
    const refToRouteId = new Map();
    for (const meta of refMeta) {
      if (!meta || !meta.refName) continue;
      refToRouteId.set(String(meta.refName), String(meta.routeId || ''));
    }

    const segments = [];
    const byRoute = new Map();
    for (const refName of refs) {
      const poly = getByName(refName);
      if (!poly || !poly.getLatLngs) continue;

      const routeId = refToRouteId.get(String(refName)) || '';
      const pts = [];
      flattenLatLngs(poly.getLatLngs(), pts);
      if (pts.length < 2) continue;

      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1];
        const b = pts[i];
        if (!a || !b) continue;
        const seg = {
          routeId: routeId,
          aLat: a.lat,
          aLon: a.lng,
          bLat: b.lat,
          bLon: b.lng,
          midLat: 0.5 * (a.lat + b.lat),
        };
        segments.push(seg);
        if (!byRoute.has(routeId)) byRoute.set(routeId, []);
        byRoute.get(routeId).push(seg);
      }
    }

    window.__routeSnapSegments = segments;
    window.__routeSnapSegmentsByRoute = byRoute;
  }

  function nearestPointOnRouteLine(lat, lon, routeIds) {
    const allSegs = Array.isArray(window.__routeSnapSegments) ? window.__routeSnapSegments : [];
    const byRoute = window.__routeSnapSegmentsByRoute;

    let candidateSegs = allSegs;
    if (Array.isArray(routeIds) && routeIds.length && byRoute && byRoute.get) {
      candidateSegs = [];
      for (const rid of routeIds) {
        const list = byRoute.get(String(rid));
        if (Array.isArray(list) && list.length) candidateSegs.push(...list);
      }
      if (!candidateSegs.length) candidateSegs = allSegs;
    }

    if (!candidateSegs.length) return null;

    let best = null;
    let bestD2 = Infinity;

    for (const seg of candidateSegs) {
      const cosLat = Math.cos((seg.midLat || lat) * Math.PI / 180.0);
      const ax = seg.aLon * cosLat;
      const ay = seg.aLat;
      const bx = seg.bLon * cosLat;
      const by = seg.bLat;
      const px = lon * cosLat;
      const py = lat;

      const vx = bx - ax;
      const vy = by - ay;
      const wx = px - ax;
      const wy = py - ay;
      const vv = vx * vx + vy * vy;
      let t = 0;
      if (vv > 0) t = clamp((wx * vx + wy * vy) / vv, 0.0, 1.0);

      const projX = ax + t * vx;
      const projY = ay + t * vy;
      const dx = px - projX;
      const dy = py - projY;
      const d2 = dx * dx + dy * dy;

      if (d2 < bestD2) {
        bestD2 = d2;
        best = {
          lat: projY,
          lon: (Math.abs(cosLat) > 1e-9) ? (projX / cosLat) : lon,
          routeId: seg.routeId || '',
        };
      }
    }

    return best;
  }

  function snapStopToRouteLine(stop) {
    if (!stop) return null;
    const snapped = nearestPointOnRouteLine(stop.lat, stop.lon, stop.routeIds || []);
    if (!snapped) return null;
    const movedM = haversineM(stop.lat, stop.lon, snapped.lat, snapped.lon);
    stop.lat = snapped.lat;
    stop.lon = snapped.lon;
    if (!Array.isArray(stop.routeIds)) stop.routeIds = [];
    if (snapped.routeId && !stop.routeIds.includes(snapped.routeId)) stop.routeIds.push(snapped.routeId);
    return movedM;
  }

  function installDraggableStops(map) {
    if (!window.__stopData || !Array.isArray(window.__stopData)) return;

    if (!window.__stopMarkers) window.__stopMarkers = {}; // stop_id -> leaflet marker

    // lightweight dot icon
    const dot = L.divIcon({
      className: 'stop-dot',
      html: '<div style="width:10px;height:10px;border-radius:10px;background:#111;opacity:0.75;border:2px solid #fff; box-shadow:0 1px 4px rgba(0,0,0,0.25);"></div>',
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    });

    function ensureStopId(s) {
      // Use GTFS stop_id if present; otherwise generate a stable-ish id
      if (s.id && s.id !== '' && s.id !== 'nan') return String(s.id);
      if (!s.__genId) s.__genId = `custom_${Date.now()}_${Math.floor(Math.random()*1e9)}`;
      return s.__genId;
    }

    function addStopMarkerFor(s) {
      const sid = ensureStopId(s);
      // Avoid duplicating
      if (window.__stopMarkers[sid]) return window.__stopMarkers[sid];

      snapStopToRouteLine(s);
      const stopName = String(s.stopName || s.name || '').trim() || 'Unknown stop';
      const stopCode = String(s.stopCode || '').trim();
      const stopTitle = stopCode ? `${stopCode} — ${stopName}` : stopName;
      const details = Array.isArray(s.distanceDetails) ? s.distanceDetails : [];
      const detailItems = details.slice(0, 8).map(function(d) {
        const dir = String(d && d.direction || '');
        const other = String(d && d.other_stop_name || '').trim() || 'Unknown stop';
        const dist = Number(d && d.distance_m || 0);
        const prefix = dir === 'to_next' ? 'To next' : 'From previous';
        return `<li>${prefix} <b>${other}</b>: ${Math.round(dist).toLocaleString()} m</li>`;
      }).join('');
      const detailHtml = detailItems
        ? `<div style="margin-top:6px;"><b>Adjacent stop distances</b><ul style="margin:4px 0 0 16px; padding:0;">${detailItems}</ul></div>`
        : '<div style="margin-top:6px; opacity:0.8;">No adjacent stop distance data.</div>';

      const mk = L.marker([s.lat, s.lon], { pane: "stopsPane", draggable: true, icon: dot, title: stopTitle || sid }).addTo(map);
      mk.bindPopup(`<div style="font-size:12px;"><b>Stop:</b> ${stopTitle}${detailHtml}</div>`);

      mk.on('dragend', function(ev) {
        const ll = ev.target.getLatLng();
        s.lat = ll.lat;
        s.lon = ll.lng;
        snapStopToRouteLine(s);
        ev.target.setLatLng([s.lat, s.lon]);
        s.__moved = true;
        const label = (s.name || s.id || sid);
        recalcAddressesFromStops(`stop moved: ${label}`, `stop:${sid}`, 'roads + tracks');
      });

      // Right click to remove
      mk.on('contextmenu', function() {
        const label = (s.name || s.id || sid);
        if (!confirm(`Remove bus stop "${label}"?`)) return;

        // Remove from map
        map.removeLayer(mk);

        // Remove from stopData
        const idx = window.__stopData.indexOf(s);
        if (idx >= 0) window.__stopData.splice(idx, 1);

        // Remove marker registry
        delete window.__stopMarkers[sid];

        // Update distances + stats/log
        recalcAddressesFromStops(`stop removed: ${label}`, `stop:${sid}`, 'roads + tracks');
      });

      window.__stopMarkers[sid] = mk;
      return mk;
    }

    // Initial markers
    for (const s of window.__stopData) addStopMarkerFor(s);

    // Expose helper so route-click can add stops later
    window.__addStop = function(lat, lon, name, routeIdHint) {
      const routeIds = routeIdHint ? [String(routeIdHint)] : Array.from(window.__activeRouteFilter || []);
      const s = { id: '', lat: lat, lon: lon, name: name || 'New stop', routeIds: routeIds, __moved: true };
      snapStopToRouteLine(s);
      const sid = ensureStopId(s);
      window.__stopData.push(s);
      addStopMarkerFor(s);
      recalcAddressesFromStops(`stop added: ${s.name}`, `stop:${sid}`, 'roads + tracks');
    };
  }

  function installRouteClickToAddStop(map) {
    if (!window.__routeShapeRefs || !Array.isArray(window.__routeShapeRefs)) return;
    const meta = Array.isArray(window.__routeShapeMeta) ? window.__routeShapeMeta : [];
    const refToRouteId = new Map();
    for (const m of meta) {
      if (!m || !m.refName) continue;
      refToRouteId.set(String(m.refName), String(m.routeId || ''));
    }

    for (const refName of window.__routeShapeRefs) {
      const poly = getByName(refName);
      if (!poly || !poly.on) continue;
      const routeId = refToRouteId.get(String(refName)) || '';

      poly.on('click', function(ev) {
        try {
          const ll = ev.latlng;
          const proposed = '';
          const name = prompt('Name for new bus stop:', proposed);
          if (name == null) return; // cancelled
          const trimmed = String(name).trim();
          if (!trimmed) return;

          if (window.__addStop) {
            window.__addStop(ll.lat, ll.lng, trimmed, routeId);
          }
        } catch (e) {
          console.error(e);
        }
      });
    }
  }

  document.addEventListener('DOMContentLoaded', function() {
    try {
      const mapName = window.__foliumMapName;
      const map = getByName(mapName);
      if (!map) return;

      installSlider(map);
      ensureStatsPanel();
      installTopRightRouteFilterControl(map);
      installRouteFilter();
      // initial graph-based nearest-stop calculations + recolour/stats
      recalcAddressesFromStops(null, null, null);
      setStartingStatsToCurrent();
      // add draggable stop overlays
      buildRouteSnapIndex();
      installDraggableStops(map);
      installRouteClickToAddStop(map);
    } catch (e) {
      console.error(e);
    }
  });
})();
</script>
"""
    m.get_root().html.add_child(Element(js))


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--addresses", default="lds-nz-addresses-CSV/nz-addresses.csv", help="Path to lds-nz-addresses-CSV/nz-addresses.csv")
    ap.add_argument(
        "--roads",
        default="lds-nz-roads-road-section-geometry-CSV/nz-roads-road-section-geometry.csv",
        help=(
            "Path to lds-nz-roads-road-section-geometry-CSV CSV (expects WKT, road_section_id, road_type, geometry_class, end_lifespan). "
            "Rows with end_lifespan are excluded (no longer exist)."
        ),
    )
    ap.add_argument("--gtfs", default="gtfs", help="Path to GTFS folder (contains routes.txt, stops.txt, shapes.txt, trips.txt, ...)")
    ap.add_argument("--out", default="map.html", help="Output HTML filename")

    ap.add_argument("--town", default=None, help="Filter addresses by town_city (e.g. 'Christchurch')")
    ap.add_argument("--ta", default=None, help="Filter addresses by territorial_authority (e.g. 'Christchurch City')")
    ap.add_argument(
        "--bbox",
        nargs="+",
        type=float,
        default=[172.536417, -43.561454, 172.555088],
        help="min_lon min_lat max_lon [max_lat]",
    )
    ap.add_argument("--ward", default=None, help="Filter by ward name from Ward_(OpenData).geojson (e.g. 'Hornby')")
    ap.add_argument("--ward-geojson", default="Ward_(OpenData).geojson", help="Path to Ward_(OpenData).geojson")
    ap.add_argument(
        "--routes",
        default=None,
        help="Optional route filter by route_id/route_short_name (comma-separated, e.g. 100,101).",
    )

    ap.add_argument("--max-addresses", type=int, default=20000, help="Max addresses to load/plot after filtering")
    ap.add_argument("--max-roads", type=int, default=200000, help="Max road rows to consider (after bbox filter)")
    ap.add_argument("--max-stops", type=int, default=20000, help="Max stops to plot")

    ap.add_argument("--densify-m", type=float, default=10.0, help="Densify road segments to this spacing (m). Lower = better snapping, heavier graph")
    ap.add_argument("--display-road-limit", type=int, default=20000, help="How many road features to draw (visual only). Graph building uses --max-roads")
    ap.add_argument("--draw-all-route-shapes", action="store_true", help="Draw all shapes per route (can be heavy). Default limits to 5 per route")

    ap.add_argument("--color-max-m", type=float, default=400.0, help="Initial colour scale clamp (slider allows 400..800)")
    ap.add_argument("--road-draw-weight", type=float, default=3.0, help="Road line thickness")
    ap.add_argument("--road-stop-sample-m", type=float, default=25.0, help="Sampling spacing (m) when estimating road distance to nearest stop")
    ap.add_argument("--tracks-geojson", default="Track_(OpenData).geojson", help="Optional GeoJSON file of walkable tracks to include in routing graph")

    args = ap.parse_args()
    bbox = parse_bbox(args.bbox)
    ward_geom = None
    ward_bbox = None
    if args.ward:
        ward_geom = load_ward_geometry(args.ward_geojson, args.ward)
        minx, miny, maxx, maxy = ward_geom.bounds
        ward_bbox = BBox(minx, miny, maxx, maxy)

    effective_bbox = bbox if bbox is not None else ward_bbox

    # --- load GTFS ---
    routes, stops, shapes, trips = load_gtfs(args.gtfs)

    selected_route_ids: Optional[Set[str]] = None
    if args.routes:
        requested = {x.strip() for x in str(args.routes).split(",") if x.strip()}
        if requested:
            rid_col = routes["route_id"].astype(str) if "route_id" in routes.columns else pd.Series(dtype=str)
            rshort_col = routes["route_short_name"].astype(str) if "route_short_name" in routes.columns else pd.Series(dtype=str)
            mask = rid_col.isin(requested) | rshort_col.isin(requested)
            selected_route_ids = set(routes.loc[mask, "route_id"].astype(str).tolist())
            if not selected_route_ids:
                raise SystemExit(f"No GTFS routes matched --routes={args.routes!r}.")
            routes = routes[routes["route_id"].astype(str).isin(selected_route_ids)].copy()
            trips = trips[trips["route_id"].astype(str).isin(selected_route_ids)].copy()

    shapes_by_route = build_shapes_by_route(routes, shapes, trips)

    # stop_id -> set(route_id), via stop_times + trips
    stop_route_ids: Dict[str, Set[str]] = {}
    stop_times: Optional[pd.DataFrame] = None
    stop_times_path = os.path.join(args.gtfs, "stop_times.txt")
    if os.path.exists(stop_times_path) and {"trip_id", "route_id"}.issubset(trips.columns):
        stop_times = pd.read_csv(stop_times_path, usecols=["trip_id", "stop_id", "stop_sequence"], dtype=str)
        trip_routes = trips[["trip_id", "route_id"]].dropna().astype(str)
        stop_trip_routes = stop_times[["trip_id", "stop_id"]].merge(trip_routes, on="trip_id", how="inner")
        for sid, grp in stop_trip_routes.groupby("stop_id"):
            stop_route_ids[str(sid)] = {
                str(x) for x in grp["route_id"].astype(str).tolist() if str(x).strip() and str(x).lower() != "nan"
            }

    # --- load + filter stops ---
    stops = stops.copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])

    if selected_route_ids is not None:
        stop_ids_for_selected = {sid for sid, rids in stop_route_ids.items() if rids & selected_route_ids}
        if stop_ids_for_selected:
            stops = stops[stops["stop_id"].astype(str).isin(stop_ids_for_selected)].copy()
        else:
            stops = stops.iloc[0:0].copy()

    if effective_bbox:
        stops = stops[
            (stops["stop_lon"] >= effective_bbox.min_lon)
            & (stops["stop_lon"] <= effective_bbox.max_lon)
            & (stops["stop_lat"] >= effective_bbox.min_lat)
            & (stops["stop_lat"] <= effective_bbox.max_lat)
        ]
    if ward_geom is not None:
        stops = stops[
            stops.apply(lambda r: ward_geom.contains(Point(float(r["stop_lon"]), float(r["stop_lat"]))), axis=1)
        ]

    if len(stops) > args.max_stops:
        stops = stops.iloc[: args.max_stops].copy()

    stop_distance_details = build_stop_distance_details(
        trips=trips,
        stops=stops,
        stop_times=stop_times,
    )

    # --- load + filter addresses ---
    usecols = {
        "WKT",
        "address_id",
        "full_address",
        "territorial_authority",
        "town_city",
        "shape_X",
        "shape_Y",
    }
    addr = pd.read_csv(args.addresses, usecols=lambda c: c in usecols)

    if args.town and "town_city" in addr.columns:
        addr = addr[addr["town_city"].astype(str) == args.town]

    if args.ta and "territorial_authority" in addr.columns:
        addr = addr[addr["territorial_authority"].astype(str) == args.ta]

    if "shape_X" in addr.columns and "shape_Y" in addr.columns:
        addr["lon"] = pd.to_numeric(addr["shape_X"], errors="coerce")
        addr["lat"] = pd.to_numeric(addr["shape_Y"], errors="coerce")
    else:

        def parse_point(w: str) -> Tuple[float, float]:
            p = wkt.loads(w)
            return float(p.x), float(p.y)  # lon, lat

        pts = addr["WKT"].astype(str).apply(parse_point)
        addr["lon"] = pts.apply(lambda t: t[0])
        addr["lat"] = pts.apply(lambda t: t[1])

    addr = addr.dropna(subset=["lon", "lat"])

    if effective_bbox:
        addr = addr[
            (addr["lon"] >= effective_bbox.min_lon)
            & (addr["lon"] <= effective_bbox.max_lon)
            & (addr["lat"] >= effective_bbox.min_lat)
            & (addr["lat"] <= effective_bbox.max_lat)
        ]
    if ward_geom is not None:
        addr = addr[addr.apply(lambda r: ward_geom.contains(Point(float(r["lon"]), float(r["lat"]))), axis=1)]

    if len(addr) > args.max_addresses:
        addr = addr.iloc[: args.max_addresses].copy()

    if len(addr) == 0:
        raise SystemExit("No addresses left after filtering. Adjust --town/--ta/--bbox/--ward.")

    # --- load roads + build graph ---

    # --- load roads + build graph ---
    # Roads: now using lds-nz-roads-road-section-geometry-CSV
    # Filter out rows with end_lifespan (these no longer exist)
    road_usecols = [
        "WKT",
        "road_section_geometry_id",
        "geometry_class",
        "road_type",
        "road_section_id",
        "end_lifespan",
        "left_suburb_locality",
        "right_suburb_locality",
        "left_town_city",
        "right_town_city",
        "left_territorial_authority",
        "right_territorial_authority",
    ]
    roads = pd.read_csv(
        args.roads,
        usecols=lambda c: c in set(road_usecols),
        dtype=str,
        keep_default_na=True,
    )

    # Normalise end_lifespan: treat empty/whitespace as NA
    if "end_lifespan" in roads.columns:
        roads["end_lifespan"] = (
            roads["end_lifespan"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan})
        )
        roads = roads[roads["end_lifespan"].isna()].copy()

    # Ensure WKT present
    roads = roads.dropna(subset=["WKT"]).copy()

    g, nodes_arr = build_road_graph(
        roads,
        bbox=effective_bbox,
        ward_geom=ward_geom,
        max_roads=args.max_roads,
        densify_m=args.densify_m,
    )

    # For snap-distance to metres + densify sampling
    tf_to_m = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=False)

    track_lines = load_track_lines_geojson(args.tracks_geojson, bbox=bbox)
    if track_lines:
        track_nodes = add_lines_to_graph(
            g,
            lines=track_lines,
            densify_m=float(args.densify_m),
            tf_to_m=tf_to_m,
            kind="track",
        )
        connect_track_nodes_to_graph(
            g,
            track_nodes=track_nodes,
            base_nodes=nodes_arr,
            tf_to_m=tf_to_m,
            max_link_m=20.0,
        )

    nodes_arr = np.array([[n[0], n[1]] for n in g.nodes()], dtype=np.float64)
    node_kd = cKDTree(nodes_arr) if len(nodes_arr) else None

    # --- choose map centre ---
    centre_lat = float(addr["lat"].mean())
    centre_lon = float(addr["lon"].mean())

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=14,
        control_scale=True,
        tiles="CartoDB positron",
    )

    # Explicit layer stack (bottom -> top): roads, routes, stops.
    folium.map.CustomPane("roadsPane", z_index=410).add_to(m)
    folium.map.CustomPane("routesPane", z_index=420).add_to(m)
    folium.map.CustomPane("stopsPane", z_index=430).add_to(m)

    # Expose routable graph to JS so all interactive distance calculations
    # are done consistently client-side via roads + tracks.
    node_to_idx = {n: i for i, n in enumerate(g.nodes())}
    graph_nodes_js = [[float(lat), float(lon)] for (lon, lat) in g.nodes()]
    graph_edges_js = [
        [int(node_to_idx[u]), int(node_to_idx[v]), float(data.get("weight", 0.0))]
        for u, v, data in g.edges(data=True)
    ]
    m.get_root().html.add_child(
        Element(
            f"<script>window.__graphData = {json.dumps({'nodes': graph_nodes_js, 'edges': graph_edges_js})};</script>"
        )
    )

    # Store folium map variable name for JS
    m.get_root().html.add_child(Element(f"<script>window.__foliumMapName = {json.dumps(m.get_name())};</script>"))

    # JS for route highlighting
    m.get_root().html.add_child(
        Element(
            """
<script>
  window.__activeRouteLine = null;
  window.__routeStore = {}; // address_id -> array of [lat, lon]
  window.__showRoute = function(map, addressId) {
    try {
      if (window.__activeRouteLine) {
        map.removeLayer(window.__activeRouteLine);
        window.__activeRouteLine = null;
      }
      const coords = window.__routeStore[addressId];
      if (!coords || coords.length < 2) return;

      window.__activeRouteLine = L.polyline(coords, {
        pane: 'routesPane',
        weight: 6,
        opacity: 0.9
      }).addTo(map);

      map.fitBounds(window.__activeRouteLine.getBounds(), { padding: [30, 30] });
    } catch (e) {
      console.error(e);
    }
  };
</script>
"""
        )
    )

    # --- layer: route shapes ---
    fg_routes = folium.FeatureGroup(name="Bus routes", show=True)
    route_shape_refs: List[str] = []
    route_shape_meta: List[dict] = []
    routes_idx = routes.set_index("route_id", drop=False) if "route_id" in routes.columns else None

    available_route_ids_in_bounds: Set[str] = set()
    for sid in stops.get("stop_id", pd.Series(dtype=str)).astype(str).tolist():
        available_route_ids_in_bounds.update(stop_route_ids.get(str(sid), set()))
    route_options_js: List[dict] = []
    route_value_to_ids_js: Dict[str, List[str]] = {}
    if routes_idx is not None:
        grouped_route_ids_by_label: Dict[str, Set[str]] = {}
        for rid in available_route_ids_in_bounds:
            if rid not in routes_idx.index:
                continue
            rr = routes_idx.loc[rid]
            short_name = str(rr.get("route_short_name", "")).strip()
            long_name = str(rr.get("route_long_name", "")).strip()
            if short_name and long_name:
                label = f"{short_name} {long_name}"
            else:
                label = short_name or long_name or f"route {rid}"
            grouped_route_ids_by_label.setdefault(label, set()).add(str(rid))

        for label in sorted(grouped_route_ids_by_label.keys(), key=natural_route_sort_key):
            route_ids_for_label = sorted(grouped_route_ids_by_label[label])
            route_value = route_ids_for_label[0]
            route_options_js.append({"value": route_value, "label": label})
            route_value_to_ids_js[route_value] = route_ids_for_label

    for route_id, polylines in shapes_by_route.items():
        if routes_idx is not None and route_id in routes_idx.index:
            r = routes_idx.loc[route_id]
            color = str(r.get("route_color", "")).strip()
            if len(color) == 6 and all(c in "0123456789ABCDEFabcdef" for c in color):
                color = "#" + color
            else:
                color = "#0066cc"  # fallback

            label = str(r.get("route_short_name", "")).strip()
            if not label:
                label = str(r.get("route_long_name", "")).strip()
            if not label:
                label = f"route {route_id}"
        else:
            color = "#0066cc"
            label = f"route {route_id}"

        max_shapes = None if args.draw_all_route_shapes else 5
        for line in (polylines if max_shapes is None else polylines[:max_shapes]):
            poly_route = folium.PolyLine(
                locations=[(lat, lon) for (lat, lon) in line],
                pane="routesPane",
                weight=3,
                opacity=0.8,
                tooltip=label,
                color=color,
            )
            poly_route.add_to(fg_routes)
            ref_name = poly_route.get_name()
            route_shape_refs.append(ref_name)
            route_shape_meta.append({"refName": ref_name, "routeId": str(route_id)})

    fg_routes.add_to(m)

    # Expose route polylines to JS so we can click to add a new stop
    m.get_root().html.add_child(Element(f"<script>window.__routeShapeRefs = {json.dumps(route_shape_refs)};</script>"))
    m.get_root().html.add_child(Element(f"<script>window.__routeShapeMeta = {json.dumps(route_shape_meta)};</script>"))

    # --- stops data for draggable JS overlay (no static stop markers) ---
    stop_js_data: List[dict] = []

    for _, s in stops.iterrows():
        stop_name = str(s.get("stop_name", "")).strip()
        stop_code = str(s.get("stop_code", "")).strip()
        title = f"{stop_code} — {stop_name}" if stop_code and stop_code != "nan" else stop_name

        lat_s = float(s["stop_lat"])
        lon_s = float(s["stop_lon"])

        sid = str(s.get("stop_id", ""))
        stop_js_data.append({
            "id": sid,
            "lat": lat_s,
            "lon": lon_s,
            "stopName": stop_name,
            "stopCode": stop_code,
            "name": title,
            "routeIds": sorted(list(stop_route_ids.get(sid, set()))),
            "distanceDetails": stop_distance_details.get(sid, []),
        })

    # Expose stops to JS for draggable overlay
    m.get_root().html.add_child(Element(f"<script>window.__stopData = {json.dumps(stop_js_data)};</script>"))
    m.get_root().html.add_child(Element(f"<script>window.__routeOptions = {json.dumps(route_options_js)};</script>"))
    m.get_root().html.add_child(Element(f"<script>window.__routeValueToIds = {json.dumps(route_value_to_ids_js)};</script>"))

    # --- layer: roads (visual + coloured by nearest stop distance) ---
    fg_roads = folium.FeatureGroup(name="Roads (distance to stop)", show=True)
    display_roads = roads.head(min(len(roads), int(args.display_road_limit)))

    road_js: List[dict] = []

    for _, row in display_roads.iterrows():
        geom = row.get("WKT")
        if not isinstance(geom, str) or not geom:
            continue

        rsid = str(row.get("road_section_id", "")).strip()
        rtype = str(row.get("road_type", "")).strip()
        gcls = str(row.get("geometry_class", "")).strip()
        rid = str(row.get("road_section_geometry_id", "")).strip()
        parts = [p for p in [rtype, gcls] if p and p.lower() != "nan"]
        base = " / ".join(parts) if parts else "road"
        if rsid and rsid.lower() != "nan":
            road_name = f"{base} (section {rsid})"
        elif rid and rid.lower() != "nan":
            road_name = f"{base} (geom {rid})"
        else:
            road_name = base

        try:
            sh = wkt.loads(geom)
        except Exception:
            continue

        if sh.geom_type == "LineString":
            lines = [sh]
        elif sh.geom_type == "MultiLineString":
            lines = list(sh.geoms)
        else:
            continue

        for ls in lines:
            coords = list(ls.coords)
            if len(coords) < 2:
                continue
            if effective_bbox and not any(effective_bbox.contains_lonlat(x, y) for (x, y) in coords):
                continue
            if ward_geom is not None and not ward_geom.intersects(ls):
                continue

            d_m = None

            color = dist_to_green_red_hex(d_m, max_m=float(args.color_max_m))
            d_label = "N/A" if d_m is None else f"{d_m:,.0f} m"
            tip = road_name if road_name else "(road)"
            tip = f"{tip} — nearest stop: {d_label}"

            poly = folium.PolyLine(
                locations=[(y, x) for (x, y) in coords],
                pane="roadsPane",
                weight=float(args.road_draw_weight),
                opacity=0.75,
                color=color,
                tooltip=tip,
            )
            poly.add_to(fg_roads)

            # store poly reference + distance so the slider can recolour
            road_js.append({
                "polyRefName": poly.get_name(),
                "dM": float(d_m) if d_m is not None else None,
                "samples": linestring_sample_points_latlon(
                    ls,
                    tf_to_m=tf_to_m,
                    sample_every_m=max(30.0, float(args.road_stop_sample_m)),
                ),
            })

    fg_roads.add_to(m)
    m.get_root().html.add_child(Element(f"<script>window.__roadData = {json.dumps(road_js)};</script>"))

    # --- layer: tracks (visual + coloured by nearest stop distance) ---
    fg_tracks = folium.FeatureGroup(name="Tracks (distance to stop)", show=True)
    track_js: List[dict] = []

    for ls, track_name in track_lines:
        coords = list(ls.coords)
        if len(coords) < 2:
            continue
        if effective_bbox and not any(effective_bbox.contains_lonlat(x, y) for (x, y) in coords):
            continue
        if ward_geom is not None and not ward_geom.intersects(ls):
            continue

        d_m = None

        color = dist_to_green_red_hex(d_m, max_m=float(args.color_max_m))
        d_label = "N/A" if d_m is None else f"{d_m:,.0f} m"
        tip = f"{track_name} — nearest stop: {d_label}"

        poly = folium.PolyLine(
            locations=[(y, x) for (x, y) in coords],
            pane="roadsPane",
            weight=max(2.0, float(args.road_draw_weight) - 0.5),
            opacity=0.8,
            color=color,
            tooltip=tip,
        )
        poly.add_to(fg_tracks)

        track_js.append({
            "polyRefName": poly.get_name(),
            "dM": float(d_m) if d_m is not None else None,
            "samples": linestring_sample_points_latlon(
                ls,
                tf_to_m=tf_to_m,
                sample_every_m=max(30.0, float(args.road_stop_sample_m)),
            ),
        })

    fg_tracks.add_to(m)
    m.get_root().html.add_child(Element(f"<script>window.__trackData = {json.dumps(track_js)};</script>"))

    # --- layer: addresses with nearest-stop distance ---
    fg_addr = folium.FeatureGroup(name="Addresses", show=True)

    bind_js: List[str] = []
    addr_js: List[dict] = []

    for _, a in addr.iterrows():
        lon = float(a["lon"])
        lat = float(a["lat"])
        full_addr = str(a.get("full_address", "")).strip()
        addr_id = str(a.get("address_id", "")).strip()

        nearest_stop_txt = "(calculating…)"
        dist_txt = "(calculating…)"
        method_txt = "roads + tracks"
        roadseg_txt = "(interactive)"

        route_coords: Optional[List[Tuple[float, float]]] = None
        dist_for_colour_m: Optional[float] = None

        # Register route in JS store (only for this address)
        if route_coords is not None and addr_id:
            route_json = json.dumps([[float(lat), float(lon)] for (lat, lon) in route_coords])
            m.get_root().html.add_child(
                Element(f"<script>window.__routeStore[{json.dumps(addr_id)}] = {route_json};</script>")
            )

        marker_color = "#9e9e9e"

        # base popup with placeholders for client-side update
        popup_html_base = f"""
        <div style=\"font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size: 13px;\">
          <div style=\"font-weight: 700; margin-bottom: 6px;\">{full_addr or "(address)"}</div>
          <div><b>Address ID:</b> {addr_id}</div>
          <div><b>Nearest stop:</b> __NEAREST__</div>
          <div><b>Approx distance:</b> __DIST__</div>
          <div><b>Method:</b> __METHOD__</div>
          <div><b>Road segments:</b> {roadseg_txt}</div>
          <div style=\"opacity: 0.7; margin-top: 6px;\">Click marker to highlight the computed path.</div>
        </div>
        """.strip()

        # initial (server-side) content
        popup_html_initial = (
            popup_html_base
            .replace("__NEAREST__", nearest_stop_txt)
            .replace("__DIST__", dist_txt)
            .replace("__METHOD__", method_txt)
        )

        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            weight=2,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.95,
            tooltip=full_addr if full_addr else None,
            popup=folium.Popup(popup_html_initial, max_width=360),
        )
        marker.add_to(fg_addr)

        if addr_id:
            bind_js.append(
                f"{marker.get_name()}.on('click', function() {{ window.__showRoute({m.get_name()}, {addr_id!r}); }});"
            )

        # expose marker + base popup + current distance for JS recolouring / recalculation
        addr_js.append(
            {
                "id": addr_id,
                "lat": float(lat),
                "lon": float(lon),
                "markerRefName": marker.get_name(),
                "basePopupHtml": popup_html_base,
                "nearestStopName": nearest_stop_txt,
                "distM": float(dist_for_colour_m) if dist_for_colour_m is not None else None,
                "baseServed": False,
                "methodology": method_txt,
            }
        )

    fg_addr.add_to(m)
    m.get_root().html.add_child(Element(f"<script>window.__addrData = {json.dumps(addr_js)};</script>"))

    # Emit the click bindings in one place (ensures marker JS variables exist)
    if bind_js:
        bindings_code = "".join(bind_js)
        script_block = f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
    {bindings_code}
    }});
    </script>
    """
        m.get_root().html.add_child(Element(script_block))

    # Inject slider + draggable stop overlays + live recolour/recalc
    m.get_root().html.add_child(Element(f"<script>window.__gradMaxM = {float(args.color_max_m)};</script>"))
    add_ui_and_interaction_js(m)


    m.save(args.out)

    print(f"Wrote: {args.out}")
    print(
        f"Addresses: {len(addr):,} | Stops: {len(stops):,} | Roads graph nodes: {len(g.nodes):,} edges: {len(g.edges):,} | Tracks loaded: {len(track_lines):,}"
    )
    if bbox:
        print(f"BBox: {bbox}")
    if args.ward:
        print(f"Ward: {args.ward}")


if __name__ == "__main__":
    main()
