# bus

Generate standalone Folium/Leaflet HTML maps for NZ addresses, roads/tracks, and GTFS routes/stops.

## What this repo does

`main.py` builds an interactive HTML map that combines:

- Address points (with nearest-stop distance info).
- Road and track geometry used for browser-side route distance estimation.
- GTFS route shapes and stops.
- Interactive UI controls (distance colour scale, route filtering, stop dragging, before/after service stats).

Because everything is embedded directly into a single HTML file, output size is mainly driven by how much data you include.

## Requirements

Install Python dependencies first:

```bash
python -m pip install -r requirements.txt
```

## Basic usage

```bash
python main.py \
  --addresses lds-nz-addresses-CSV/nz-addresses.csv \
  --roads lds-nz-roads-road-section-geometry-CSV/nz-roads-road-section-geometry.csv \
  --gtfs gtfs \
  --out index.html
```

Open `index.html` in a browser after generation.

## Reducing output HTML file size

If you need smaller generated maps, these are the highest-impact knobs (already supported by `main.py`).

### 1) Restrict geographic scope first (largest impact)

- `--ward "..."` limits data to one ward.
- `--bbox min_lon min_lat max_lon [max_lat]` limits to a bounding box.
- `--town` and `--ta` filter address records.
- `--all-wards-out-dir` generates one smaller HTML per ward rather than one huge all-in-one file.

### 2) Reduce feature counts

- `--max-addresses` lowers number of address markers and address payload entries.
- `--max-roads` limits road rows used in graph construction.
- `--display-road-limit` limits road features drawn (visual layer size).
- `--max-stops` caps number of stop markers and stop payload entries.

### 3) Reduce per-feature geometry/payload complexity

- `--draw-simplify-m` simplifies rendered road/track geometry.
- `--graph-precision` reduces embedded numeric precision for graph/data JSON.
- `--max-road-samples` reduces sampled points retained per line for nearest-stop recalculation.
- Keep default route-shape behavior; avoid `--draw-all-route-shapes` unless you specifically need every shape variant.

### Practical size-vs-quality tuning order

For best results with minimal quality loss, tune in this order:

1. Constrain area (`--ward` or `--bbox`).
2. Lower high-volume counts (`--max-addresses`, `--display-road-limit`, `--max-stops`).
3. Increase simplification (`--draw-simplify-m`).
4. Lower precision (`--graph-precision`).
5. Lower line samples (`--max-road-samples`).

## Preset examples

### A) Fast local preview (small output)

```bash
python main.py \
  --ward "Hornby" \
  --max-addresses 2500 \
  --max-roads 30000 \
  --display-road-limit 3000 \
  --max-stops 800 \
  --draw-simplify-m 10 \
  --graph-precision 4 \
  --max-road-samples 6 \
  --out hornby-preview.html
```

### B) Balanced output (good detail, moderate size)

```bash
python main.py \
  --ward "Hornby" \
  --max-addresses 5000 \
  --max-roads 50000 \
  --display-road-limit 6000 \
  --max-stops 1500 \
  --draw-simplify-m 6 \
  --graph-precision 5 \
  --max-road-samples 8 \
  --out hornby-balanced.html
```

### C) Higher-detail output (larger file)

```bash
python main.py \
  --ward "Hornby" \
  --max-addresses 12000 \
  --max-roads 120000 \
  --display-road-limit 14000 \
  --max-stops 4000 \
  --draw-simplify-m 2 \
  --graph-precision 6 \
  --max-road-samples 16 \
  --out hornby-detailed.html
```

## Additional ideas for even smaller outputs

Potential future improvements if you want to shrink files further:

- Add a CLI flag to skip route polylines and route metadata payloads entirely.
- Add a CLI flag to disable road/track `samples` payloads (trades off live recalculation fidelity).
- Split one large output into multiple pages by area and load per-area only.
- Post-process generated HTML with minifiers.
- Serve generated HTML with gzip/brotli compression (very effective over HTTP).

## Troubleshooting notes

- If generation is slow or output is too large, first reduce scope (`--ward`/`--bbox`) before lowering precision.
- If map visuals look too coarse, reduce `--draw-simplify-m`.
- If nearest-stop recalculation looks unstable after dragging stops, increase `--max-road-samples`.
