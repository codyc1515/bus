# bus

## Reducing output HTML file size

If you need smaller generated maps, these are the highest-impact knobs (already supported by `main.py`):

- Narrow the geographic scope with `--bbox`, `--ward`, `--town`, and/or `--ta`.
- Lower feature counts with `--max-addresses`, `--max-roads`, and `--max-stops`.
- Draw fewer road features with `--display-road-limit`.
- Keep the default limited route-shape rendering (avoid `--draw-all-route-shapes` unless needed).
- Increase geometry simplification with `--draw-simplify-m`.
- Reduce decimal precision in embedded JSON with `--graph-precision`.
- Reduce sampled points stored per road/track with `--max-road-samples`.

Example leaner build:

```bash
python main.py \
  --ward "Hornby" \
  --max-addresses 5000 \
  --max-roads 50000 \
  --display-road-limit 6000 \
  --max-stops 1500 \
  --draw-simplify-m 8 \
  --graph-precision 4 \
  --max-road-samples 8 \
  --out hornby-small.html
```

### Additional ideas (if you want even smaller files)

- Make routes optional with a flag (skip rendering route polylines and route metadata payloads).
- Make road/track `samples` optional (disables some live recalculation detail but cuts JSON size).
- Split output into multiple ward files (`--all-wards-out-dir`) and serve per-ward pages.
- Post-process generated HTML with minifiers (HTML/JS) and gzip/brotli at web-server level.
