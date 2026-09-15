---
name: visualize-mot-grids
description: >-
  Generate and host HTML tracker-comparison grids from existing
  experiments/_findings via visualize_all.sh (no re-detect / re-track).
  Use when visualizing tracker outputs, overlaying tracks, serving the
  viewer, comparing trackers side-by-side on video, or when the user
  mentions visualize_all, Dataset/Video dropdowns, N_SEQS, or snippet grids.
---

# Visualize MOT tracker grids

Never re-detect or re-track. Overlay the **same latest findings row** as
`compare_findings.py` (`mot_pipeline/findings.py:latest_record`) onto MOT
`img1` frames.

Always use `.venv/bin/python`. Output lives on the SSD, not the NAS:

`/media/7TBSSD/data/tracking/visualizations/`

## Two steps

```bash
# 1) generate JPEG cells + HTML (no server)
./scripts/batch/visualize_all.sh

# 2) host (separate terminal; headless box — do not xdg-open)
./scripts/batch/serve_visualizations.sh
```

Then open e.g. `http://127.0.0.1:8765/yolov8m_expert_eff/fps5/lumana_benchmark/index.html`.

Stop the server with Ctrl+C, or `pkill -f 'visualize_all.py --serve-only'`.

Python entry: `scripts/visualize/visualize_all.py`. Viewer HTML:
`scripts/visualize/grid_page.py`. Overlay: `scripts/visualize/visualize_mot_results.py`.

## Viewer

- One sequence at a time; all trackers tiled on screen.
- **Dataset** dropdown → sibling grids under the same FPS folder
  (`../<bench>/index.html`).
- **Video** dropdown → sequences that were actually rendered for that page.
- JPEG frames + JS scrub (not `<video>`). Play / pause / ±1 / arrows.

If the user sees only 3 videos: default `N_SEQS=3`. Tracking can exist for
every sequence while visualization snippets do not. Re-generate with `N_SEQS=0`
(all sequences that have `img1/`). Hard-refresh after generate.

## Env / flags (generate)

| Intent | How |
|--------|-----|
| One bench | `BENCHMARKS=lumana_benchmark` |
| One FPS | `FPS_VALUES="5"` (`5`, `10`, or `full`) |
| All sequences with frames | `N_SEQS=0` (default **3**) |
| Full clip, no intro skip | `SNIPPET_SEC=0` (default **8**; `START_FRAC=0.2` ignored when 0) |
| Re-encode existing cells | `FORCE=1` |
| Pin a sweep timestamp | `RUN_ID_SUBSTR=20260817_120720` |
| Rewrite HTML only | `.venv/bin/python scripts/visualize/visualize_all.py --html-only` |
| Catalog landing pages | `--refresh-index` |

Typical slice:

```bash
N_SEQS=0 BENCHMARKS=lumana_benchmark FPS_VALUES="5" ./scripts/batch/visualize_all.sh
```

Do **not** set `N_SEQS=0` for every benchmark without asking (UA-DETRAC 60 /
CityFlow 36 × 7 trackers is a large encode). Keep bash default at 3.

Defaults match `run_all.sh` YOLO:
`yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles` → stem `yolov8m_expert_eff`.

## Output layout

```
visualizations/index.html
  yolov8m_expert_eff/index.html
    fps5/<benchmark>/index.html
    fps5/<benchmark>/manifest.json
    fps5/<benchmark>/cells/<tracker>/<seq>/000000.jpg …
```

Sequences without `img1/` are skipped (Lumana GT-only seqs until videos land).
Cells already present are skipped unless `FORCE=1`.

## Wiring a new bench / tracker

- `scripts/batch/visualize_all.sh` — `BENCHMARKS` / `TRACKERS` defaults.
- `scripts/visualize/visualize_all.py` — `DEFAULT_BENCHMARKS`, `DEFAULT_TRACKERS`,
  `PREFERRED_SEQUENCES` (shown first when `N_SEQS>0`).
- `scripts/visualize/grid_page.py` — `DATASET_LABELS` for the Dataset dropdown.

Operator docs: `AGENTS.md`, `important_commands.md`, `README.md`.
