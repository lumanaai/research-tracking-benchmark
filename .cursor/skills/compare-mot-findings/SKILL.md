---
name: compare-mot-findings
description: >-
  Build MOT tracker comparison tables from experiments/_findings via
  compare_findings.py. Use when comparing trackers, runs, detectors, or FPS
  settings; regenerating results/comparisons tables; or summarizing HOTA/MOTA/IDF1/IDCons.
---

# Compare MOT findings

Always **run** `scripts/analysis/compare_findings.py` (or `run_all_compare_findings.sh`). Do not hand-build markdown tables from raw CSVs unless the script cannot express the filter.

## Project constants

- Always use `.venv/bin/python`.
- Findings: `/media/7TBSSD/data/tracking/experiments/_findings/<bench>/<tracker>/<detector_id>/findings.csv`
- Published tables: `results/comparisons/` (project copy). Prefer `--format both` and `--out results/comparisons/<stem>.md`.

## Workflow

1. Clarify intent: which detector? full-rate vs target FPS? all benches or one? one tracker or all?
2. Map to flags (below). Default = latest row per `bench × tracker × detector_id × target_fps`.
3. Run the script; show the user the output path(s).
4. If empty: say which combo is missing (tracker not swept yet, wrong `detector_id`, or FPS filter).

## Flag cheat sheet

| Intent | Flags |
|--------|--------|
| GT oracle, full rate | `--detector-id gt_vehicles` |
| GT @ 5 / 10 FPS | `--detector-id gt_vehicles --target-fps 5` (or `10`) |
| Expert YOLO, full rate (legacy square 1280 / conf 0.25) | `--detector-id yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles` |
| Expert YOLO, product default (704×1280 / conf 0.3) | `--detector-id yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles` |
| Expert YOLO @ FPS | matching detector-id + `--target-fps N` |
| One benchmark | `--benchmark ua_detrac` |
| One tracker | `--tracker botsort` |
| Sweep timestamp / run family | `--run-id-substr 20260729_085336` |
| Keep every historical row | `--all-rows` |
| md + csv | `--format both` |

Omit `--target-fps` for full-rate runs (table shows each bench’s native FPS).

### Standard refresh (all published tables)

```bash
./scripts/analysis/run_all_compare_findings.sh
```

### One-off examples

```bash
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --format both \
  --out results/comparisons/gt_vehicles.md

.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --target-fps 5 --format both \
  --out results/comparisons/gt_vehicles_fps5.md

.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles --target-fps 5 --format both \
  --out results/comparisons/yolov8m_expert_eff_fps5.md
```

Naming: `<detector_stem>.md` for full rate; append `_fps5` / `_fps10` when `--target-fps` is set. Shorten long YOLO ids in the filename (e.g. `yolov8m_expert_eff`) but keep the **full** `--detector-id`.

Discover ids if unsure:

```bash
find /media/7TBSSD/data/tracking/experiments/_findings -type d -mindepth 3 -maxdepth 3 | sort
```

More detector aliases and interpretation notes: [reference.md](reference.md).

## How to read the table

- Tracker column shows `name (year)`; `pref_det` is the detector the paper/upstream designed for (usually YOLOX).
- `ms/frame` is association-only compute time (detector excluded). Per-run when `timing.json` exists; else reference from `results/comparisons/tracker_speed.json` (dense GT on `task_day_occlusion`).
- **GT (`gt_vehicles`)**: association-only (perfect boxes). High DetA expected; gaps are IDSW/Frag/AssA.
- **YOLO / existing**: joint detector+tracker score; DetA is capped by the detector.
- Bold cells = best within that benchmark block (↑ HOTA/DetA/AssA/MOTA/IDF1/IDCons/MT; ↓ IDSW/Frag/ML/ms/frame).
- Prefer HOTA + AssA + IDF1 for tracker ranking; MOTA alone can mislead under detector FP/FN.
- Each markdown table ends with a **Metric glossary** covering HOTA/DetA/AssA/MOTA/IDF1/IDCons/IDSW/Frag/MT/ML/FPS/ms/frame/pref_det.

Side-by-side video of the same latest findings rows: `./scripts/batch/visualize_all.sh` then `./scripts/batch/serve_visualizations.sh` (skill: `visualize-mot-grids`). Does not re-track.

### Refresh speed refs

```bash
.venv/bin/python scripts/analysis/bench_tracker_speed.py
```
