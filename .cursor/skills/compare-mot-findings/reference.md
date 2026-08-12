# Compare MOT findings — reference

## Common `detector_id` values

| Role | `detector_id` |
|------|----------------|
| GT oracle | `gt_vehicles` |
| Reused seq-local dets | `existing_det_vehicles` |
| Team expert YOLO (default pipeline) | `yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles` |
| Stock YOLOv8s cache (if present) | `yolov8s_imgsz1280_conf0.25_vehicles` |

Exact strings are directory names under `_findings/<bench>/<tracker>/`. When the user says “expert YOLO” or “yolov8m”, use the expert id above unless they name another cache.

## Native FPS (full-rate, no `--target-fps`)

| benchmark | native FPS |
|-----------|------------|
| `fasttracker_bench` | 30 |
| `ua_detrac` | 25 |
| `trafficmot` | 10 |
| `cityflow` | 10 |

`--target-fps N` keeps only runs tracked at that subsampled rate (run_id usually contains `_fpsN_`).

## Script behavior

- Source: `scripts/analysis/compare_findings.py`
- Default: latest `evaluated_at` per bench×tracker×detector_id×target_fps.
- Sort: `BENCH_ORDER` then `TRACKER_ORDER` (missing trackers simply absent).
- Without `--out` and with `--format md`: prints table and also writes under `experiments/_comparisons/` on the SSD — still prefer explicit `--out results/comparisons/…` for the project copy.

## Cross-detector side-by-side

The script emits one detector filter at a time. For GT vs YOLO in one narrative table, run twice (or use `run_all_compare_findings.sh`) and join manually only if the user asks — do not invent metrics.
