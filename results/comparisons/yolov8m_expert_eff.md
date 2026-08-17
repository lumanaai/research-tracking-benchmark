# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | pref_det | detector_id | FPS | ms/frame | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 2.39 | 12 | 0.534 | 0.414 | 0.710 | 0.463 | 0.602 | 0.953 | 271 | 844 | 259 | 298 | fasttracker_bench_fasttracker_yolov8_20260817_083521 |
| fasttracker_bench | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 3.23 | 12 | 0.547 | 0.425 | 0.720 | 0.482 | 0.617 | 0.963 | 256 | 1421 | 437 | 295 | fasttracker_bench_ocsort_yolov8_20260817_083521 |
| fasttracker_bench | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 4.91 | 12 | 0.546 | 0.425 | 0.718 | 0.482 | 0.615 | 0.963 | 218 | 1390 | 436 | 297 | fasttracker_bench_hybridsort_yolov8_20260817_083521 |
| fasttracker_bench | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 3.66 | 12 | 0.556 | **0.428** | 0.738 | **0.486** | 0.637 | **0.972** | **163** | 1354 | **449** | **284** | fasttracker_bench_analytics_bytetrack_yolov8_20260817_083521 |
| fasttracker_bench | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 6.07 | 12 | **0.560** | 0.427 | **0.752** | 0.484 | **0.639** | 0.968 | 222 | 1356 | 448 | **284** | fasttracker_bench_analytics_bytetrack_plus_yolov8_20260817_083521 |
| fasttracker_bench | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | **2.20** | 12 | 0.522 | 0.395 | 0.704 | 0.448 | 0.581 | 0.956 | 229 | **526** | 423 | 327 | fasttracker_bench_botsort_yolov8_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 2.11 | 60 | 0.477 | 0.362 | 0.633 | -0.213 | 0.573 | 0.989 | 370 | 2266 | 4234 | 260 | ua_detrac_fasttracker_yolov8_20260817_083521 |
| ua_detrac | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 2.93 | 60 | 0.520 | 0.398 | 0.682 | -0.067 | 0.614 | 0.988 | 432 | 1740 | 5129 | 166 | ua_detrac_ocsort_yolov8_20260817_083521 |
| ua_detrac | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 4.64 | 60 | 0.521 | 0.399 | 0.683 | -0.063 | 0.615 | 0.988 | 371 | 1674 | 5113 | 174 | ua_detrac_hybridsort_yolov8_20260817_083521 |
| ua_detrac | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 3.88 | 60 | 0.514 | 0.392 | 0.676 | -0.111 | 0.609 | **0.992** | **319** | 1958 | **5274** | **136** | ua_detrac_analytics_bytetrack_yolov8_20260817_083521 |
| ua_detrac | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 6.10 | 60 | 0.516 | 0.393 | 0.681 | -0.109 | 0.613 | 0.991 | 372 | 1946 | 5265 | **136** | ua_detrac_analytics_bytetrack_plus_yolov8_20260817_083521 |
| ua_detrac | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | **2.07** | 60 | **0.536** | **0.422** | **0.685** | **0.065** | **0.642** | 0.987 | 495 | **1073** | 5072 | 195 | ua_detrac_botsort_yolov8_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 6.22 | 27 | 0.562 | 0.396 | 0.823 | 0.406 | 0.647 | **0.991** | 16 | 186 | 315 | 253 | trafficmot_fasttracker_yolov8_20260817_083521 |
| trafficmot | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.80 | 27 | 0.581 | 0.407 | 0.855 | 0.441 | 0.660 | 0.990 | 20 | 237 | 332 | 257 | trafficmot_ocsort_yolov8_20260817_083521 |
| trafficmot | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 5.58 | 27 | 0.581 | 0.406 | 0.858 | 0.441 | 0.661 | 0.990 | **15** | 234 | 333 | 259 | trafficmot_hybridsort_yolov8_20260817_083521 |
| trafficmot | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.35 | 27 | **0.593** | **0.423** | 0.858 | **0.454** | **0.679** | 0.987 | 41 | 275 | **356** | **226** | trafficmot_analytics_bytetrack_yolov8_20260817_083521 |
| trafficmot | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 4.93 | 27 | **0.593** | **0.423** | 0.858 | **0.454** | **0.679** | 0.987 | 39 | 277 | **356** | **226** | trafficmot_analytics_bytetrack_plus_yolov8_20260817_083521 |
| trafficmot | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | **2.94** | 27 | 0.557 | 0.372 | **0.859** | 0.408 | 0.624 | 0.989 | 42 | **140** | 313 | 306 | trafficmot_botsort_yolov8_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 1.60 | 36 | 0.259 | 0.131 | 0.543 | -3.030 | 0.302 | 0.987 | **42** | 579 | 473 | 102 | cityflow_fasttracker_yolov8_20260817_083521 |
| cityflow | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 2.05 | 36 | 0.272 | 0.138 | 0.557 | -2.894 | 0.321 | 0.982 | 85 | 628 | 724 | **4** | cityflow_ocsort_yolov8_20260817_083521 |
| cityflow | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.51 | 36 | 0.273 | 0.138 | **0.563** | -2.886 | 0.322 | 0.982 | 72 | 624 | 722 | 5 | cityflow_hybridsort_yolov8_20260817_083521 |
| cityflow | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 2.20 | 36 | 0.264 | 0.137 | 0.528 | -2.965 | 0.317 | **0.990** | 49 | 589 | **766** | **4** | cityflow_analytics_bytetrack_yolov8_20260817_083521 |
| cityflow | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.62 | 36 | 0.266 | 0.137 | 0.539 | -2.962 | 0.317 | 0.985 | 77 | 614 | 751 | **4** | cityflow_analytics_bytetrack_plus_yolov8_20260817_083521 |
| cityflow | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | **1.42** | 36 | **0.281** | **0.150** | 0.549 | **-2.376** | **0.347** | 0.983 | 91 | **496** | 736 | 11 | cityflow_botsort_yolov8_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 20 | 1.26 | 24 | 0.764 | 0.743 | 0.795 | 0.809 | 0.801 | 0.896 | 275 | 720 | 305 | 61 | lumana_benchmark_fasttracker_yolov8_20260817_083521 |
| lumana_benchmark | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 20 | 1.59 | 24 | 0.778 | 0.759 | 0.804 | 0.820 | 0.818 | 0.904 | 272 | 985 | 291 | 58 | lumana_benchmark_ocsort_yolov8_20260817_083521 |
| lumana_benchmark | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 20 | 2.44 | 24 | 0.779 | **0.760** | 0.805 | **0.821** | 0.821 | 0.911 | 273 | 993 | 288 | 59 | lumana_benchmark_hybridsort_yolov8_20260817_083521 |
| lumana_benchmark | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 20 | 1.42 | 24 | 0.781 | 0.755 | 0.814 | 0.813 | 0.822 | 0.922 | 259 | 1012 | **320** | **46** | lumana_benchmark_analytics_bytetrack_yolov8_20260817_083521 |
| lumana_benchmark | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 20 | 2.27 | 24 | **0.796** | 0.755 | **0.846** | 0.812 | **0.847** | **0.934** | 264 | 1036 | 317 | **46** | lumana_benchmark_analytics_bytetrack_plus_yolov8_20260817_083521 |
| lumana_benchmark | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 20 | **1.10** | 24 | 0.750 | 0.716 | 0.793 | 0.771 | 0.778 | 0.894 | **254** | **691** | 283 | 88 | lumana_benchmark_botsort_yolov8_20260817_083521 |

## Metric glossary

- **HOTA** — Higher Order Tracking Accuracy — geometric mean of detection (DetA) and association (AssA) accuracy over IoU thresholds; primary overall ranking metric.
- **DetA** — Detection Accuracy — how well predicted boxes cover GT detections (localization + presence), independent of ID quality.
- **AssA** — Association Accuracy — how consistently the same tracker ID stays on the same GT identity over time (identity preservation).
- **MOTA** — Multiple Object Tracking Accuracy — CLEAR metric: 1 − (FN + FP + IDSW) / GT_dets. Sensitive to detector FP/FN; can go negative.
- **IDF1** — ID F1 — harmonic mean of ID precision/recall from bipartite ID matching (Identity metrics). Strong signal for ID stability.
- **IDCons** — ID Consistency — mean per-GT purity of the tracker IDs assigned to that object (how little a single GT is fragmented across tracker IDs).
- **IDSW** — ID Switches — times a GT trajectory changes which tracker ID it is matched to (↓ better).
- **Frag** — Fragmentations — times a tracked GT goes from matched → unmatched → matched again (trajectory breaks; ↓ better).
- **MT** — Mostly Tracked — GT trajectories covered for ≥80% of their lifetime (↑ better).
- **ML** — Mostly Lost — GT trajectories covered for ≤20% of their lifetime (↓ better).
- **FPS** — Effective video frame rate fed to the tracker (target --fps, else native benchmark rate). Not compute throughput.
- **ms/frame** — Average tracker compute time per processed frame (association only; excludes detector). Per-run when timing.json exists; otherwise a reference timing on dense GT dets (task_day_occlusion).
- **pref_det** — Detector the method prefers upstream / in production (YOLOX for most SORT-family papers; YOLO for in-house ByteTrack variants). This pipeline often sweeps YOLOv8m-expert or GT boxes instead.
