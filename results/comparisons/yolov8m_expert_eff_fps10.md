# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | pref_det | detector_id | FPS | ms/frame | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.11 | 12 | 0.537 | 0.406 | 0.734 | 0.445 | 0.606 | 0.953 | 218 | 585 | 226 | 330 | fasttracker_bench_fasttracker_yolov8_fps10_20260817_083521 |
| fasttracker_bench | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.47 | 12 | 0.555 | 0.421 | 0.749 | 0.478 | 0.632 | 0.975 | 164 | 808 | 310 | 304 | fasttracker_bench_ocsort_yolov8_fps10_20260817_083521 |
| fasttracker_bench | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 5.50 | 12 | 0.556 | 0.421 | 0.751 | 0.478 | 0.633 | 0.975 | 141 | 790 | 307 | 304 | fasttracker_bench_hybridsort_yolov8_fps10_20260817_083521 |
| fasttracker_bench | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.21 | 12 | **0.564** | **0.429** | 0.758 | **0.487** | 0.644 | 0.974 | **122** | 857 | **434** | **294** | fasttracker_bench_analytics_bytetrack_yolov8_fps10_20260817_083521 |
| fasttracker_bench | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 5.74 | 12 | **0.564** | 0.428 | **0.761** | **0.487** | **0.648** | **0.976** | 140 | 874 | 432 | **294** | fasttracker_bench_analytics_bytetrack_plus_yolov8_fps10_20260817_083521 |
| fasttracker_bench | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | **2.21** | 12 | 0.530 | 0.395 | 0.729 | 0.449 | 0.595 | 0.958 | 243 | **373** | 413 | 329 | fasttracker_bench_botsort_yolov8_fps10_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 2.43 | 60 | 0.453 | 0.336 | 0.617 | -0.300 | 0.547 | 0.986 | 397 | 2104 | 3316 | 365 | ua_detrac_fasttracker_yolov8_fps10_20260817_083521 |
| ua_detrac | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.21 | 60 | 0.518 | 0.396 | 0.680 | -0.056 | 0.611 | 0.987 | 369 | 1018 | 4699 | 209 | ua_detrac_ocsort_yolov8_fps10_20260817_083521 |
| ua_detrac | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 5.07 | 60 | 0.518 | 0.396 | 0.680 | -0.053 | 0.612 | 0.987 | 365 | 1022 | 4682 | 214 | ua_detrac_hybridsort_yolov8_fps10_20260817_083521 |
| ua_detrac | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.42 | 60 | 0.516 | 0.393 | 0.679 | -0.101 | 0.612 | **0.990** | 314 | 1232 | **5161** | **134** | ua_detrac_analytics_bytetrack_yolov8_fps10_20260817_083521 |
| ua_detrac | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 5.84 | 60 | 0.519 | 0.394 | **0.686** | -0.097 | 0.616 | **0.990** | **292** | 1227 | 5156 | 137 | ua_detrac_analytics_bytetrack_plus_yolov8_fps10_20260817_083521 |
| ua_detrac | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | **2.00** | 60 | **0.529** | **0.414** | 0.682 | **0.073** | **0.643** | 0.984 | 611 | **866** | 4987 | 216 | ua_detrac_botsort_yolov8_fps10_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 6.70 | 27 | 0.562 | 0.396 | 0.823 | 0.406 | 0.647 | **0.991** | 16 | 186 | 315 | 253 | trafficmot_fasttracker_yolov8_fps10_20260817_083521 |
| trafficmot | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.79 | 27 | 0.581 | 0.407 | 0.855 | 0.441 | 0.660 | 0.990 | 20 | 237 | 332 | 257 | trafficmot_ocsort_yolov8_fps10_20260817_083521 |
| trafficmot | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 5.98 | 27 | 0.581 | 0.406 | 0.858 | 0.441 | 0.661 | 0.990 | **15** | 234 | 333 | 259 | trafficmot_hybridsort_yolov8_fps10_20260817_083521 |
| trafficmot | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.14 | 27 | **0.593** | **0.423** | 0.858 | **0.454** | **0.679** | 0.987 | 41 | 275 | **356** | **226** | trafficmot_analytics_bytetrack_yolov8_fps10_20260817_083521 |
| trafficmot | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 4.44 | 27 | **0.593** | **0.423** | 0.858 | **0.454** | **0.679** | 0.987 | 39 | 277 | **356** | **226** | trafficmot_analytics_bytetrack_plus_yolov8_fps10_20260817_083521 |
| trafficmot | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | **2.82** | 27 | 0.557 | 0.372 | **0.859** | 0.408 | 0.624 | 0.989 | 42 | **140** | 313 | 306 | trafficmot_botsort_yolov8_fps10_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 1.55 | 36 | 0.259 | 0.131 | 0.543 | -3.030 | 0.302 | 0.987 | **42** | 579 | 473 | 102 | cityflow_fasttracker_yolov8_fps10_20260817_083521 |
| cityflow | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 2.09 | 36 | 0.272 | 0.138 | 0.557 | -2.894 | 0.321 | 0.982 | 85 | 628 | 724 | **4** | cityflow_ocsort_yolov8_fps10_20260817_083521 |
| cityflow | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.23 | 36 | 0.273 | 0.138 | **0.563** | -2.886 | 0.322 | 0.982 | 72 | 624 | 722 | 5 | cityflow_hybridsort_yolov8_fps10_20260817_083521 |
| cityflow | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 2.09 | 36 | 0.264 | 0.137 | 0.528 | -2.965 | 0.317 | **0.990** | 49 | 589 | **766** | **4** | cityflow_analytics_bytetrack_yolov8_fps10_20260817_083521 |
| cityflow | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 3.52 | 36 | 0.266 | 0.137 | 0.539 | -2.962 | 0.317 | 0.985 | 77 | 614 | 751 | **4** | cityflow_analytics_bytetrack_plus_yolov8_fps10_20260817_083521 |
| cityflow | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | **1.43** | 36 | **0.281** | **0.150** | 0.549 | **-2.376** | **0.347** | 0.983 | 91 | **496** | 736 | 11 | cityflow_botsort_yolov8_fps10_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 1.34 | 24 | 0.759 | 0.728 | 0.800 | 0.793 | 0.803 | 0.899 | 258 | 616 | 299 | 74 | lumana_benchmark_fasttracker_yolov8_fps10_20260817_083521 |
| lumana_benchmark | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 1.83 | 24 | 0.783 | **0.754** | 0.820 | **0.812** | 0.829 | 0.917 | 196 | 805 | 275 | 66 | lumana_benchmark_ocsort_yolov8_fps10_20260817_083521 |
| lumana_benchmark | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 2.53 | 24 | 0.785 | 0.753 | 0.825 | 0.811 | 0.831 | 0.924 | **189** | 788 | 269 | 70 | lumana_benchmark_hybridsort_yolov8_fps10_20260817_083521 |
| lumana_benchmark | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 1.50 | 24 | 0.777 | 0.749 | 0.813 | 0.803 | 0.823 | 0.922 | 230 | 862 | **322** | **47** | lumana_benchmark_analytics_bytetrack_yolov8_fps10_20260817_083521 |
| lumana_benchmark | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 2.59 | 24 | **0.799** | 0.746 | **0.863** | 0.799 | **0.856** | **0.939** | 236 | 877 | 315 | **47** | lumana_benchmark_analytics_bytetrack_plus_yolov8_fps10_20260817_083521 |
| lumana_benchmark | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | **1.20** | 24 | 0.751 | 0.706 | 0.807 | 0.761 | 0.784 | 0.897 | 234 | **593** | 279 | 92 | lumana_benchmark_botsort_yolov8_fps10_20260817_083521 |

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
