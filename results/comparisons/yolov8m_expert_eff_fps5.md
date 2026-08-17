# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | pref_det | detector_id | FPS | ms/frame | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 4.88 | 12 | 0.529 | 0.404 | 0.720 | 0.436 | 0.608 | 0.921 | 326 | 458 | 238 | 283 | fasttracker_bench_fasttracker_yolov8_fps5_20260817_120720 |
| fasttracker_bench | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 5.34 | 12 | 0.539 | 0.407 | 0.732 | 0.458 | 0.626 | **0.972** | 120 | 638 | 213 | 315 | fasttracker_bench_ocsort_yolov8_fps5_20260817_120720 |
| fasttracker_bench | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 8.19 | 12 | 0.538 | 0.407 | 0.728 | 0.458 | 0.625 | 0.970 | **115** | 622 | 214 | 312 | fasttracker_bench_hybridsort_yolov8_fps5_20260817_120720 |
| fasttracker_bench | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 4.96 | 12 | **0.558** | **0.429** | 0.743 | **0.481** | **0.653** | 0.961 | 160 | 932 | 360 | **226** | fasttracker_bench_analytics_bytetrack_yolov8_fps5_20260817_120720 |
| fasttracker_bench | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 7.56 | 12 | **0.558** | 0.428 | **0.744** | **0.481** | **0.653** | **0.972** | 128 | 926 | 366 | 228 | fasttracker_bench_analytics_bytetrack_plus_yolov8_fps5_20260817_120720 |
| fasttracker_bench | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | **3.73** | 12 | 0.517 | 0.389 | 0.706 | 0.443 | 0.590 | 0.894 | 586 | **329** | **398** | 260 | fasttracker_bench_botsort_yolov8_fps5_20260817_120720 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 3.75 | 60 | 0.364 | 0.263 | 0.518 | -0.579 | 0.418 | 0.773 | 9143 | 1991 | 2356 | 1290 | ua_detrac_fasttracker_yolov8_fps5_20260817_120720 |
| ua_detrac | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 6.50 | 60 | 0.421 | 0.282 | 0.632 | -0.218 | 0.475 | **0.965** | 537 | 600 | 1251 | 2658 | ua_detrac_ocsort_yolov8_fps5_20260817_120720 |
| ua_detrac | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 10.03 | 60 | 0.424 | 0.283 | 0.638 | -0.209 | 0.476 | **0.965** | **531** | **593** | 1297 | 2746 | ua_detrac_hybridsort_yolov8_fps5_20260817_120720 |
| ua_detrac | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 4.11 | 60 | 0.487 | 0.379 | 0.629 | -0.110 | 0.573 | 0.918 | 2567 | 1255 | 4047 | 415 | ua_detrac_analytics_bytetrack_yolov8_fps5_20260817_120720 |
| ua_detrac | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 6.28 | 60 | **0.495** | 0.378 | **0.649** | -0.101 | **0.585** | 0.949 | 1506 | 972 | 4003 | 502 | ua_detrac_analytics_bytetrack_plus_yolov8_fps5_20260817_120720 |
| ua_detrac | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | **2.58** | 60 | 0.446 | **0.405** | 0.497 | **-0.071** | 0.499 | 0.604 | 23914 | 1115 | **4603** | **301** | ua_detrac_botsort_yolov8_fps5_20260817_120720 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 15.04 | 27 | 0.529 | 0.354 | 0.815 | 0.342 | 0.599 | 0.986 | 33 | **71** | 284 | 305 | trafficmot_fasttracker_yolov8_fps5_20260817_120720 |
| trafficmot | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 7.09 | 27 | 0.549 | 0.364 | 0.849 | 0.380 | 0.601 | 0.947 | 82 | 113 | 280 | 297 | trafficmot_ocsort_yolov8_fps5_20260817_120720 |
| trafficmot | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 8.07 | 27 | 0.549 | 0.364 | 0.850 | 0.379 | 0.601 | 0.947 | 79 | 116 | 277 | 297 | trafficmot_hybridsort_yolov8_fps5_20260817_120720 |
| trafficmot | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | **4.09** | 27 | 0.577 | **0.402** | 0.855 | **0.434** | 0.656 | **0.993** | 15 | 139 | **322** | **258** | trafficmot_analytics_bytetrack_yolov8_fps5_20260817_120720 |
| trafficmot | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 5.31 | 27 | **0.578** | **0.402** | **0.856** | **0.434** | **0.657** | **0.993** | **13** | 139 | **322** | **258** | trafficmot_analytics_bytetrack_plus_yolov8_fps5_20260817_120720 |
| trafficmot | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 4.47 | 27 | 0.521 | 0.345 | 0.809 | 0.362 | 0.565 | 0.919 | 260 | 73 | 284 | 328 | trafficmot_botsort_yolov8_fps5_20260817_120720 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 2.01 | 36 | 0.251 | 0.126 | 0.551 | -2.896 | 0.292 | 0.963 | 149 | 512 | 424 | 131 | cityflow_fasttracker_yolov8_fps5_20260817_120720 |
| cityflow | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 2.66 | 36 | 0.268 | 0.136 | 0.570 | -2.609 | 0.315 | 0.966 | 97 | 505 | 496 | 110 | cityflow_ocsort_yolov8_fps5_20260817_120720 |
| cityflow | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 3.99 | 36 | 0.269 | 0.136 | **0.574** | -2.606 | 0.316 | 0.966 | 101 | 506 | 489 | 112 | cityflow_hybridsort_yolov8_fps5_20260817_120720 |
| cityflow | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 2.35 | 36 | 0.264 | 0.137 | 0.550 | -2.722 | 0.319 | **0.978** | **81** | 513 | 669 | 28 | cityflow_analytics_bytetrack_yolov8_fps5_20260817_120720 |
| cityflow | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 3.89 | 36 | 0.257 | 0.136 | 0.512 | -2.731 | 0.310 | 0.969 | 124 | 547 | 665 | 44 | cityflow_analytics_bytetrack_plus_yolov8_fps5_20260817_120720 |
| cityflow | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | **1.71** | 36 | **0.277** | **0.150** | 0.568 | **-2.187** | **0.336** | 0.867 | 850 | **439** | **696** | **11** | cityflow_botsort_yolov8_fps5_20260817_120720 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker (2025) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 2.01 | 24 | 0.789 | 0.744 | 0.841 | 0.779 | 0.817 | 0.864 | 290 | 470 | 216 | 110 | lumana_benchmark_fasttracker_yolov8_fps5_20260817_120720 |
| lumana_benchmark | ocsort (2023) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 2.29 | 24 | 0.821 | 0.775 | 0.872 | 0.803 | 0.851 | 0.916 | **153** | 518 | 159 | 158 | lumana_benchmark_ocsort_yolov8_fps5_20260817_120720 |
| lumana_benchmark | hybridsort (2024) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 3.15 | 24 | 0.813 | 0.771 | 0.859 | 0.800 | 0.843 | **0.918** | 156 | 504 | 154 | 159 | lumana_benchmark_hybridsort_yolov8_fps5_20260817_120720 |
| lumana_benchmark | analytics_bytetrack (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 1.63 | 24 | 0.835 | **0.804** | 0.868 | **0.832** | 0.864 | 0.909 | 257 | 729 | **253** | **77** | lumana_benchmark_analytics_bytetrack_yolov8_fps5_20260817_120720 |
| lumana_benchmark | analytics_bytetrack_plus (2022) | YOLO | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | 2.68 | 24 | **0.844** | 0.802 | **0.891** | 0.830 | **0.881** | 0.910 | 279 | 762 | 243 | 89 | lumana_benchmark_analytics_bytetrack_plus_yolov8_fps5_20260817_120720 |
| lumana_benchmark | botsort (2022) | YOLOX | yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles | 5 | **1.30** | 24 | 0.757 | 0.696 | 0.825 | 0.718 | 0.764 | 0.795 | 501 | **419** | 207 | 116 | lumana_benchmark_botsort_yolov8_fps5_20260817_120720 |

## Notes

- **YOLO cache.** Each `detector_id` encodes weights, `imgsz`, `conf`, and whether motorcycles are kept. The current pipeline default is `yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles`: Ultralytics `imgsz=(704, 1280)` (H×W), `conf=0.30`, keep-set bicycle/car/motorcycle/bus/truck/forklift/boat (`1,2,3,4,6,19,23`). Person is dropped. An older square-letterbox cache `…_imgsz1280_conf0.25_vehicles` (keep-set without bicycle/boat) still exists on disk for full-rate / 10 FPS tables until those are re-detected.
- **NMS / preprocess vs in-house.** This pipeline uses Ultralytics default NMS IoU `0.7` and Ultralytics letterbox. Production may differ in NMS IoU, letterbox vs stretch, package version, or post-NMS class filtering. A one-sequence replay against an in-house 5 FPS export matched boxes at mean IoU ~0.99; leftover extras were a few percent of boxes (concentrated on a couple of IDs), not a global association mismatch.
- **Bicycle / boat vs vehicle GT.** Those classes are kept to match product detections. On vehicle-only ground truth (e.g. Lumana class=1) they can count as false positives and slightly lower MOTA/DetA.
- **`--fps` eval.** When tracking at a target FPS, TrackEval GT is filtered to the same kept frames as the tracker (skipped frames do not exist). Full-rate eval is unchanged.
- **IDCons** is mean per-GT modal tracker-ID purity in this repo's TrackEval patch. Other groups may report a different identity metric under a similar name.

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
