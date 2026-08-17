# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | pref_det | detector_id | FPS | ms/frame | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker (2025) | YOLOX | gt_vehicles | 5 | 5.22 | 12 | 0.919 | 0.900 | 0.940 | 0.937 | 0.961 | 0.949 | 302 | 243 | 776 | 52 | fasttracker_bench_fasttracker_gt_fps5_20260817_083521 |
| fasttracker_bench | ocsort (2023) | YOLOX | gt_vehicles | 5 | 5.55 | 12 | 0.968 | 0.971 | 0.965 | 0.970 | 0.976 | **0.986** | 82 | 75 | 740 | 76 | fasttracker_bench_ocsort_gt_fps5_20260817_083521 |
| fasttracker_bench | hybridsort (2024) | YOLOX | gt_vehicles | 5 | 10.22 | 12 | 0.966 | 0.971 | 0.961 | 0.971 | 0.974 | 0.985 | **79** | 70 | 738 | 77 | fasttracker_bench_hybridsort_gt_fps5_20260817_083521 |
| fasttracker_bench | analytics_bytetrack (2022) | YOLO | gt_vehicles | 5 | 4.91 | 12 | **0.984** | **0.991** | 0.977 | 0.990 | **0.987** | 0.970 | 152 | 70 | 929 | 9 | fasttracker_bench_analytics_bytetrack_gt_fps5_20260817_083521 |
| fasttracker_bench | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 5 | 8.61 | 12 | **0.984** | **0.991** | **0.978** | 0.990 | **0.987** | 0.984 | 96 | 58 | 936 | 11 | fasttracker_bench_analytics_bytetrack_plus_gt_fps5_20260817_083521 |
| fasttracker_bench | botsort (2022) | YOLOX | gt_vehicles | 5 | **3.95** | 12 | 0.971 | 0.975 | 0.968 | **0.993** | 0.986 | 0.924 | 725 | **5** | **1017** | **0** | fasttracker_bench_botsort_gt_fps5_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker (2025) | YOLOX | gt_vehicles | 5 | 2.01 | 60 | 0.588 | 0.524 | 0.681 | 0.426 | 0.676 | 0.781 | 10204 | 1948 | 3102 | 906 | ua_detrac_fasttracker_gt_fps5_20260817_083521 |
| ua_detrac | ocsort (2023) | YOLOX | gt_vehicles | 5 | 4.42 | 60 | 0.719 | 0.619 | 0.834 | 0.615 | 0.754 | 0.966 | 515 | 330 | 1304 | 2626 | ua_detrac_ocsort_gt_fps5_20260817_083521 |
| ua_detrac | hybridsort (2024) | YOLOX | gt_vehicles | 5 | 7.45 | 60 | 0.719 | 0.615 | 0.842 | 0.611 | 0.751 | **0.967** | **485** | 310 | 1325 | 2782 | ua_detrac_hybridsort_gt_fps5_20260817_083521 |
| ua_detrac | analytics_bytetrack (2022) | YOLO | gt_vehicles | 5 | 2.56 | 60 | 0.904 | **0.923** | 0.885 | 0.904 | 0.921 | 0.927 | 2493 | 778 | 4751 | 117 | ua_detrac_analytics_bytetrack_gt_fps5_20260817_083521 |
| ua_detrac | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 5 | 4.55 | 60 | **0.916** | 0.921 | **0.910** | **0.912** | **0.938** | 0.964 | 1217 | 383 | 4767 | 216 | ua_detrac_analytics_bytetrack_plus_gt_fps5_20260817_083521 |
| ua_detrac | botsort (2022) | YOLOX | gt_vehicles | 5 | **1.53** | 60 | 0.757 | 0.870 | 0.661 | 0.789 | 0.778 | 0.623 | 26009 | **1** | **5949** | **0** | ua_detrac_botsort_gt_fps5_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker (2025) | YOLOX | gt_vehicles | 5 | 11.06 | 27 | 0.855 | 0.819 | 0.896 | 0.921 | 0.958 | 0.991 | 43 | 43 | 671 | 7 | trafficmot_fasttracker_gt_fps5_20260817_083521 |
| trafficmot | ocsort (2023) | YOLOX | gt_vehicles | 5 | 5.98 | 27 | 0.966 | 0.961 | 0.972 | 0.951 | 0.969 | 0.968 | 95 | 54 | 656 | 13 | trafficmot_ocsort_gt_fps5_20260817_083521 |
| trafficmot | hybridsort (2024) | YOLOX | gt_vehicles | 5 | 8.95 | 27 | 0.967 | 0.962 | 0.972 | 0.952 | 0.969 | 0.968 | 96 | 53 | 657 | 13 | trafficmot_hybridsort_gt_fps5_20260817_083521 |
| trafficmot | analytics_bytetrack (2022) | YOLO | gt_vehicles | 5 | **3.92** | 27 | 0.991 | **0.992** | 0.990 | 0.990 | 0.993 | 0.995 | 23 | 25 | 710 | 3 | trafficmot_analytics_bytetrack_gt_fps5_20260817_083521 |
| trafficmot | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 5 | 5.30 | 27 | **0.992** | **0.992** | **0.991** | **0.991** | **0.994** | **0.997** | **15** | 21 | 711 | 3 | trafficmot_analytics_bytetrack_plus_gt_fps5_20260817_083521 |
| trafficmot | botsort (2022) | YOLOX | gt_vehicles | 5 | 4.51 | 27 | 0.937 | 0.937 | 0.939 | 0.981 | 0.981 | 0.973 | 182 | **19** | **731** | **0** | trafficmot_botsort_gt_fps5_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker (2025) | YOLOX | gt_vehicles | 5 | 0.74 | 36 | 0.829 | 0.816 | 0.846 | 0.899 | 0.923 | 0.967 | 149 | 173 | 677 | 65 | cityflow_fasttracker_gt_fps5_20260817_083521 |
| cityflow | ocsort (2023) | YOLOX | gt_vehicles | 5 | 0.94 | 36 | 0.906 | 0.911 | 0.901 | 0.909 | 0.926 | 0.975 | 69 | **132** | 497 | 120 | cityflow_ocsort_gt_fps5_20260817_083521 |
| cityflow | hybridsort (2024) | YOLOX | gt_vehicles | 5 | 1.33 | 36 | 0.905 | 0.911 | 0.899 | 0.909 | 0.925 | 0.975 | 72 | 135 | 498 | 120 | cityflow_hybridsort_gt_fps5_20260817_083521 |
| cityflow | analytics_bytetrack (2022) | YOLO | gt_vehicles | 5 | 0.73 | 36 | **0.949** | **0.973** | **0.925** | 0.972 | 0.948 | **0.983** | **62** | 169 | 689 | 35 | cityflow_analytics_bytetrack_gt_fps5_20260817_083521 |
| cityflow | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 5 | 1.06 | 36 | 0.934 | 0.971 | 0.897 | 0.969 | 0.935 | 0.977 | 100 | 204 | 683 | 37 | cityflow_analytics_bytetrack_plus_gt_fps5_20260817_083521 |
| cityflow | botsort (2022) | YOLOX | gt_vehicles | 5 | **0.63** | 36 | 0.926 | 0.942 | 0.912 | **0.984** | **0.959** | 0.903 | 532 | 169 | **806** | **0** | cityflow_botsort_gt_fps5_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker (2025) | YOLOX | gt_vehicles | 5 | 1.80 | 24 | 0.892 | 0.901 | 0.885 | 0.934 | 0.915 | 0.878 | 355 | 458 | 329 | 51 | lumana_benchmark_fasttracker_gt_fps5_20260817_083521 |
| lumana_benchmark | ocsort (2023) | YOLOX | gt_vehicles | 5 | 1.95 | 24 | 0.927 | 0.959 | 0.896 | 0.956 | 0.921 | **0.917** | **185** | **376** | 244 | 113 | lumana_benchmark_ocsort_gt_fps5_20260817_083521 |
| lumana_benchmark | hybridsort (2024) | YOLOX | gt_vehicles | 5 | 2.86 | 24 | 0.913 | 0.957 | 0.871 | 0.954 | 0.904 | 0.909 | 221 | 394 | 240 | 113 | lumana_benchmark_hybridsort_gt_fps5_20260817_083521 |
| lumana_benchmark | analytics_bytetrack (2022) | YOLO | gt_vehicles | 5 | 1.53 | 24 | 0.949 | **0.990** | 0.910 | **0.985** | 0.941 | 0.908 | 284 | 433 | 367 | 34 | lumana_benchmark_analytics_bytetrack_gt_fps5_20260817_083521 |
| lumana_benchmark | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 5 | 2.38 | 24 | **0.956** | 0.987 | **0.926** | 0.983 | **0.952** | 0.910 | 298 | 465 | 360 | 41 | lumana_benchmark_analytics_bytetrack_plus_gt_fps5_20260817_083521 |
| lumana_benchmark | botsort (2022) | YOLOX | gt_vehicles | 5 | **1.37** | 24 | 0.922 | 0.979 | 0.869 | 0.983 | 0.907 | 0.793 | 822 | 430 | **455** | **0** | lumana_benchmark_botsort_gt_fps5_20260817_083521 |

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
