# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | pref_det | detector_id | FPS | ms/frame | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker (2025) | YOLOX | gt_vehicles | 30 | **3.94** | 12 | 0.971 | 0.969 | 0.974 | 0.982 | 0.986 | 0.991 | 133 | 358 | 848 | 8 | fasttracker_bench_fasttracker_gt_20260817_083521 |
| fasttracker_bench | ocsort (2023) | YOLOX | gt_vehicles | 30 | 4.46 | 12 | 0.991 | 0.996 | 0.986 | 0.996 | 0.992 | **0.992** | 58 | 52 | 1009 | 7 | fasttracker_bench_ocsort_gt_20260817_083521 |
| fasttracker_bench | hybridsort (2024) | YOLOX | gt_vehicles | 30 | 7.97 | 12 | 0.991 | 0.996 | 0.986 | 0.996 | 0.992 | **0.992** | 70 | 69 | 1009 | 7 | fasttracker_bench_hybridsort_gt_20260817_083521 |
| fasttracker_bench | analytics_bytetrack (2022) | YOLO | gt_vehicles | 30 | 4.64 | 12 | 0.992 | **0.999** | 0.985 | 0.999 | 0.992 | **0.992** | **49** | 35 | 1013 | 6 | fasttracker_bench_analytics_bytetrack_gt_20260817_083521 |
| fasttracker_bench | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 30 | 8.39 | 12 | 0.990 | 0.998 | 0.981 | 0.998 | 0.990 | 0.988 | 107 | 103 | 1013 | 6 | fasttracker_bench_analytics_bytetrack_plus_gt_20260817_083521 |
| fasttracker_bench | botsort (2022) | YOLOX | gt_vehicles | 30 | 4.02 | 12 | **0.994** | 0.998 | **0.989** | **1.000** | **0.995** | **0.992** | 164 | **13** | **1021** | **0** | fasttracker_bench_botsort_gt_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker (2025) | YOLOX | gt_vehicles | 25 | 1.39 | 60 | 0.880 | 0.863 | 0.900 | 0.942 | 0.971 | 0.999 | 25 | 855 | 5605 | 38 | ua_detrac_fasttracker_gt_20260817_083521 |
| ua_detrac | ocsort (2023) | YOLOX | gt_vehicles | 25 | 1.63 | 60 | 0.972 | 0.972 | 0.973 | 0.972 | 0.986 | **1.000** | **11** | 17 | 5798 | 27 | ua_detrac_ocsort_gt_20260817_083521 |
| ua_detrac | hybridsort (2024) | YOLOX | gt_vehicles | 25 | 2.47 | 60 | 0.972 | 0.972 | 0.972 | 0.972 | 0.986 | **1.000** | 12 | 17 | 5798 | 26 | ua_detrac_hybridsort_gt_20260817_083521 |
| ua_detrac | analytics_bytetrack (2022) | YOLO | gt_vehicles | 25 | 2.30 | 60 | 0.988 | **0.991** | 0.985 | 0.991 | 0.992 | 0.999 | 27 | **1** | 5918 | 21 | ua_detrac_analytics_bytetrack_gt_20260817_083521 |
| ua_detrac | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 25 | 3.76 | 60 | 0.988 | **0.991** | 0.985 | 0.991 | 0.992 | 0.999 | 86 | 33 | 5914 | 25 | ua_detrac_analytics_bytetrack_plus_gt_20260817_083521 |
| ua_detrac | botsort (2022) | YOLOX | gt_vehicles | 25 | **1.24** | 60 | **0.989** | 0.987 | **0.991** | **1.000** | **1.000** | **1.000** | 16 | **1** | **5952** | **0** | ua_detrac_botsort_gt_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker (2025) | YOLOX | gt_vehicles | 10 | 7.10 | 27 | 0.900 | 0.880 | 0.924 | 0.971 | 0.985 | **1.000** | **5** | 42 | 706 | 1 | trafficmot_fasttracker_gt_20260817_083521 |
| trafficmot | ocsort (2023) | YOLOX | gt_vehicles | 10 | 5.05 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 0.999 | 7 | 32 | 706 | 3 | trafficmot_ocsort_gt_20260817_083521 |
| trafficmot | hybridsort (2024) | YOLOX | gt_vehicles | 10 | 7.15 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 0.999 | 7 | 32 | 706 | 3 | trafficmot_hybridsort_gt_20260817_083521 |
| trafficmot | analytics_bytetrack (2022) | YOLO | gt_vehicles | 10 | **3.69** | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 0.999 | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_gt_20260817_083521 |
| trafficmot | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 10 | 6.00 | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 0.999 | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_plus_gt_20260817_083521 |
| trafficmot | botsort (2022) | YOLOX | gt_vehicles | 10 | 4.26 | 27 | 0.970 | 0.965 | 0.975 | **0.999** | **0.998** | 0.997 | 18 | 35 | **731** | **0** | trafficmot_botsort_gt_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker (2025) | YOLOX | gt_vehicles | 10 | 0.62 | 36 | 0.896 | 0.897 | 0.897 | 0.955 | 0.956 | 0.985 | 60 | 193 | 764 | 1 | cityflow_fasttracker_gt_20260817_083521 |
| cityflow | ocsort (2023) | YOLOX | gt_vehicles | 10 | 0.82 | 36 | 0.948 | 0.961 | 0.936 | 0.960 | 0.958 | 0.982 | 60 | **156** | 654 | 16 | cityflow_ocsort_gt_20260817_083521 |
| cityflow | hybridsort (2024) | YOLOX | gt_vehicles | 10 | 1.00 | 36 | 0.942 | 0.960 | 0.924 | 0.959 | 0.951 | 0.981 | 65 | 163 | 654 | 16 | cityflow_hybridsort_gt_20260817_083521 |
| cityflow | analytics_bytetrack (2022) | YOLO | gt_vehicles | 10 | 0.74 | 36 | 0.964 | **0.989** | 0.939 | 0.989 | 0.957 | **0.990** | **29** | 186 | 766 | 1 | cityflow_analytics_bytetrack_gt_20260817_083521 |
| cityflow | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 10 | 1.05 | 36 | 0.963 | 0.988 | 0.939 | 0.987 | 0.960 | 0.987 | 51 | 215 | 764 | 1 | cityflow_analytics_bytetrack_plus_gt_20260817_083521 |
| cityflow | botsort (2022) | YOLOX | gt_vehicles | 10 | **0.56** | 36 | **0.965** | 0.979 | **0.951** | **0.999** | **0.973** | 0.981 | 85 | 194 | **808** | **0** | cityflow_botsort_gt_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker (2025) | YOLOX | gt_vehicles | 20 | **1.23** | 24 | 0.903 | 0.954 | 0.854 | 0.979 | 0.904 | 0.911 | 277 | 578 | 404 | 7 | lumana_benchmark_fasttracker_gt_20260817_083521 |
| lumana_benchmark | ocsort (2023) | YOLOX | gt_vehicles | 20 | 1.64 | 24 | 0.919 | 0.989 | 0.855 | 0.988 | 0.899 | 0.896 | 310 | **555** | 346 | 25 | lumana_benchmark_ocsort_gt_20260817_083521 |
| lumana_benchmark | hybridsort (2024) | YOLOX | gt_vehicles | 20 | 2.42 | 24 | 0.918 | 0.989 | 0.852 | 0.988 | 0.896 | 0.897 | 308 | 560 | 346 | 25 | lumana_benchmark_hybridsort_gt_20260817_083521 |
| lumana_benchmark | analytics_bytetrack (2022) | YOLO | gt_vehicles | 20 | 1.37 | 24 | 0.932 | **0.997** | 0.871 | 0.997 | 0.916 | 0.923 | **238** | 558 | 435 | 6 | lumana_benchmark_analytics_bytetrack_gt_20260817_083521 |
| lumana_benchmark | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 20 | 2.36 | 24 | **0.952** | 0.995 | **0.912** | 0.994 | **0.940** | **0.928** | 304 | 627 | 425 | 8 | lumana_benchmark_analytics_bytetrack_plus_gt_20260817_083521 |
| lumana_benchmark | botsort (2022) | YOLOX | gt_vehicles | 20 | 1.34 | 24 | 0.913 | 0.993 | 0.840 | **0.998** | 0.889 | 0.893 | 348 | 558 | **464** | **0** | lumana_benchmark_botsort_gt_20260817_083521 |

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
