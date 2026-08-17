# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | pref_det | detector_id | FPS | ms/frame | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker (2025) | YOLOX | gt_vehicles | 10 | 4.91 | 12 | 0.945 | 0.935 | 0.956 | 0.958 | 0.972 | 0.981 | 138 | 326 | 770 | 32 | fasttracker_bench_fasttracker_gt_fps10_20260817_083521 |
| fasttracker_bench | ocsort (2023) | YOLOX | gt_vehicles | 10 | 5.16 | 12 | 0.982 | 0.987 | 0.976 | 0.987 | 0.986 | 0.991 | **61** | 57 | 848 | 9 | fasttracker_bench_ocsort_gt_fps10_20260817_083521 |
| fasttracker_bench | hybridsort (2024) | YOLOX | gt_vehicles | 10 | 8.20 | 12 | 0.982 | 0.987 | 0.976 | 0.987 | 0.986 | 0.991 | **61** | 57 | 851 | 9 | fasttracker_bench_hybridsort_gt_fps10_20260817_083521 |
| fasttracker_bench | analytics_bytetrack (2022) | YOLO | gt_vehicles | 10 | 4.70 | 12 | **0.990** | **0.996** | **0.984** | 0.996 | **0.991** | **0.992** | 64 | 43 | 1008 | 9 | fasttracker_bench_analytics_bytetrack_gt_fps10_20260817_083521 |
| fasttracker_bench | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 10 | 8.85 | 12 | 0.988 | **0.996** | 0.980 | 0.995 | 0.990 | 0.990 | 89 | 70 | 1008 | 9 | fasttracker_bench_analytics_bytetrack_plus_gt_fps10_20260817_083521 |
| fasttracker_bench | botsort (2022) | YOLOX | gt_vehicles | 10 | **3.89** | 12 | 0.985 | 0.990 | 0.981 | **0.999** | **0.991** | 0.981 | 293 | **11** | **1020** | **0** | fasttracker_bench_botsort_gt_fps10_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker (2025) | YOLOX | gt_vehicles | 10 | 1.32 | 60 | 0.811 | 0.776 | 0.857 | 0.875 | 0.938 | 0.998 | 51 | 1269 | 5033 | 56 | ua_detrac_fasttracker_gt_fps10_20260817_083521 |
| ua_detrac | ocsort (2023) | YOLOX | gt_vehicles | 10 | 1.78 | 60 | 0.944 | 0.943 | 0.945 | 0.943 | 0.970 | **0.999** | 62 | 65 | 5277 | 70 | ua_detrac_ocsort_gt_fps10_20260817_083521 |
| ua_detrac | hybridsort (2024) | YOLOX | gt_vehicles | 10 | 2.69 | 60 | 0.944 | 0.943 | 0.945 | 0.943 | 0.970 | **0.999** | 61 | 64 | 5277 | 70 | ua_detrac_hybridsort_gt_fps10_20260817_083521 |
| ua_detrac | analytics_bytetrack (2022) | YOLO | gt_vehicles | 10 | 2.37 | 60 | **0.980** | **0.982** | **0.979** | 0.981 | 0.989 | **0.999** | **36** | 3 | 5835 | 7 | ua_detrac_analytics_bytetrack_gt_fps10_20260817_083521 |
| ua_detrac | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 10 | 3.75 | 60 | 0.979 | 0.981 | 0.977 | 0.981 | 0.988 | **0.999** | 43 | 17 | 5828 | 8 | ua_detrac_analytics_bytetrack_plus_gt_fps10_20260817_083521 |
| ua_detrac | botsort (2022) | YOLOX | gt_vehicles | 10 | **1.30** | 60 | 0.947 | 0.935 | 0.963 | **0.999** | **0.999** | **0.999** | 163 | **0** | **5934** | **0** | ua_detrac_botsort_gt_fps10_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker (2025) | YOLOX | gt_vehicles | 10 | 7.28 | 27 | 0.900 | 0.880 | 0.924 | 0.971 | 0.985 | **1.000** | **5** | 42 | 706 | 1 | trafficmot_fasttracker_gt_fps10_20260817_083521 |
| trafficmot | ocsort (2023) | YOLOX | gt_vehicles | 10 | 4.45 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 0.999 | 7 | 32 | 706 | 3 | trafficmot_ocsort_gt_fps10_20260817_083521 |
| trafficmot | hybridsort (2024) | YOLOX | gt_vehicles | 10 | 7.25 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 0.999 | 7 | 32 | 706 | 3 | trafficmot_hybridsort_gt_fps10_20260817_083521 |
| trafficmot | analytics_bytetrack (2022) | YOLO | gt_vehicles | 10 | **3.49** | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 0.999 | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_gt_fps10_20260817_083521 |
| trafficmot | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 10 | 5.08 | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 0.999 | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_plus_gt_fps10_20260817_083521 |
| trafficmot | botsort (2022) | YOLOX | gt_vehicles | 10 | 3.51 | 27 | 0.970 | 0.965 | 0.975 | **0.999** | **0.998** | 0.997 | 18 | 35 | **731** | **0** | trafficmot_botsort_gt_fps10_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker (2025) | YOLOX | gt_vehicles | 10 | 0.57 | 36 | 0.896 | 0.897 | 0.897 | 0.955 | 0.956 | 0.985 | 60 | 193 | 764 | 1 | cityflow_fasttracker_gt_fps10_20260817_083521 |
| cityflow | ocsort (2023) | YOLOX | gt_vehicles | 10 | 0.77 | 36 | 0.948 | 0.961 | 0.936 | 0.960 | 0.958 | 0.982 | 60 | **156** | 654 | 16 | cityflow_ocsort_gt_fps10_20260817_083521 |
| cityflow | hybridsort (2024) | YOLOX | gt_vehicles | 10 | 1.07 | 36 | 0.942 | 0.960 | 0.924 | 0.959 | 0.951 | 0.981 | 65 | 163 | 654 | 16 | cityflow_hybridsort_gt_fps10_20260817_083521 |
| cityflow | analytics_bytetrack (2022) | YOLO | gt_vehicles | 10 | 0.72 | 36 | 0.964 | **0.989** | 0.939 | 0.989 | 0.957 | **0.990** | **29** | 186 | 766 | 1 | cityflow_analytics_bytetrack_gt_fps10_20260817_083521 |
| cityflow | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 10 | 1.08 | 36 | 0.963 | 0.988 | 0.939 | 0.987 | 0.960 | 0.987 | 51 | 215 | 764 | 1 | cityflow_analytics_bytetrack_plus_gt_fps10_20260817_083521 |
| cityflow | botsort (2022) | YOLOX | gt_vehicles | 10 | **0.56** | 36 | **0.965** | 0.979 | **0.951** | **0.999** | **0.973** | 0.981 | 85 | 194 | **808** | **0** | cityflow_botsort_gt_fps10_20260817_083521 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker (2025) | YOLOX | gt_vehicles | 10 | 1.44 | 24 | 0.888 | 0.935 | 0.844 | 0.968 | 0.893 | 0.908 | 276 | 540 | 394 | 12 | lumana_benchmark_fasttracker_gt_fps10_20260817_083521 |
| lumana_benchmark | ocsort (2023) | YOLOX | gt_vehicles | 10 | 1.68 | 24 | 0.930 | 0.982 | 0.881 | 0.980 | 0.916 | 0.915 | **231** | **487** | 323 | 38 | lumana_benchmark_ocsort_gt_fps10_20260817_083521 |
| lumana_benchmark | hybridsort (2024) | YOLOX | gt_vehicles | 10 | 2.49 | 24 | 0.921 | 0.982 | 0.865 | 0.980 | 0.902 | 0.909 | 245 | 501 | 325 | 37 | lumana_benchmark_hybridsort_gt_fps10_20260817_083521 |
| lumana_benchmark | analytics_bytetrack (2022) | YOLO | gt_vehicles | 10 | 1.43 | 24 | 0.935 | **0.997** | 0.876 | 0.995 | 0.918 | 0.921 | 238 | 524 | 429 | 6 | lumana_benchmark_analytics_bytetrack_gt_fps10_20260817_083521 |
| lumana_benchmark | analytics_bytetrack_plus (2022) | YOLO | gt_vehicles | 10 | 2.34 | 24 | **0.957** | 0.991 | **0.924** | 0.990 | **0.949** | **0.931** | 246 | 570 | 421 | 8 | lumana_benchmark_analytics_bytetrack_plus_gt_fps10_20260817_083521 |
| lumana_benchmark | botsort (2022) | YOLOX | gt_vehicles | 10 | **1.22** | 24 | 0.918 | 0.988 | 0.852 | **0.997** | 0.897 | 0.897 | 334 | 523 | **461** | **0** | lumana_benchmark_botsort_gt_fps10_20260817_083521 |

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
