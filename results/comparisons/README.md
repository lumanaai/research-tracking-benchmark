# Tracker comparisons

Side-by-side **GT oracle dets** vs **yolov8m-expert_eff-1_2** (team vehicle detector).

| Source | File |
|---|---|
| GT-only tables | [`gt_vehicles_20260729_085336.md`](gt_vehicles_20260729_085336.md) / [`.csv`](gt_vehicles_20260729_085336.csv) |
| Expert YOLO tables | [`yolov8m_expert_eff_20260729_104145.md`](yolov8m_expert_eff_20260729_104145.md) / [`.csv`](yolov8m_expert_eff_20260729_104145.csv) |

## HOTA / MOTA / IDF1 (GT vs expert YOLO)

| benchmark | tracker | HOTA (GT) | HOTA (YOLO) | MOTA (GT) | MOTA (YOLO) | IDF1 (GT) | IDF1 (YOLO) | AssA (GT) | AssA (YOLO) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fasttracker_bench | fasttracker | 0.971 | 0.479 | 0.982 | 0.351 | 0.986 | 0.493 | 0.974 | 0.749 |
| fasttracker_bench | ocsort | 0.991 | 0.489 | 0.996 | 0.364 | 0.992 | 0.506 | 0.986 | 0.756 |
| fasttracker_bench | hybridsort | 0.991 | 0.487 | 0.996 | 0.364 | 0.992 | 0.504 | 0.986 | 0.750 |
| ua_detrac | fasttracker | 0.880 | 0.465 | 0.942 | -0.183 | 0.971 | 0.557 | 0.900 | 0.624 |
| ua_detrac | ocsort | 0.972 | 0.510 | 0.972 | -0.041 | 0.986 | 0.599 | 0.973 | 0.676 |
| ua_detrac | hybridsort | 0.972 | 0.511 | 0.972 | -0.037 | 0.986 | 0.600 | 0.972 | 0.677 |
| trafficmot | fasttracker | 0.900 | 0.562 | 0.971 | 0.406 | 0.985 | 0.647 | 0.924 | 0.823 |
| trafficmot | ocsort | 0.990 | 0.581 | 0.989 | 0.441 | 0.994 | 0.660 | 0.991 | 0.855 |
| trafficmot | hybridsort | 0.990 | 0.581 | 0.989 | 0.441 | 0.994 | 0.661 | 0.991 | 0.858 |
| cityflow | fasttracker | 0.896 | 0.259 | 0.955 | -3.030 | 0.956 | 0.302 | 0.897 | 0.543 |
| cityflow | ocsort | 0.948 | 0.272 | 0.960 | -2.894 | 0.958 | 0.321 | 0.936 | 0.557 |
| cityflow | hybridsort | 0.942 | 0.273 | 0.959 | -2.886 | 0.951 | 0.322 | 0.924 | 0.563 |

Notes:
- GT = association-only stress test (perfect boxes).
- YOLO = joint detector+tracker score; DetA is limited by the detector.
- Same expert det cache shared across all three trackers per benchmark.
