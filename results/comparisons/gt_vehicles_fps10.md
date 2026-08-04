# Tracker comparison

_20 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | FPS | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | gt_vehicles | 10 | 12 | 0.326 | 0.323 | 0.328 | 0.324 | 0.492 | 138 | 326 | 0 | 143 | fasttracker_bench_fasttracker_gt_fps10_20260804_155025 |
| fasttracker_bench | ocsort | gt_vehicles | 10 | 12 | 0.333 | 0.334 | 0.333 | 0.334 | 0.497 | **61** | 57 | 0 | 22 | fasttracker_bench_ocsort_gt_fps10_20260804_155025 |
| fasttracker_bench | hybridsort | gt_vehicles | 10 | 12 | 0.333 | 0.334 | 0.333 | 0.334 | 0.497 | **61** | 57 | 0 | 23 | fasttracker_bench_hybridsort_gt_fps10_20260804_155025 |
| fasttracker_bench | analytics_bytetrack | gt_vehicles | 10 | 12 | **0.336** | **0.337** | **0.335** | 0.337 | 0.500 | 64 | 43 | 0 | 10 | fasttracker_bench_analytics_bytetrack_gt_fps10_20260804_155025 |
| fasttracker_bench | botsort | gt_vehicles | 10 | 12 | 0.335 | 0.336 | 0.334 | **0.338** | **0.501** | 293 | **11** | **5** | **1** | fasttracker_bench_botsort_gt_fps10_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker | gt_vehicles | 10 | 60 | 0.422 | 0.407 | 0.442 | 0.438 | 0.628 | 51 | 1269 | 8 | 251 | ua_detrac_fasttracker_gt_fps10_20260804_155025 |
| ua_detrac | ocsort | gt_vehicles | 10 | 60 | 0.473 | 0.472 | 0.473 | 0.472 | 0.641 | 62 | 65 | 8 | 113 | ua_detrac_ocsort_gt_fps10_20260804_155025 |
| ua_detrac | hybridsort | gt_vehicles | 10 | 60 | 0.473 | 0.472 | 0.473 | 0.472 | 0.641 | 61 | 64 | 8 | 113 | ua_detrac_hybridsort_gt_fps10_20260804_155025 |
| ua_detrac | analytics_bytetrack | gt_vehicles | 10 | 60 | **0.491** | **0.491** | **0.490** | 0.491 | 0.658 | **36** | 3 | 9 | 25 | ua_detrac_analytics_bytetrack_gt_fps10_20260804_155025 |
| ua_detrac | botsort | gt_vehicles | 10 | 60 | 0.478 | 0.472 | 0.484 | **0.500** | **0.667** | 163 | **0** | **15** | **18** | ua_detrac_botsort_gt_fps10_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker | gt_vehicles | 10 | 27 | 0.900 | 0.880 | 0.924 | 0.971 | 0.985 | **5** | 42 | 706 | 1 | trafficmot_fasttracker_gt_fps10_20260804_155025 |
| trafficmot | ocsort | gt_vehicles | 10 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 7 | 32 | 706 | 3 | trafficmot_ocsort_gt_fps10_20260804_155025 |
| trafficmot | hybridsort | gt_vehicles | 10 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 7 | 32 | 706 | 3 | trafficmot_hybridsort_gt_fps10_20260804_155025 |
| trafficmot | analytics_bytetrack | gt_vehicles | 10 | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_gt_fps10_20260804_155025 |
| trafficmot | botsort | gt_vehicles | 10 | 27 | 0.970 | 0.965 | 0.975 | **0.999** | **0.998** | 18 | 35 | **731** | **0** | trafficmot_botsort_gt_fps10_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker | gt_vehicles | 10 | 36 | 0.896 | 0.897 | 0.897 | 0.955 | 0.956 | 60 | 193 | 764 | 1 | cityflow_fasttracker_gt_fps10_20260804_155025 |
| cityflow | ocsort | gt_vehicles | 10 | 36 | 0.948 | 0.961 | 0.936 | 0.960 | 0.958 | 60 | **156** | 654 | 16 | cityflow_ocsort_gt_fps10_20260804_155025 |
| cityflow | hybridsort | gt_vehicles | 10 | 36 | 0.942 | 0.960 | 0.924 | 0.959 | 0.951 | 65 | 163 | 654 | 16 | cityflow_hybridsort_gt_fps10_20260804_155025 |
| cityflow | analytics_bytetrack | gt_vehicles | 10 | 36 | 0.964 | **0.989** | 0.939 | 0.989 | 0.957 | **29** | 186 | 766 | 1 | cityflow_analytics_bytetrack_gt_fps10_20260804_155025 |
| cityflow | botsort | gt_vehicles | 10 | 36 | **0.965** | 0.979 | **0.951** | **0.999** | **0.973** | 85 | 194 | **808** | **0** | cityflow_botsort_gt_fps10_20260804_155025 |
