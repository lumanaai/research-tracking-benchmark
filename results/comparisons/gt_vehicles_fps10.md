# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | FPS | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | gt_vehicles | 10 | 12 | 0.326 | 0.323 | 0.328 | 0.324 | 0.492 | 0.981 | 138 | 326 | 0 | 143 | fasttracker_bench_fasttracker_gt_fps10_20260816_112714 |
| fasttracker_bench | ocsort | gt_vehicles | 10 | 12 | 0.333 | 0.334 | 0.333 | 0.334 | 0.497 | 0.991 | **61** | 57 | 0 | 22 | fasttracker_bench_ocsort_gt_fps10_20260816_112714 |
| fasttracker_bench | hybridsort | gt_vehicles | 10 | 12 | 0.333 | 0.334 | 0.333 | 0.334 | 0.497 | 0.991 | **61** | 57 | 0 | 23 | fasttracker_bench_hybridsort_gt_fps10_20260816_112714 |
| fasttracker_bench | analytics_bytetrack | gt_vehicles | 10 | 12 | **0.336** | **0.337** | **0.335** | 0.337 | 0.500 | **0.992** | 64 | 43 | 0 | 10 | fasttracker_bench_analytics_bytetrack_gt_fps10_20260816_112714 |
| fasttracker_bench | analytics_bytetrack_plus | gt_vehicles | 10 | 12 | **0.336** | **0.337** | 0.334 | 0.337 | 0.500 | 0.990 | 89 | 70 | 0 | 10 | fasttracker_bench_analytics_bytetrack_plus_gt_fps10_20260816_112714 |
| fasttracker_bench | botsort | gt_vehicles | 10 | 12 | 0.335 | 0.336 | 0.334 | **0.338** | **0.501** | 0.981 | 293 | **11** | **5** | **1** | fasttracker_bench_botsort_gt_fps10_20260816_112714 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker | gt_vehicles | 10 | 60 | 0.422 | 0.407 | 0.442 | 0.438 | 0.628 | 0.998 | 51 | 1269 | 8 | 251 | ua_detrac_fasttracker_gt_fps10_20260816_112714 |
| ua_detrac | ocsort | gt_vehicles | 10 | 60 | 0.473 | 0.472 | 0.473 | 0.472 | 0.641 | **0.999** | 62 | 65 | 8 | 113 | ua_detrac_ocsort_gt_fps10_20260816_112714 |
| ua_detrac | hybridsort | gt_vehicles | 10 | 60 | 0.473 | 0.472 | 0.473 | 0.472 | 0.641 | **0.999** | 61 | 64 | 8 | 113 | ua_detrac_hybridsort_gt_fps10_20260816_112714 |
| ua_detrac | analytics_bytetrack | gt_vehicles | 10 | 60 | **0.491** | **0.491** | **0.490** | 0.491 | 0.658 | **0.999** | **36** | 3 | 9 | 25 | ua_detrac_analytics_bytetrack_gt_fps10_20260816_112714 |
| ua_detrac | analytics_bytetrack_plus | gt_vehicles | 10 | 60 | 0.490 | **0.491** | **0.490** | 0.491 | 0.657 | **0.999** | 43 | 17 | 8 | 26 | ua_detrac_analytics_bytetrack_plus_gt_fps10_20260816_112714 |
| ua_detrac | botsort | gt_vehicles | 10 | 60 | 0.478 | 0.472 | 0.484 | **0.500** | **0.667** | **0.999** | 163 | **0** | **15** | **18** | ua_detrac_botsort_gt_fps10_20260816_112714 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker | gt_vehicles | 10 | 27 | 0.900 | 0.880 | 0.924 | 0.971 | 0.985 | **1.000** | **5** | 42 | 706 | 1 | trafficmot_fasttracker_gt_fps10_20260816_112714 |
| trafficmot | ocsort | gt_vehicles | 10 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 0.999 | 7 | 32 | 706 | 3 | trafficmot_ocsort_gt_fps10_20260816_112714 |
| trafficmot | hybridsort | gt_vehicles | 10 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 0.999 | 7 | 32 | 706 | 3 | trafficmot_hybridsort_gt_fps10_20260816_112714 |
| trafficmot | analytics_bytetrack | gt_vehicles | 10 | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 0.999 | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_gt_fps10_20260816_112714 |
| trafficmot | analytics_bytetrack_plus | gt_vehicles | 10 | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 0.999 | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_plus_gt_fps10_20260816_112714 |
| trafficmot | botsort | gt_vehicles | 10 | 27 | 0.970 | 0.965 | 0.975 | **0.999** | **0.998** | 0.997 | 18 | 35 | **731** | **0** | trafficmot_botsort_gt_fps10_20260816_112714 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker | gt_vehicles | 10 | 36 | 0.896 | 0.897 | 0.897 | 0.955 | 0.956 | 0.985 | 60 | 193 | 764 | 1 | cityflow_fasttracker_gt_fps10_20260816_112714 |
| cityflow | ocsort | gt_vehicles | 10 | 36 | 0.948 | 0.961 | 0.936 | 0.960 | 0.958 | 0.982 | 60 | **156** | 654 | 16 | cityflow_ocsort_gt_fps10_20260816_112714 |
| cityflow | hybridsort | gt_vehicles | 10 | 36 | 0.942 | 0.960 | 0.924 | 0.959 | 0.951 | 0.981 | 65 | 163 | 654 | 16 | cityflow_hybridsort_gt_fps10_20260816_112714 |
| cityflow | analytics_bytetrack | gt_vehicles | 10 | 36 | 0.964 | **0.989** | 0.939 | 0.989 | 0.957 | **0.990** | **29** | 186 | 766 | 1 | cityflow_analytics_bytetrack_gt_fps10_20260816_112714 |
| cityflow | analytics_bytetrack_plus | gt_vehicles | 10 | 36 | 0.963 | 0.988 | 0.939 | 0.987 | 0.960 | 0.987 | 51 | 215 | 764 | 1 | cityflow_analytics_bytetrack_plus_gt_fps10_20260816_112714 |
| cityflow | botsort | gt_vehicles | 10 | 36 | **0.965** | 0.979 | **0.951** | **0.999** | **0.973** | 0.981 | 85 | 194 | **808** | **0** | cityflow_botsort_gt_fps10_20260816_112714 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker | gt_vehicles | 10 | 24 | 0.548 | 0.545 | 0.552 | 0.554 | 0.652 | 0.908 | 276 | 540 | 140 | 17 | lumana_benchmark_fasttracker_gt_fps10_20260816_112714 |
| lumana_benchmark | ocsort | gt_vehicles | 10 | 24 | 0.567 | 0.563 | 0.571 | 0.562 | 0.666 | 0.915 | **231** | **487** | 103 | 49 | lumana_benchmark_ocsort_gt_fps10_20260816_112714 |
| lumana_benchmark | hybridsort | gt_vehicles | 10 | 24 | 0.562 | 0.562 | 0.562 | 0.561 | 0.656 | 0.909 | 245 | 501 | 103 | 50 | lumana_benchmark_hybridsort_gt_fps10_20260816_112714 |
| lumana_benchmark | analytics_bytetrack | gt_vehicles | 10 | 24 | 0.571 | **0.571** | 0.571 | 0.570 | 0.668 | 0.921 | 238 | 524 | 154 | 9 | lumana_benchmark_analytics_bytetrack_gt_fps10_20260816_112714 |
| lumana_benchmark | analytics_bytetrack_plus | gt_vehicles | 10 | 24 | **0.587** | 0.568 | **0.606** | 0.567 | **0.691** | **0.931** | 246 | 570 | 146 | 12 | lumana_benchmark_analytics_bytetrack_plus_gt_fps10_20260816_112714 |
| lumana_benchmark | botsort | gt_vehicles | 10 | 24 | 0.560 | 0.567 | 0.554 | **0.571** | 0.653 | 0.897 | 334 | 523 | **177** | **3** | lumana_benchmark_botsort_gt_fps10_20260816_112714 |
