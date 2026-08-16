# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | FPS | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | 0.189 | 0.143 | 0.257 | 0.150 | 0.267 | 0.953 | 218 | 585 | **0** | 659 | fasttracker_bench_fasttracker_yolov8_fps10_20260816_112714 |
| fasttracker_bench | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | 0.193 | 0.146 | 0.260 | 0.162 | 0.276 | 0.975 | 164 | 808 | **0** | 504 | fasttracker_bench_ocsort_yolov8_fps10_20260816_112714 |
| fasttracker_bench | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | 0.193 | 0.146 | 0.261 | 0.162 | 0.277 | 0.975 | 141 | 790 | **0** | 506 | fasttracker_bench_hybridsort_yolov8_fps10_20260816_112714 |
| fasttracker_bench | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | **0.197** | **0.149** | 0.265 | **0.165** | 0.283 | 0.974 | **122** | 857 | **0** | **472** | fasttracker_bench_analytics_bytetrack_yolov8_fps10_20260816_112714 |
| fasttracker_bench | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | **0.197** | **0.149** | **0.266** | **0.165** | **0.285** | **0.976** | 140 | 874 | **0** | **472** | fasttracker_bench_analytics_bytetrack_plus_yolov8_fps10_20260816_112714 |
| fasttracker_bench | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | 0.184 | 0.137 | 0.253 | 0.152 | 0.256 | 0.958 | 243 | **373** | **0** | 506 | fasttracker_bench_botsort_yolov8_fps10_20260816_112714 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.278 | 0.229 | 0.343 | -0.150 | 0.408 | 0.986 | 397 | 2104 | 9 | 712 | ua_detrac_fasttracker_yolov8_fps10_20260816_112714 |
| ua_detrac | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.310 | 0.263 | 0.368 | -0.028 | 0.452 | 0.987 | 369 | 1018 | 8 | 345 | ua_detrac_ocsort_yolov8_fps10_20260816_112714 |
| ua_detrac | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.311 | 0.263 | 0.368 | -0.026 | 0.452 | 0.987 | 365 | 1022 | 8 | 358 | ua_detrac_hybridsort_yolov8_fps10_20260816_112714 |
| ua_detrac | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.313 | 0.264 | 0.372 | -0.050 | 0.456 | **0.990** | 314 | 1232 | 9 | **241** | ua_detrac_analytics_bytetrack_yolov8_fps10_20260816_112714 |
| ua_detrac | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | **0.315** | 0.265 | **0.376** | -0.049 | 0.460 | **0.990** | **292** | 1227 | 9 | 245 | ua_detrac_analytics_bytetrack_plus_yolov8_fps10_20260816_112714 |
| ua_detrac | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.314 | **0.269** | 0.370 | **0.037** | **0.470** | 0.984 | 611 | **866** | **11** | 352 | ua_detrac_botsort_yolov8_fps10_20260816_112714 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.562 | 0.396 | 0.823 | 0.406 | 0.647 | **0.991** | 16 | 186 | 315 | 253 | trafficmot_fasttracker_yolov8_fps10_20260816_112714 |
| trafficmot | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.581 | 0.407 | 0.855 | 0.441 | 0.660 | 0.990 | 20 | 237 | 332 | 257 | trafficmot_ocsort_yolov8_fps10_20260816_112714 |
| trafficmot | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.581 | 0.406 | 0.858 | 0.441 | 0.661 | 0.990 | **15** | 234 | 333 | 259 | trafficmot_hybridsort_yolov8_fps10_20260816_112714 |
| trafficmot | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | **0.593** | **0.423** | 0.858 | **0.454** | **0.679** | 0.987 | 41 | 275 | **356** | **226** | trafficmot_analytics_bytetrack_yolov8_fps10_20260816_112714 |
| trafficmot | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | **0.593** | **0.423** | 0.858 | **0.454** | **0.679** | 0.987 | 39 | 277 | **356** | **226** | trafficmot_analytics_bytetrack_plus_yolov8_fps10_20260816_112714 |
| trafficmot | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.557 | 0.372 | **0.859** | 0.408 | 0.624 | 0.989 | 42 | **140** | 313 | 306 | trafficmot_botsort_yolov8_fps10_20260816_112714 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.259 | 0.131 | 0.543 | -3.030 | 0.302 | 0.987 | **42** | 579 | 473 | 102 | cityflow_fasttracker_yolov8_fps10_20260816_112714 |
| cityflow | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.272 | 0.138 | 0.557 | -2.894 | 0.321 | 0.982 | 85 | 628 | 724 | **4** | cityflow_ocsort_yolov8_fps10_20260816_112714 |
| cityflow | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.273 | 0.138 | **0.563** | -2.886 | 0.322 | 0.982 | 72 | 624 | 722 | 5 | cityflow_hybridsort_yolov8_fps10_20260816_112714 |
| cityflow | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.264 | 0.137 | 0.528 | -2.965 | 0.317 | **0.990** | 49 | 589 | **766** | **4** | cityflow_analytics_bytetrack_yolov8_fps10_20260816_112714 |
| cityflow | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.266 | 0.137 | 0.539 | -2.962 | 0.317 | 0.985 | 77 | 614 | 751 | **4** | cityflow_analytics_bytetrack_plus_yolov8_fps10_20260816_112714 |
| cityflow | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | **0.281** | **0.150** | 0.549 | **-2.376** | **0.347** | 0.983 | 91 | **496** | 736 | 11 | cityflow_botsort_yolov8_fps10_20260816_112714 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 24 | 0.470 | 0.433 | 0.514 | 0.454 | 0.579 | 0.899 | 258 | 616 | 110 | 84 | lumana_benchmark_fasttracker_yolov8_fps10_20260816_112714 |
| lumana_benchmark | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 24 | 0.479 | **0.445** | 0.519 | **0.465** | 0.596 | 0.917 | 196 | 805 | 107 | 81 | lumana_benchmark_ocsort_yolov8_fps10_20260816_112714 |
| lumana_benchmark | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 24 | 0.481 | 0.444 | 0.525 | **0.465** | 0.597 | 0.924 | **189** | 788 | 104 | 84 | lumana_benchmark_hybridsort_yolov8_fps10_20260816_112714 |
| lumana_benchmark | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 24 | 0.481 | 0.444 | 0.524 | 0.460 | 0.593 | 0.922 | 230 | 862 | **126** | **67** | lumana_benchmark_analytics_bytetrack_yolov8_fps10_20260816_112714 |
| lumana_benchmark | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 24 | **0.496** | 0.442 | **0.560** | 0.458 | **0.617** | **0.939** | 236 | 877 | 124 | **67** | lumana_benchmark_analytics_bytetrack_plus_yolov8_fps10_20260816_112714 |
| lumana_benchmark | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 24 | 0.457 | 0.411 | 0.511 | 0.436 | 0.554 | 0.897 | 234 | **593** | 112 | 110 | lumana_benchmark_botsort_yolov8_fps10_20260816_112714 |
