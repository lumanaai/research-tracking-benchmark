# Tracker comparison

_20 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | FPS | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | 0.189 | 0.143 | 0.257 | 0.150 | 0.267 | 218 | 585 | **0** | 659 | fasttracker_bench_fasttracker_yolov8_fps10_20260804_155025 |
| fasttracker_bench | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | 0.193 | 0.146 | 0.260 | 0.162 | 0.276 | 164 | 808 | **0** | 504 | fasttracker_bench_ocsort_yolov8_fps10_20260804_155025 |
| fasttracker_bench | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | 0.193 | 0.146 | 0.261 | 0.162 | 0.277 | 141 | 790 | **0** | 506 | fasttracker_bench_hybridsort_yolov8_fps10_20260804_155025 |
| fasttracker_bench | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | **0.197** | **0.149** | **0.265** | **0.165** | **0.283** | **122** | 857 | **0** | **472** | fasttracker_bench_analytics_bytetrack_yolov8_fps10_20260804_155025 |
| fasttracker_bench | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 12 | 0.184 | 0.137 | 0.253 | 0.152 | 0.256 | 243 | **373** | **0** | 506 | fasttracker_bench_botsort_yolov8_fps10_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.278 | 0.229 | 0.343 | -0.150 | 0.408 | 397 | 2104 | 9 | 712 | ua_detrac_fasttracker_yolov8_fps10_20260804_155025 |
| ua_detrac | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.310 | 0.263 | 0.368 | -0.028 | 0.452 | 369 | 1018 | 8 | 345 | ua_detrac_ocsort_yolov8_fps10_20260804_155025 |
| ua_detrac | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.311 | 0.263 | 0.368 | -0.026 | 0.452 | 365 | 1022 | 8 | 358 | ua_detrac_hybridsort_yolov8_fps10_20260804_155025 |
| ua_detrac | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | 0.313 | 0.264 | **0.372** | -0.050 | 0.456 | **314** | 1232 | 9 | **241** | ua_detrac_analytics_bytetrack_yolov8_fps10_20260804_155025 |
| ua_detrac | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 60 | **0.314** | **0.269** | 0.370 | **0.037** | **0.470** | 611 | **866** | **11** | 352 | ua_detrac_botsort_yolov8_fps10_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.562 | 0.396 | 0.823 | 0.406 | 0.647 | 16 | 186 | 315 | 253 | trafficmot_fasttracker_yolov8_fps10_20260804_155025 |
| trafficmot | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.581 | 0.407 | 0.855 | 0.441 | 0.660 | 20 | 237 | 332 | 257 | trafficmot_ocsort_yolov8_fps10_20260804_155025 |
| trafficmot | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.581 | 0.406 | 0.858 | 0.441 | 0.661 | **15** | 234 | 333 | 259 | trafficmot_hybridsort_yolov8_fps10_20260804_155025 |
| trafficmot | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | **0.593** | **0.423** | 0.858 | **0.454** | **0.679** | 41 | 275 | **356** | **226** | trafficmot_analytics_bytetrack_yolov8_fps10_20260804_155025 |
| trafficmot | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.557 | 0.372 | **0.859** | 0.408 | 0.624 | 42 | **140** | 313 | 306 | trafficmot_botsort_yolov8_fps10_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.259 | 0.131 | 0.543 | -3.030 | 0.302 | **42** | 579 | 473 | 102 | cityflow_fasttracker_yolov8_fps10_20260804_155025 |
| cityflow | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.272 | 0.138 | 0.557 | -2.894 | 0.321 | 85 | 628 | 724 | **4** | cityflow_ocsort_yolov8_fps10_20260804_155025 |
| cityflow | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.273 | 0.138 | **0.563** | -2.886 | 0.322 | 72 | 624 | 722 | 5 | cityflow_hybridsort_yolov8_fps10_20260804_155025 |
| cityflow | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.264 | 0.137 | 0.528 | -2.965 | 0.317 | 49 | 589 | **766** | **4** | cityflow_analytics_bytetrack_yolov8_fps10_20260804_155025 |
| cityflow | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | **0.281** | **0.150** | 0.549 | **-2.376** | **0.347** | 91 | **496** | 736 | 11 | cityflow_botsort_yolov8_fps10_20260804_155025 |
