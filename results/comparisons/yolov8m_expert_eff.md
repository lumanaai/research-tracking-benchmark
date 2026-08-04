# Tracker comparison

_20 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | FPS | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 12 | 0.534 | 0.414 | 0.710 | 0.463 | 0.602 | 271 | 844 | 259 | 298 | fasttracker_bench_fasttracker_yolov8_20260804_155025 |
| fasttracker_bench | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 12 | 0.547 | 0.425 | 0.720 | 0.482 | 0.617 | 256 | 1421 | 437 | 295 | fasttracker_bench_ocsort_yolov8_20260804_155025 |
| fasttracker_bench | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 12 | 0.546 | 0.425 | 0.718 | 0.482 | 0.615 | 218 | 1390 | 436 | 297 | fasttracker_bench_hybridsort_yolov8_20260804_155025 |
| fasttracker_bench | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 12 | **0.556** | **0.428** | **0.738** | **0.486** | **0.637** | **163** | 1354 | **449** | **284** | fasttracker_bench_analytics_bytetrack_yolov8_20260804_155025 |
| fasttracker_bench | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 30 | 12 | 0.522 | 0.395 | 0.704 | 0.448 | 0.581 | 229 | **526** | 423 | 327 | fasttracker_bench_botsort_yolov8_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 60 | 0.477 | 0.362 | 0.633 | -0.213 | 0.573 | 370 | 2266 | 4234 | 260 | ua_detrac_fasttracker_yolov8_20260804_155025 |
| ua_detrac | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 60 | 0.520 | 0.398 | 0.682 | -0.067 | 0.614 | 432 | 1740 | 5129 | 166 | ua_detrac_ocsort_yolov8_20260804_155025 |
| ua_detrac | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 60 | 0.521 | 0.399 | 0.683 | -0.063 | 0.615 | 371 | 1674 | 5113 | 174 | ua_detrac_hybridsort_yolov8_20260804_155025 |
| ua_detrac | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 60 | 0.514 | 0.392 | 0.676 | -0.111 | 0.609 | **319** | 1958 | **5274** | **136** | ua_detrac_analytics_bytetrack_yolov8_20260804_155025 |
| ua_detrac | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 25 | 60 | **0.536** | **0.422** | **0.685** | **0.065** | **0.642** | 495 | **1073** | 5072 | 195 | ua_detrac_botsort_yolov8_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.562 | 0.396 | 0.823 | 0.406 | 0.647 | 16 | 186 | 315 | 253 | trafficmot_fasttracker_yolov8_20260804_155025 |
| trafficmot | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.581 | 0.407 | 0.855 | 0.441 | 0.660 | 20 | 237 | 332 | 257 | trafficmot_ocsort_yolov8_20260804_155025 |
| trafficmot | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.581 | 0.406 | 0.858 | 0.441 | 0.661 | **15** | 234 | 333 | 259 | trafficmot_hybridsort_yolov8_20260804_155025 |
| trafficmot | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | **0.593** | **0.423** | 0.858 | **0.454** | **0.679** | 41 | 275 | **356** | **226** | trafficmot_analytics_bytetrack_yolov8_20260804_155025 |
| trafficmot | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 27 | 0.557 | 0.372 | **0.859** | 0.408 | 0.624 | 42 | **140** | 313 | 306 | trafficmot_botsort_yolov8_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.259 | 0.131 | 0.543 | -3.030 | 0.302 | **42** | 579 | 473 | 102 | cityflow_fasttracker_yolov8_20260804_155025 |
| cityflow | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.272 | 0.138 | 0.557 | -2.894 | 0.321 | 85 | 628 | 724 | **4** | cityflow_ocsort_yolov8_20260804_155025 |
| cityflow | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.273 | 0.138 | **0.563** | -2.886 | 0.322 | 72 | 624 | 722 | 5 | cityflow_hybridsort_yolov8_20260804_155025 |
| cityflow | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | 0.264 | 0.137 | 0.528 | -2.965 | 0.317 | 49 | 589 | **766** | **4** | cityflow_analytics_bytetrack_yolov8_20260804_155025 |
| cityflow | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 10 | 36 | **0.281** | **0.150** | 0.549 | **-2.376** | **0.347** | 91 | **496** | 736 | 11 | cityflow_botsort_yolov8_20260804_155025 |
