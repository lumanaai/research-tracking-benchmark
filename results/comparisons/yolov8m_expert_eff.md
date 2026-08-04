# Tracker comparison

_16 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 12 | 0.479 | 0.316 | 0.749 | 0.351 | 0.493 | 186 | 602 | 189 | 551 | fasttracker_bench_fasttracker_yolov8_20260729_104145 |
| fasttracker_bench | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 12 | 0.489 | 0.324 | 0.756 | 0.364 | 0.506 | 167 | 979 | 310 | 547 | fasttracker_bench_ocsort_yolov8_20260729_104145 |
| fasttracker_bench | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 12 | 0.487 | 0.324 | 0.750 | 0.364 | 0.504 | 148 | 951 | 310 | 549 | fasttracker_bench_hybridsort_yolov8_20260729_104145 |
| fasttracker_bench | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 1 | 0.474 | 0.405 | 0.562 | 0.482 | 0.616 | 14 | 244 | 85 | 35 | fasttracker_bench_analytics_bytetrack_yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles_benchmark_20260804_133216 |
| ua_detrac | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 60 | 0.465 | 0.349 | 0.624 | -0.183 | 0.557 | 363 | 2223 | 4006 | 504 | ua_detrac_fasttracker_yolov8_20260729_104145 |
| ua_detrac | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 60 | 0.510 | 0.386 | 0.676 | -0.041 | 0.599 | 410 | 1632 | 4893 | 410 | ua_detrac_ocsort_yolov8_20260729_104145 |
| ua_detrac | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 60 | 0.511 | 0.386 | 0.677 | -0.037 | 0.600 | 355 | 1574 | 4877 | 418 | ua_detrac_hybridsort_yolov8_20260729_104145 |
| ua_detrac | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 60 | 0.514 | 0.392 | 0.676 | -0.111 | 0.609 | 319 | 1958 | 5274 | 136 | ua_detrac_analytics_bytetrack_yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles_benchmark_20260804_140454 |
| trafficmot | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 27 | 0.562 | 0.396 | 0.823 | 0.406 | 0.647 | 16 | 186 | 315 | 253 | trafficmot_fasttracker_yolov8_20260729_104145 |
| trafficmot | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 27 | 0.581 | 0.407 | 0.855 | 0.441 | 0.660 | 20 | 237 | 332 | 257 | trafficmot_ocsort_yolov8_20260729_104145 |
| trafficmot | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 27 | 0.581 | 0.406 | 0.858 | 0.441 | 0.661 | 15 | 234 | 333 | 259 | trafficmot_hybridsort_yolov8_20260729_104145 |
| trafficmot | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 27 | 0.593 | 0.423 | 0.858 | 0.454 | 0.679 | 41 | 275 | 356 | 226 | trafficmot_analytics_bytetrack_yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles_benchmark_20260804_125854 |
| cityflow | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 36 | 0.259 | 0.131 | 0.543 | -3.030 | 0.302 | 42 | 579 | 473 | 102 | cityflow_fasttracker_yolov8_20260729_104145 |
| cityflow | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 36 | 0.272 | 0.138 | 0.557 | -2.894 | 0.321 | 85 | 628 | 724 | 4 | cityflow_ocsort_yolov8_20260729_104145 |
| cityflow | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 36 | 0.273 | 0.138 | 0.563 | -2.886 | 0.322 | 72 | 624 | 722 | 5 | cityflow_hybridsort_yolov8_20260729_104145 |
| cityflow | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 36 | 0.264 | 0.137 | 0.528 | -2.965 | 0.317 | 49 | 589 | 766 | 4 | cityflow_analytics_bytetrack_yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles_benchmark_20260804_140531 |
