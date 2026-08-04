# Tracker comparison

_20 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | FPS | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | gt_vehicles | 5 | 12 | 0.164 | 0.161 | 0.168 | 0.161 | 0.283 | 302 | 243 | 0 | 962 | fasttracker_bench_fasttracker_gt_fps5_20260804_155025 |
| fasttracker_bench | ocsort | gt_vehicles | 5 | 12 | 0.168 | 0.166 | 0.170 | 0.166 | 0.283 | 82 | 75 | 0 | 952 | fasttracker_bench_ocsort_gt_fps5_20260804_155025 |
| fasttracker_bench | hybridsort | gt_vehicles | 5 | 12 | 0.168 | 0.166 | 0.169 | 0.166 | 0.282 | **79** | 70 | 0 | 952 | fasttracker_bench_hybridsort_gt_fps5_20260804_155025 |
| fasttracker_bench | analytics_bytetrack | gt_vehicles | 5 | 12 | **0.171** | **0.170** | **0.172** | **0.170** | 0.288 | 152 | 70 | 0 | 949 | fasttracker_bench_analytics_bytetrack_gt_fps5_20260804_155025 |
| fasttracker_bench | botsort | gt_vehicles | 5 | 12 | 0.170 | 0.169 | 0.171 | **0.170** | **0.289** | 725 | **5** | **3** | **940** | fasttracker_bench_botsort_gt_fps5_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker | gt_vehicles | 5 | 60 | 0.140 | 0.134 | 0.149 | 0.088 | 0.239 | 10204 | 1948 | 15 | 5258 | ua_detrac_fasttracker_gt_fps5_20260804_155025 |
| ua_detrac | ocsort | gt_vehicles | 5 | 60 | 0.148 | 0.128 | 0.170 | 0.127 | 0.224 | 515 | 330 | 8 | 5680 | ua_detrac_ocsort_gt_fps5_20260804_155025 |
| ua_detrac | hybridsort | gt_vehicles | 5 | 60 | 0.148 | 0.127 | 0.172 | 0.126 | 0.223 | **485** | 310 | 8 | 5683 | ua_detrac_hybridsort_gt_fps5_20260804_155025 |
| ua_detrac | analytics_bytetrack | gt_vehicles | 5 | 60 | **0.188** | **0.191** | **0.184** | **0.187** | **0.308** | 2493 | 778 | 15 | 5244 | ua_detrac_analytics_bytetrack_gt_fps5_20260804_155025 |
| ua_detrac | botsort | gt_vehicles | 5 | 60 | 0.160 | 0.187 | 0.138 | 0.163 | 0.267 | 26009 | **1** | **31** | **259** | ua_detrac_botsort_gt_fps5_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker | gt_vehicles | 5 | 27 | 0.440 | 0.427 | 0.456 | 0.461 | 0.643 | 43 | 43 | **1** | 13 | trafficmot_fasttracker_gt_fps5_20260804_155025 |
| trafficmot | ocsort | gt_vehicles | 5 | 27 | 0.484 | 0.481 | 0.487 | 0.477 | 0.643 | 95 | 54 | **1** | 26 | trafficmot_ocsort_gt_fps5_20260804_155025 |
| trafficmot | hybridsort | gt_vehicles | 5 | 27 | 0.485 | 0.482 | 0.487 | 0.477 | 0.643 | 96 | 53 | **1** | 26 | trafficmot_hybridsort_gt_fps5_20260804_155025 |
| trafficmot | analytics_bytetrack | gt_vehicles | 5 | 27 | **0.497** | **0.497** | **0.496** | **0.496** | **0.662** | **23** | 25 | **1** | 5 | trafficmot_analytics_bytetrack_gt_fps5_20260804_155025 |
| trafficmot | botsort | gt_vehicles | 5 | 27 | 0.473 | 0.474 | 0.473 | 0.492 | 0.655 | 182 | **19** | **1** | **0** | trafficmot_botsort_gt_fps5_20260804_155025 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker | gt_vehicles | 5 | 36 | 0.431 | 0.427 | 0.436 | 0.449 | 0.618 | 149 | 173 | **0** | 73 | cityflow_fasttracker_gt_fps5_20260804_155025 |
| cityflow | ocsort | gt_vehicles | 5 | 36 | 0.454 | 0.456 | 0.452 | 0.455 | 0.608 | 69 | **132** | **0** | 137 | cityflow_ocsort_gt_fps5_20260804_155025 |
| cityflow | hybridsort | gt_vehicles | 5 | 36 | 0.454 | 0.456 | 0.452 | 0.454 | 0.607 | 72 | 135 | **0** | 137 | cityflow_hybridsort_gt_fps5_20260804_155025 |
| cityflow | analytics_bytetrack | gt_vehicles | 5 | 36 | **0.477** | **0.487** | **0.468** | 0.486 | 0.629 | **62** | 169 | **0** | 40 | cityflow_analytics_bytetrack_gt_fps5_20260804_155025 |
| cityflow | botsort | gt_vehicles | 5 | 36 | 0.467 | 0.476 | 0.459 | **0.492** | **0.639** | 532 | 169 | **0** | **1** | cityflow_botsort_gt_fps5_20260804_155025 |
