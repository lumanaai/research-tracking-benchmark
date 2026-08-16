# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | FPS | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 12 | 0.098 | 0.073 | 0.135 | 0.074 | 0.146 | 0.917 | 311 | 409 | **0** | 986 | fasttracker_bench_fasttracker_yolov8_fps5_20260816_111023 |
| fasttracker_bench | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 12 | 0.099 | 0.073 | 0.136 | 0.080 | 0.149 | **0.973** | 124 | 548 | **0** | 975 | fasttracker_bench_ocsort_yolov8_fps5_20260816_111023 |
| fasttracker_bench | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 12 | 0.098 | 0.073 | 0.136 | 0.080 | 0.148 | 0.969 | **123** | 541 | **0** | 975 | fasttracker_bench_hybridsort_yolov8_fps5_20260816_111023 |
| fasttracker_bench | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 12 | **0.102** | **0.076** | **0.139** | **0.083** | 0.155 | 0.954 | 171 | 640 | **0** | **972** | fasttracker_bench_analytics_bytetrack_yolov8_fps5_20260816_111023 |
| fasttracker_bench | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 12 | **0.102** | **0.076** | **0.139** | **0.083** | **0.156** | 0.971 | 144 | 634 | **0** | **972** | fasttracker_bench_analytics_bytetrack_plus_yolov8_fps5_20260816_111023 |
| fasttracker_bench | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 12 | 0.094 | 0.069 | 0.132 | 0.076 | 0.139 | 0.888 | 589 | **338** | **0** | 973 | fasttracker_bench_botsort_yolov8_fps5_20260816_111023 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 60 | 0.108 | 0.095 | 0.125 | -0.124 | 0.178 | 0.775 | 9144 | 2062 | 8 | 5301 | ua_detrac_fasttracker_yolov8_fps5_20260816_111023 |
| ua_detrac | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 60 | 0.113 | 0.090 | 0.142 | -0.049 | 0.181 | 0.964 | 560 | 586 | 8 | 5602 | ua_detrac_ocsort_yolov8_fps5_20260816_111023 |
| ua_detrac | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 60 | 0.114 | 0.090 | 0.144 | -0.046 | 0.181 | **0.967** | **501** | **531** | 8 | 5595 | ua_detrac_hybridsort_yolov8_fps5_20260816_111023 |
| ua_detrac | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 60 | 0.139 | **0.129** | 0.151 | -0.020 | 0.242 | 0.921 | 2497 | 1169 | 11 | 4281 | ua_detrac_analytics_bytetrack_yolov8_fps5_20260816_111023 |
| ua_detrac | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 60 | **0.141** | 0.128 | **0.156** | -0.019 | **0.247** | 0.953 | 1422 | 849 | 10 | 4353 | ua_detrac_analytics_bytetrack_plus_yolov8_fps5_20260816_111023 |
| ua_detrac | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 60 | 0.120 | 0.128 | 0.114 | **-0.015** | 0.202 | 0.608 | 23547 | 1001 | **14** | **3363** | ua_detrac_botsort_yolov8_fps5_20260816_111023 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 27 | 0.283 | 0.197 | 0.418 | 0.178 | 0.383 | 0.987 | 29 | 90 | **0** | 326 | trafficmot_fasttracker_yolov8_fps5_20260816_111023 |
| trafficmot | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 27 | 0.293 | 0.203 | 0.433 | 0.200 | 0.383 | 0.947 | 81 | 113 | **0** | 326 | trafficmot_ocsort_yolov8_fps5_20260816_111023 |
| trafficmot | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 27 | 0.293 | 0.202 | 0.434 | 0.199 | 0.382 | 0.948 | 84 | 111 | **0** | 326 | trafficmot_hybridsort_yolov8_fps5_20260816_111023 |
| trafficmot | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 27 | **0.308** | 0.223 | **0.435** | **0.222** | **0.422** | **0.989** | **26** | 145 | **0** | **271** | trafficmot_analytics_bytetrack_yolov8_fps5_20260816_111023 |
| trafficmot | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 27 | **0.308** | **0.224** | **0.435** | **0.222** | **0.422** | 0.988 | **26** | 144 | **0** | **271** | trafficmot_analytics_bytetrack_plus_yolov8_fps5_20260816_111023 |
| trafficmot | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 27 | 0.275 | 0.188 | 0.413 | 0.186 | 0.354 | 0.913 | 274 | **89** | **0** | 343 | trafficmot_botsort_yolov8_fps5_20260816_111023 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 36 | 0.172 | 0.103 | 0.305 | -1.574 | 0.243 | 0.969 | 124 | 486 | **0** | 198 | cityflow_fasttracker_yolov8_fps5_20260816_111023 |
| cityflow | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 36 | 0.183 | 0.111 | **0.315** | -1.432 | 0.266 | 0.972 | 91 | 439 | **0** | 123 | cityflow_ocsort_yolov8_fps5_20260816_111023 |
| cityflow | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 36 | 0.182 | 0.111 | **0.315** | -1.427 | 0.265 | 0.970 | 91 | 453 | **0** | 124 | cityflow_hybridsort_yolov8_fps5_20260816_111023 |
| cityflow | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 36 | 0.181 | 0.113 | 0.303 | -1.475 | 0.269 | **0.977** | **89** | 438 | **0** | 45 | cityflow_analytics_bytetrack_yolov8_fps5_20260816_111023 |
| cityflow | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 36 | 0.179 | 0.113 | 0.296 | -1.478 | 0.265 | 0.972 | 120 | 464 | **0** | 54 | cityflow_analytics_bytetrack_plus_yolov8_fps5_20260816_111023 |
| cityflow | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 36 | **0.188** | **0.121** | 0.312 | **-1.189** | **0.284** | 0.876 | 798 | **432** | **0** | **15** | cityflow_botsort_yolov8_fps5_20260816_111023 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 24 | 0.205 | 0.192 | 0.221 | 0.194 | 0.323 | 0.865 | 314 | 482 | 0 | 253 | lumana_benchmark_fasttracker_yolov8_fps5_20260816_112009 |
| lumana_benchmark | ocsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 24 | 0.208 | 0.195 | 0.222 | **0.201** | 0.327 | **0.922** | **152** | 542 | 0 | 306 | lumana_benchmark_ocsort_yolov8_fps5_20260816_112009 |
| lumana_benchmark | hybridsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 24 | 0.206 | 0.194 | 0.219 | 0.200 | 0.323 | 0.915 | 153 | 535 | 0 | 311 | lumana_benchmark_hybridsort_yolov8_fps5_20260816_112009 |
| lumana_benchmark | analytics_bytetrack | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 24 | 0.210 | **0.198** | 0.224 | 0.200 | 0.328 | 0.896 | 277 | 624 | **1** | **223** | lumana_benchmark_analytics_bytetrack_yolov8_fps5_20260816_112009 |
| lumana_benchmark | analytics_bytetrack_plus | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 24 | **0.212** | 0.197 | **0.229** | 0.200 | **0.333** | 0.908 | 282 | 648 | 0 | 231 | lumana_benchmark_analytics_bytetrack_plus_yolov8_fps5_20260816_112009 |
| lumana_benchmark | botsort | yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles | 5 | 24 | 0.195 | 0.180 | 0.213 | 0.187 | 0.293 | 0.780 | 582 | **481** | **1** | 250 | lumana_benchmark_botsort_yolov8_fps5_20260816_112009 |
