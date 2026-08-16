# Tracker comparison

_30 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | FPS | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDCons | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | gt_vehicles | 30 | 12 | 0.971 | 0.969 | 0.974 | 0.982 | 0.986 | 0.991 | 133 | 358 | 848 | 8 | fasttracker_bench_fasttracker_gt_20260816_113819 |
| fasttracker_bench | ocsort | gt_vehicles | 30 | 12 | 0.991 | 0.996 | 0.986 | 0.996 | 0.992 | **0.992** | 58 | 52 | 1009 | 7 | fasttracker_bench_ocsort_gt_20260816_113819 |
| fasttracker_bench | hybridsort | gt_vehicles | 30 | 12 | 0.991 | 0.996 | 0.986 | 0.996 | 0.992 | **0.992** | 70 | 69 | 1009 | 7 | fasttracker_bench_hybridsort_gt_20260816_113819 |
| fasttracker_bench | analytics_bytetrack | gt_vehicles | 30 | 12 | 0.992 | **0.999** | 0.985 | 0.999 | 0.992 | **0.992** | **49** | 35 | 1013 | 6 | fasttracker_bench_analytics_bytetrack_gt_20260816_113819 |
| fasttracker_bench | analytics_bytetrack_plus | gt_vehicles | 30 | 12 | 0.990 | 0.998 | 0.981 | 0.998 | 0.990 | 0.988 | 107 | 103 | 1013 | 6 | fasttracker_bench_analytics_bytetrack_plus_gt_20260816_113819 |
| fasttracker_bench | botsort | gt_vehicles | 30 | 12 | **0.994** | 0.998 | **0.989** | **1.000** | **0.995** | **0.992** | 164 | **13** | **1021** | **0** | fasttracker_bench_botsort_gt_20260816_113819 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| ua_detrac | fasttracker | gt_vehicles | 25 | 60 | 0.880 | 0.863 | 0.900 | 0.942 | 0.971 | 0.999 | 25 | 855 | 5605 | 38 | ua_detrac_fasttracker_gt_20260816_113819 |
| ua_detrac | ocsort | gt_vehicles | 25 | 60 | 0.972 | 0.972 | 0.973 | 0.972 | 0.986 | **1.000** | **11** | 17 | 5798 | 27 | ua_detrac_ocsort_gt_20260816_113819 |
| ua_detrac | hybridsort | gt_vehicles | 25 | 60 | 0.972 | 0.972 | 0.972 | 0.972 | 0.986 | **1.000** | 12 | 17 | 5798 | 26 | ua_detrac_hybridsort_gt_20260816_113819 |
| ua_detrac | analytics_bytetrack | gt_vehicles | 25 | 60 | 0.988 | **0.991** | 0.985 | 0.991 | 0.992 | 0.999 | 27 | **1** | 5918 | 21 | ua_detrac_analytics_bytetrack_gt_20260816_113819 |
| ua_detrac | analytics_bytetrack_plus | gt_vehicles | 25 | 60 | 0.988 | **0.991** | 0.985 | 0.991 | 0.992 | 0.999 | 86 | 33 | 5914 | 25 | ua_detrac_analytics_bytetrack_plus_gt_20260816_113819 |
| ua_detrac | botsort | gt_vehicles | 25 | 60 | **0.989** | 0.987 | **0.991** | **1.000** | **1.000** | **1.000** | 16 | **1** | **5952** | **0** | ua_detrac_botsort_gt_20260816_113819 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| trafficmot | fasttracker | gt_vehicles | 10 | 27 | 0.900 | 0.880 | 0.924 | 0.971 | 0.985 | **1.000** | **5** | 42 | 706 | 1 | trafficmot_fasttracker_gt_20260816_113819 |
| trafficmot | ocsort | gt_vehicles | 10 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 0.999 | 7 | 32 | 706 | 3 | trafficmot_ocsort_gt_20260816_113819 |
| trafficmot | hybridsort | gt_vehicles | 10 | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 0.999 | 7 | 32 | 706 | 3 | trafficmot_hybridsort_gt_20260816_113819 |
| trafficmot | analytics_bytetrack | gt_vehicles | 10 | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 0.999 | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_gt_20260816_113819 |
| trafficmot | analytics_bytetrack_plus | gt_vehicles | 10 | 27 | **0.996** | **0.997** | **0.996** | 0.997 | **0.998** | 0.999 | 6 | **31** | 724 | **0** | trafficmot_analytics_bytetrack_plus_gt_20260816_113819 |
| trafficmot | botsort | gt_vehicles | 10 | 27 | 0.970 | 0.965 | 0.975 | **0.999** | **0.998** | 0.997 | 18 | 35 | **731** | **0** | trafficmot_botsort_gt_20260816_113819 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| cityflow | fasttracker | gt_vehicles | 10 | 36 | 0.896 | 0.897 | 0.897 | 0.955 | 0.956 | 0.985 | 60 | 193 | 764 | 1 | cityflow_fasttracker_gt_20260816_113819 |
| cityflow | ocsort | gt_vehicles | 10 | 36 | 0.948 | 0.961 | 0.936 | 0.960 | 0.958 | 0.982 | 60 | **156** | 654 | 16 | cityflow_ocsort_gt_20260816_113819 |
| cityflow | hybridsort | gt_vehicles | 10 | 36 | 0.942 | 0.960 | 0.924 | 0.959 | 0.951 | 0.981 | 65 | 163 | 654 | 16 | cityflow_hybridsort_gt_20260816_113819 |
| cityflow | analytics_bytetrack | gt_vehicles | 10 | 36 | 0.964 | **0.989** | 0.939 | 0.989 | 0.957 | **0.990** | **29** | 186 | 766 | 1 | cityflow_analytics_bytetrack_gt_20260816_113819 |
| cityflow | analytics_bytetrack_plus | gt_vehicles | 10 | 36 | 0.963 | 0.988 | 0.939 | 0.987 | 0.960 | 0.987 | 51 | 215 | 764 | 1 | cityflow_analytics_bytetrack_plus_gt_20260816_113819 |
| cityflow | botsort | gt_vehicles | 10 | 36 | **0.965** | 0.979 | **0.951** | **0.999** | **0.973** | 0.981 | 85 | 194 | **808** | **0** | cityflow_botsort_gt_20260816_113819 |
| ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ | ═══ |
| lumana_benchmark | fasttracker | gt_vehicles | 20 | 24 | 0.903 | 0.954 | 0.854 | 0.979 | 0.904 | 0.911 | 277 | 578 | 404 | 7 | lumana_benchmark_fasttracker_gt_20260816_113819 |
| lumana_benchmark | ocsort | gt_vehicles | 20 | 24 | 0.919 | 0.989 | 0.855 | 0.988 | 0.899 | 0.896 | 310 | **555** | 346 | 25 | lumana_benchmark_ocsort_gt_20260816_113819 |
| lumana_benchmark | hybridsort | gt_vehicles | 20 | 24 | 0.918 | 0.989 | 0.852 | 0.988 | 0.896 | 0.897 | 308 | 560 | 346 | 25 | lumana_benchmark_hybridsort_gt_20260816_113819 |
| lumana_benchmark | analytics_bytetrack | gt_vehicles | 20 | 24 | 0.932 | **0.997** | 0.871 | 0.997 | 0.916 | 0.923 | **238** | 558 | 435 | 6 | lumana_benchmark_analytics_bytetrack_gt_20260816_113819 |
| lumana_benchmark | analytics_bytetrack_plus | gt_vehicles | 20 | 24 | **0.952** | 0.995 | **0.912** | 0.994 | **0.940** | **0.928** | 304 | 627 | 425 | 8 | lumana_benchmark_analytics_bytetrack_plus_gt_20260816_113819 |
| lumana_benchmark | botsort | gt_vehicles | 20 | 24 | 0.913 | 0.993 | 0.840 | **0.998** | 0.889 | 0.893 | 348 | 558 | **464** | **0** | lumana_benchmark_botsort_gt_20260816_113819 |
