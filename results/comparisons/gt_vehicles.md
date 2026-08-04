# Tracker comparison

_16 runs from `/media/7TBSSD/data/tracking/experiments/_findings`_

| benchmark | tracker | detector_id | seqs | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | MT | ML | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fasttracker_bench | fasttracker | gt_vehicles | 12 | 0.971 | 0.969 | 0.974 | 0.982 | 0.986 | 133 | 358 | 848 | 8 | fasttracker_bench_fasttracker_gt_20260729_085336 |
| fasttracker_bench | ocsort | gt_vehicles | 12 | 0.991 | 0.996 | 0.986 | 0.996 | 0.992 | 58 | 52 | 1009 | 7 | fasttracker_bench_ocsort_gt_20260729_085336 |
| fasttracker_bench | hybridsort | gt_vehicles | 12 | 0.991 | 0.996 | 0.986 | 0.996 | 0.992 | 70 | 69 | 1009 | 7 | fasttracker_bench_hybridsort_gt_20260729_085336 |
| fasttracker_bench | analytics_bytetrack | gt_vehicles | 12 | 0.992 | 0.999 | 0.985 | 0.999 | 0.992 | 49 | 35 | 1013 | 6 | fasttracker_bench_analytics_bytetrack_gt_vehicles_benchmark_20260804_125023 |
| ua_detrac | fasttracker | gt_vehicles | 60 | 0.880 | 0.863 | 0.900 | 0.942 | 0.971 | 25 | 855 | 5605 | 38 | ua_detrac_fasttracker_gt_20260729_085336 |
| ua_detrac | ocsort | gt_vehicles | 60 | 0.972 | 0.972 | 0.973 | 0.972 | 0.986 | 11 | 17 | 5798 | 27 | ua_detrac_ocsort_gt_20260729_085336 |
| ua_detrac | hybridsort | gt_vehicles | 60 | 0.972 | 0.972 | 0.972 | 0.972 | 0.986 | 12 | 17 | 5798 | 26 | ua_detrac_hybridsort_gt_20260729_085336 |
| ua_detrac | analytics_bytetrack | gt_vehicles | 60 | 0.988 | 0.991 | 0.985 | 0.991 | 0.992 | 27 | 1 | 5918 | 21 | ua_detrac_analytics_bytetrack_gt_vehicles_benchmark_20260804_125219 |
| trafficmot | fasttracker | gt_vehicles | 27 | 0.900 | 0.880 | 0.924 | 0.971 | 0.985 | 5 | 42 | 706 | 1 | trafficmot_fasttracker_gt_20260729_085336 |
| trafficmot | ocsort | gt_vehicles | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 7 | 32 | 706 | 3 | trafficmot_ocsort_gt_20260729_085336 |
| trafficmot | hybridsort | gt_vehicles | 27 | 0.990 | 0.989 | 0.991 | 0.989 | 0.994 | 7 | 32 | 706 | 3 | trafficmot_hybridsort_gt_20260729_085336 |
| trafficmot | analytics_bytetrack | gt_vehicles | 27 | 0.996 | 0.997 | 0.996 | 0.997 | 0.998 | 6 | 31 | 724 | 0 | trafficmot_analytics_bytetrack_gt_vehicles_benchmark_20260804_125215 |
| cityflow | fasttracker | gt_vehicles | 36 | 0.896 | 0.897 | 0.897 | 0.955 | 0.956 | 60 | 193 | 764 | 1 | cityflow_fasttracker_gt_20260729_085336 |
| cityflow | ocsort | gt_vehicles | 36 | 0.948 | 0.961 | 0.936 | 0.960 | 0.958 | 60 | 156 | 654 | 16 | cityflow_ocsort_gt_20260729_085336 |
| cityflow | hybridsort | gt_vehicles | 36 | 0.942 | 0.960 | 0.924 | 0.959 | 0.951 | 65 | 163 | 654 | 16 | cityflow_hybridsort_gt_20260729_085336 |
| cityflow | analytics_bytetrack | gt_vehicles | 36 | 0.964 | 0.989 | 0.939 | 0.989 | 0.957 | 29 | 186 | 766 | 1 | cityflow_analytics_bytetrack_gt_vehicles_benchmark_20260804_125608 |
