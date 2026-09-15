# Add MOT benchmark — reference

## Existing bench ids

| `bench_id` | `mot_folder` | `default_split` | Converter / ensure |
|---|---|---|---|
| `fasttracker_bench` | `FastTracker-Benchmark` | `train` | symlink in `benchmarks/fasttracker_bench.py` |
| `ua_detrac` | `UA-DETRAC` | `train` | `converters/prepare_ua_detrac.py` |
| `trafficmot` | `TrafficMOT` | `Fully_annotate` | `converters/prepare_trafficmot.py` |
| `cityflow` | `CityFlow` | `train` | `converters/prepare_cityflow.py` (extracts `vdo.avi` → `img1`) |
| `lumana_benchmark` | `LumanaBenchmark` | `train` | symlink in `benchmarks/lumana_benchmark.py` → `gt_annotations_manually_validated/` |

## Minimum file touch list

1. `mot_pipeline/paths.py` — `RAW_DATASETS`
2. `mot_pipeline/converters/prepare_<bench>.py` (if needed)
3. `mot_pipeline/benchmarks/<bench>.py` + `benchmarks/__init__.py`
4. `mot_pipeline/class_maps.py`
5. `mot_pipeline/run.py` — `--benchmark` choices
6. `scripts/batch/run_all{,_parallel,_gt_trackers}.sh` — `BENCHMARKS`
7. `scripts/batch/visualize_all.sh` — `BENCHMARKS`
8. `scripts/visualize/visualize_all.py` — `DEFAULT_BENCHMARKS`, `PREFERRED_SEQUENCES`
9. `scripts/visualize/grid_page.py` — `DATASET_LABELS`
10. `scripts/analysis/compare_findings.py` — `BENCH_ORDER`, `NATIVE_FPS`
11. `scripts/visualize/visualize_<bench>.py`
12. `AGENTS.md` (+ `README.md` / `important_commands.md`)

## Class policy notes

- Default keep-sets drop person / bicycle / ignore; **include** motorcycles unless `--exclude-motorcycles`.
- Eval rewrites kept GT to class `1` and scores TrackEval’s `person` slot as class-agnostic vehicle.
- Document GT class numbering in the converter / `AGENTS.md` dataset section.
