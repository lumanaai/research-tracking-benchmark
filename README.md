# research-tracking-benchmark

Unified detect → track → eval pipeline for vehicle / traffic multi-object tracking.

This repo wraps several public trackers behind one CLI, caches detections once, and scores runs with TrackEval (HOTA / CLEAR / Identity). Datasets and heavy artifacts are expected on a local data root (defaults point at `/media/7TBSSD/data/tracking/`); source and configs live here.

For the longer operator notes, see [`AGENTS.md`](AGENTS.md).

## Benchmarks

| CLI name | Dataset |
|---|---|
| `fasttracker_bench` | [FastTracker-Benchmark](https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark) |
| `ua_detrac` | [UA-DETRAC](https://detrac-db.rit.albany.edu/) |
| `trafficmot` | TrafficMOT |
| `cityflow` | CityFlow / AIC22 (CityFlowV2) |
| `lumana_benchmark` | LumanaBenchmark (internal vehicle GT) |

Converters turn each dataset into a MOTChallenge-style layout under `mot/<Benchmark>/<split>/<seq>/`.

## Trackers

| CLI name | Notes |
|---|---|
| `fasttracker` | Vendored [FastTracker](https://github.com/Hamidreza-Hashempoor/FastTracker) |
| `ocsort` | Vendored [OC-SORT](https://github.com/noahcao/OC_SORT) (motion-only) |
| `hybridsort` | Vendored [HybridSORT](https://github.com/ymzisalok/HybridSORT) (motion-only path) |
| `botsort` | Vendored [BoT-SORT](https://github.com/NirAharon/BoT-SORT) (motion-only; ReID off) |
| `analytics_bytetrack` | In-house ByteTrack via the vendored `analytics/` tree |
| `traffictrack` | Stub only (see [`TODOs.md`](TODOs.md)) |

Upstream clones are committed as plain directories (not git submodules). BoT-SORT appearance ReID is deferred (`TODOs.md`); CMC defaults to `none` (enable `sparseOptFlow` via tracker JSON if needed).

## Detectors

| CLI name | Behavior |
|---|---|
| `gt` | Oracle: feed ground-truth boxes as detections (association-only) |
| `yolov8` | Ultralytics YOLO; writes a shared cache under `detections/` |
| `existing` | Reuse a sequence-local `det/det.txt` (optional `--det-name`) |
| `yolox` | Stub |

Default keep-sets are vehicle-oriented (motorcycles included; person / bicycle / ignore regions dropped). Pass `--exclude-motorcycles` to drop moto classes too. Class maps live in `mot_pipeline/class_maps.py`.

## Layout

```
mot_pipeline/          # convert / detect / track / eval package
scripts/               # visualize, convert helpers, batch sweeps, analysis
FastTracker/           # vendored tracker + TrackEval
OC_SORT/
HybridSORT/
BoT-SORT/              # motion-only via botsort adapter (FastReID stubbed)
analytics/             # in-house analytics (ByteTrack source used by the shim)
results/comparisons/   # published comparison tables
```

Typical data root (SSD, not this repo):

```
<DATA>/mot/                  # normalized MOTChallenge sequences
<DATA>/detections/           # shared detector cache
<DATA>/experiments/          # per-run config, tracks, eval
<DATA>/weights/              # YOLO / ReID checkpoints
```

Override with env / CLI where supported; see `mot_pipeline/paths.py`.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
# Tracker adapters also need torch, scipy, lap, cython_bbox, loguru, etc.
# Use a CUDA torch build if you will run yolov8 on GPU.
```

Point `.venv` at a fast local drive if the project root is on network storage.

## Usage

```bash
# MOT layout (idempotent)
.venv/bin/python -m mot_pipeline.run convert --benchmark fasttracker_bench

# Full run: detect → track → eval
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench \
  --tracker fasttracker \
  --detector yolov8 \
  --device cuda:0

# Oracle GT detections
.venv/bin/python -m mot_pipeline.run all \
  --benchmark ua_detrac \
  --tracker ocsort \
  --detector gt

# BoT-SORT (motion-only, no ReID)
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench \
  --tracker botsort \
  --detector gt \
  --sequences task_day_occlusion
```

Stage-wise: `detect`, `track`, `eval --run-id <id>`.

Batch helpers: `scripts/batch/run_all.sh`, `scripts/batch/run_all_gt_trackers.sh`, `scripts/batch/run_all_parallel.sh` (default tracker list includes `botsort`).  
Comparison tables: `scripts/analysis/compare_findings.py`, `scripts/analysis/run_all_compare_findings.sh`, and `results/comparisons/`.

## Metrics

Each successful eval writes `experiments/<run_id>/eval/summary.csv` (and JSON) with per-sequence and `COMBINED` **HOTA**, **CLEAR** (MOTA / IDSW / …), and **Identity** (IDF1). Runs are also upserted into `experiments/_findings/<benchmark>/<tracker>/<detector_id>/`.

## License

Pipeline code in this repo: see project owners. Vendored trackers keep their upstream licenses (`FastTracker/LICENSE`, `OC_SORT/LICENSE`, `HybridSORT/LICENSE`, `BoT-SORT/LICENSE`).
