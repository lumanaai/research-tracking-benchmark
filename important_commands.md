# Important commands

Project root: `/mnt/nas/Users/ibrahim/tracking`  
Always use the project venv:

```bash
cd /mnt/nas/Users/ibrahim/tracking
.venv/bin/python ...
```

Data / artifacts live on the SSD under `/media/7TBSSD/data/tracking/`.

| Artifact | Path |
|----------|------|
| Normalized MOT layouts | `mot/<Benchmark>/<split>/<seq>/` |
| Detection cache | `detections/<benchmark>/<split>/<detector_id>/<seq>/det.txt` |
| Experiments | `experiments/<run_id>/{tracks,eval,config.json}` |
| Findings indexes | `experiments/_findings/<benchmark>/<tracker>/<detector_id>/findings.csv` |
| GT viz videos | `*_visualizations/` siblings of each dataset |
| Tracker-grid HTML | `visualizations/<detector_stem>/fps{5,10,full}/<benchmark>/` |
| One-off MOT overlay | `*_visualizations/tracked/` or an explicit `--out` path |

---

## 1. Visualize ground truth (per benchmark)

```bash
# UA-DETRAC → UA-DETRAC_visualizations/{train,test}/
.venv/bin/python scripts/visualize/visualize_ua_detrac.py /media/7TBSSD/data/tracking/UA-DETRAC
# Useful: --split train|test --sequences MVI_39811 --show-ignored --max-sequences 3

# TrafficMOT → TrafficMOT_visualizations/{Fully_annotate,FirstFrame_annotate}/
.venv/bin/python scripts/visualize/visualize_trafficmot.py /media/7TBSSD/data/tracking/TrafficMOT
# Useful: --split Fully_annotate --sequences BLR_1651660240.5260816 --max-sequences 3

# CityFlow → CityFlow_visualizations/{train,validation}/<scenario>/<camera>.mp4
.venv/bin/python scripts/visualize/visualize_cityflow.py /media/7TBSSD/data/tracking/CityFlow
# Useful: --split train --sequences S01/c001 --show-roi --max-sequences 3 --max-frames 300

# FastTracker-Benchmark → FastTracker-Benchmark_visualizations/<seq>.mp4
# Pass --extract once to unzip train/*.zip
.venv/bin/python scripts/visualize/visualize_fasttracker.py /media/7TBSSD/data/tracking/FastTracker-Benchmark --extract
# Useful: --sequences task_day_occlusion --show-ignore --max-sequences 3 --max-frames 300

# LumanaBenchmark → LumanaBenchmark_visualizations/<seq>.mp4 (skips seqs without img1/)
.venv/bin/python scripts/visualize/visualize_lumana.py /media/7TBSSD/data/tracking/LumanaBenchmark
```

---

## 2. MOT pipeline — convert datasets to MOT layout

Idempotent. Pass `--sequences` to limit work (especially important for CityFlow frame extract).

```bash
.venv/bin/python -m mot_pipeline.run convert --benchmark fasttracker_bench
.venv/bin/python -m mot_pipeline.run convert --benchmark ua_detrac
.venv/bin/python -m mot_pipeline.run convert --benchmark trafficmot          # default split: Fully_annotate
.venv/bin/python -m mot_pipeline.run convert --benchmark cityflow --sequences S01_c001
.venv/bin/python -m mot_pipeline.run convert --benchmark cityflow --force    # full train extract
.venv/bin/python -m mot_pipeline.run convert --benchmark lumana_benchmark    # symlink GT annotations
```

---

## 3. Detect (cache under `detections/`)

### Oracle GT boxes

```bash
.venv/bin/python -m mot_pipeline.run detect --benchmark fasttracker_bench --detector gt
.venv/bin/python -m mot_pipeline.run detect --benchmark ua_detrac --detector gt
.venv/bin/python -m mot_pipeline.run detect --benchmark trafficmot --detector gt
.venv/bin/python -m mot_pipeline.run detect --benchmark cityflow --detector gt --sequences S01_c001
.venv/bin/python -m mot_pipeline.run detect --benchmark lumana_benchmark --detector gt
```

### Reuse existing sequence-local dets (`<seq>/det/det.txt`)

FastTracker-Benchmark already has YOLOv8s `det/det.txt`. CityFlow convert links baseline `det_yolo3.txt` → `det/det.txt`.

```bash
.venv/bin/python -m mot_pipeline.run detect \
  --benchmark fasttracker_bench --detector existing

.venv/bin/python -m mot_pipeline.run detect \
  --benchmark cityflow --detector existing --sequences S01_c001

# CityFlow alternate baseline file:
.venv/bin/python -m mot_pipeline.run detect \
  --benchmark cityflow --detector existing \
  --det-name det/det_ssd512.txt --sequences S01_c001
```

### Run YOLOv8s from scratch

```bash
.venv/bin/python -m mot_pipeline.run detect \
  --benchmark fasttracker_bench --detector yolov8 --device cuda:0

.venv/bin/python -m mot_pipeline.run detect \
  --benchmark ua_detrac --detector yolov8 --device cuda:0

.venv/bin/python -m mot_pipeline.run detect \
  --benchmark trafficmot --detector yolov8 --device cuda:0

.venv/bin/python -m mot_pipeline.run detect \
  --benchmark cityflow --detector yolov8 --device cuda:0 --sequences S01_c001

# Useful knobs: --imgsz 1280 --conf 0.25 --batch 16 --force-detect --exclude-motorcycles
# Weights default: /media/7TBSSD/data/tracking/weights/yolov8m-expert_eff-1_2.pt
```

Standalone (legacy, writes into each sequence’s `det/det.txt`):

```bash
.venv/bin/python scripts/detect/run_yolo_dets.py \
  /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
  --weights /media/7TBSSD/data/tracking/weights/yolov8m-expert_eff-1_2.pt \
  --device cuda:0 --imgsz 1280 --conf 0.25 --batch 16
```

---

## 4. Track + evaluate (preferred: `mot_pipeline`)

`all` = detect (cached) → track → TrackEval → update findings index.

Trackers: `fasttracker`, `ocsort`, `hybridsort`, `analytics_bytetrack`, `analytics_bytetrack_plus`, `analytics_bytetrack_plus_aug19`, `botsort` (all motion-only). `traffictrack` is still a stub (`TODOs.md`).

Default configs:
- FastTracker: per-benchmark (`fasttracker_bench.json`, `detrac_no_roi.json`, `general_no_roi.json`)
- OC-SORT: `mot_pipeline/configs/trackers/ocsort/default.json`
- HybridSORT: `mot_pipeline/configs/trackers/hybridsort/default.json`
- BoT-SORT: `mot_pipeline/configs/trackers/botsort/default.json` (ReID off; CMC `none` by default)
- Analytics ByteTrack: `mot_pipeline/configs/trackers/analytics_bytetrack/benchmark.json` (pass `production.json` for the as-deployed settings)
- Analytics ByteTrack Plus: `mot_pipeline/configs/trackers/analytics_bytetrack_plus/benchmark.json`
- Analytics ByteTrack Plus Aug19: `mot_pipeline/configs/trackers/analytics_bytetrack_plus_aug19/aug19.json`

### FastTracker-Benchmark

```bash
# Reuse existing YOLO dets
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker fasttracker --detector existing

# Fresh YOLO
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker fasttracker --detector yolov8 --device cuda:0

# Oracle GT dets
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker fasttracker --detector gt

# Single hard sequence
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker fasttracker --detector existing \
  --sequences task_day_occlusion --run-id ft_occ_existing

# OC-SORT / HybridSORT / BoT-SORT on the same dets
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker ocsort --detector existing \
  --sequences task_day_occlusion --run-id oc_occ_existing

.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker hybridsort --detector existing \
  --sequences task_day_occlusion --run-id hs_occ_existing

.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker botsort --detector existing \
  --sequences task_day_occlusion --run-id bs_occ_existing
```

### Analytics ByteTrack (in-house tracker)

Imported live out of the `analytics/` clone — no vendored copy, so a `git pull`
there changes what runs here. Works on every benchmark; CPU-only.

```bash
# Oracle GT dets (association-only)
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker analytics_bytetrack --detector gt

# Real dets, benchmark config (n_init=1, per-class thresholds)
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker analytics_bytetrack --detector existing \
  --sequences task_day_occlusion

# As deployed in the analyzer (n_init=3, flat 0.45 threshold)
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker analytics_bytetrack --detector existing \
  --sequences task_day_occlusion \
  --tracker-config mot_pipeline/configs/trackers/analytics_bytetrack/production.json

# Other benchmarks
.venv/bin/python -m mot_pipeline.run all --benchmark ua_detrac  --tracker analytics_bytetrack --detector gt
.venv/bin/python -m mot_pipeline.run all --benchmark trafficmot --tracker analytics_bytetrack --detector gt
.venv/bin/python -m mot_pipeline.run all --benchmark cityflow   --tracker analytics_bytetrack --detector gt
.venv/bin/python -m mot_pipeline.run all --benchmark lumana_benchmark --tracker analytics_bytetrack --detector gt
```

### Analytics ByteTrack Plus

Same analytics wrapper path; engine is `ByteTrackerPlus/byte_tracker_plus.py`
(crossover protection, parked-vehicle hold). Default:
`mot_pipeline/configs/trackers/analytics_bytetrack_plus/benchmark.json`.

```bash
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker analytics_bytetrack_plus --detector existing \
  --sequences task_day_occlusion

.venv/bin/python -m mot_pipeline.run all \
  --benchmark lumana_benchmark --tracker analytics_bytetrack_plus --detector gt
```

### Analytics ByteTrack Plus Aug19

Independent Plus engine (`ByteTrackerPlusAug19/`) with CIoU + Aug19 tuning.
Default: `mot_pipeline/configs/trackers/analytics_bytetrack_plus_aug19/aug19.json`.

```bash
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker analytics_bytetrack_plus_aug19 --detector yolov8 \
  --fps 5 --sequences task_day_occlusion
```

### UA-DETRAC

```bash
.venv/bin/python -m mot_pipeline.run all \
  --benchmark ua_detrac --tracker fasttracker --detector gt

.venv/bin/python -m mot_pipeline.run all \
  --benchmark ua_detrac --tracker fasttracker --detector yolov8 --device cuda:0 \
  --sequences MVI_39811
```

### TrafficMOT

```bash
.venv/bin/python -m mot_pipeline.run all \
  --benchmark trafficmot --tracker fasttracker --detector gt

.venv/bin/python -m mot_pipeline.run all \
  --benchmark trafficmot --tracker fasttracker --detector yolov8 --device cuda:0
```

### CityFlow

```bash
# Full convert once (extracts vdo.avi → img1), then track
.venv/bin/python -m mot_pipeline.run convert --benchmark cityflow --force

.venv/bin/python -m mot_pipeline.run all \
  --benchmark cityflow --tracker fasttracker --detector existing

.venv/bin/python -m mot_pipeline.run all \
  --benchmark cityflow --tracker fasttracker --detector yolov8 --device cuda:0 \
  --sequences S01_c001
```

### Stage-wise (same experiment pieces)

```bash
.venv/bin/python -m mot_pipeline.run detect --benchmark fasttracker_bench --detector existing
.venv/bin/python -m mot_pipeline.run track  --benchmark fasttracker_bench --tracker fasttracker --detector existing \
  --run-id my_run
.venv/bin/python -m mot_pipeline.run eval --run-id my_run
```

### Eval only (re-score an existing experiment)

```bash
.venv/bin/python -m mot_pipeline.run eval --run-id smoke_ft_existing_occ
```

Metrics land in:

- `experiments/<run_id>/eval/summary.csv` / `summary.json`
- `experiments/_findings/<benchmark>/fasttracker/<detector_id>/findings.csv`

Useful common flags: `--sequences …`, `--run-id …`, `--tracker-config path.json`, `--exclude-motorcycles`, `--force-detect`.

---

## 5. Visualize tracker outputs

Generic MOT overlay (works for any benchmark once you have frames + a track txt):

```bash
.venv/bin/python scripts/visualize/visualize_mot_results.py \
  --frames /media/7TBSSD/data/tracking/mot/FastTracker-Benchmark/train/task_day_occlusion/img1 \
  --results /media/7TBSSD/data/tracking/experiments/<run_id>/tracks/task_day_occlusion.txt \
  --out /media/7TBSSD/data/tracking/FastTracker-Benchmark_visualizations/tracked/task_day_occlusion.mp4 \
  --fps 30 --label "FastTracker"
```

### Batch grids from run_all.sh findings (`visualize_all.sh`)

Does **not** re-detect or re-track. Reads the **same latest findings row** as
`compare_findings.py` and overlays `experiments/<run_id>/tracks/*.txt` on MOT
`img1` frames.

Default generate: YOLO (`yolov8m-expert_eff`), **3 sequences** × 8 s snippets
from 20% in, max width 480, one HTML grid per benchmark × FPS. The viewer shows
one video at a time with all trackers tiled; **Dataset** and **Video** dropdowns
switch grids / sequences (dataset list = sibling folders under the same FPS
dir that already have `manifest.json`).

If only 3 videos appear, the rest were never rendered (`N_SEQS=3`). Tracking
results can still exist for every sequence.

```bash
# Step 1 — generate snippets + HTML (does not start a server)
./scripts/batch/visualize_all.sh
BENCHMARKS=lumana_benchmark FPS_VALUES="5" ./scripts/batch/visualize_all.sh
N_SEQS=0 BENCHMARKS=lumana_benchmark FPS_VALUES="5" ./scripts/batch/visualize_all.sh  # all seqs with img1/

# Step 2 — host
./scripts/batch/serve_visualizations.sh
# PORT=9000 ./scripts/batch/serve_visualizations.sh
```

Then open `http://127.0.0.1:8765/yolov8m_expert_eff/fps5/lumana_benchmark/index.html`
(or `http://127.0.0.1:8765/` for the landing list). Hard-refresh after re-generate.
Stop with Ctrl+C, or `pkill -f 'visualize_all.py --serve-only'`.

| Env | Default | Meaning |
|-----|---------|---------|
| `BENCHMARKS` | all five | CLI ids (`lumana_benchmark`, `fasttracker_bench`, …) |
| `TRACKERS` | all wired motion trackers | same list as `run_all.sh` |
| `FPS_VALUES` | `5 10 full` | must match findings `target_fps` |
| `N_SEQS` | `3` | `0` = every sequence with `img1/` |
| `SNIPPET_SEC` | `8` | `0` = full clip from the start |
| `START_FRAC` | `0.2` | ignored when `SNIPPET_SEC=0` |
| `FORCE` | `0` | `1` = re-encode existing JPEG cells |
| `RUN_ID_SUBSTR` | latest row | pin a sweep timestamp |
| `OUT_ROOT` | SSD `visualizations/` | do not write grids to the NAS |

Rewrite HTML from existing `cells/` + `manifest.json` (no re-encode):

```bash
.venv/bin/python scripts/visualize/visualize_all.py --html-only \
  --out-root /media/7TBSSD/data/tracking/visualizations
```

One benchmark / FPS via Python:

```bash
.venv/bin/python scripts/visualize/visualize_all.py \
  --benchmark fasttracker_bench --detector yolov8 --fps 5
```

Layout on SSD:

```
visualizations/<detector_stem>/fps{5,10,full}/<benchmark>/{index.html,manifest.json,cells/…}
```

---

### One experiment, all sequences

```bash
RUN=my_run
BENCH_MOT=/media/7TBSSD/data/tracking/mot/FastTracker-Benchmark/train
OUT=/media/7TBSSD/data/tracking/FastTracker-Benchmark_visualizations/tracked
mkdir -p "$OUT"
for res in /media/7TBSSD/data/tracking/experiments/$RUN/tracks/*.txt; do
  name=$(basename "$res" .txt)
  .venv/bin/python scripts/visualize/visualize_mot_results.py \
    --frames "$BENCH_MOT/$name/img1" \
    --results "$res" \
    --out "$OUT/${name}.mp4" \
    --fps 30 --label "FastTracker ($RUN)"
done
```

UA-DETRAC example (frames via MOT symlink; DETRAC fps=25):

```bash
.venv/bin/python scripts/visualize/visualize_mot_results.py \
  --frames /media/7TBSSD/data/tracking/mot/UA-DETRAC/train/MVI_39811/img1 \
  --results /media/7TBSSD/data/tracking/experiments/<run_id>/tracks/MVI_39811.txt \
  --out /media/7TBSSD/data/tracking/UA-DETRAC_visualizations/tracked/MVI_39811.mp4 \
  --fps 25 --label "FastTracker"
```

TrafficMOT (fps≈10) / CityFlow (fps≈10; c015 is 8):

```bash
# TrafficMOT
.venv/bin/python scripts/visualize/visualize_mot_results.py \
  --frames /media/7TBSSD/data/tracking/mot/TrafficMOT/Fully_annotate/<seq>/img1 \
  --results /media/7TBSSD/data/tracking/experiments/<run_id>/tracks/<seq>.txt \
  --out /media/7TBSSD/data/tracking/TrafficMOT_visualizations/tracked/<seq>.mp4 \
  --fps 10 --label "FastTracker"

# CityFlow
.venv/bin/python scripts/visualize/visualize_mot_results.py \
  --frames /media/7TBSSD/data/tracking/mot/CityFlow/train/S01_c001/img1 \
  --results /media/7TBSSD/data/tracking/experiments/<run_id>/tracks/S01_c001.txt \
  --out /media/7TBSSD/data/tracking/CityFlow_visualizations/tracked/S01_c001.mp4 \
  --fps 10 --label "FastTracker"
```

---

## 6. Legacy FastTracker CLI (still works; bypasses `mot_pipeline`)

Detection-fed tracker only (no YOLOX), from the `FastTracker/` clone:

```bash
cd /mnt/nas/Users/ibrahim/tracking/FastTracker

# FastTracker-Benchmark, GT-as-dets (drop ignore class 9)
PYTHONPATH=. ../.venv/bin/python tools/track_from_dets.py \
  /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
  --det-source gt --drop-classes 9 \
  --config configs/fasttracker_bench.json

# Same benchmark, YOLO det/det.txt, vehicles only (COCO: car/moto/bus/truck)
PYTHONPATH=. ../.venv/bin/python tools/track_from_dets.py \
  /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
  --det-source det --config configs/fasttracker_bench.json \
  --keep-classes 2 3 5 7

# One sequence
PYTHONPATH=. ../.venv/bin/python tools/track_from_dets.py \
  /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
  --sequences task_day_occlusion --det-source det \
  --config configs/fasttracker_bench.json --keep-classes 2 3 5 7
```

Default results path:  
`/media/7TBSSD/data/tracking/FastTracker-Benchmark/fasttracker_results/track_results/<seq>.txt`

---

## 7. Findings / comparison

After each successful `eval` / `all`, a row is upserted into:

```text
/media/7TBSSD/data/tracking/experiments/_findings/<benchmark>/<tracker>/<detector_id>/findings.csv
```

Aggregate a comparison table (project copy under `results/comparisons/`):

```bash
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --format both \
  --out results/comparisons/gt_vehicles.md

.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles --format both \
  --out results/comparisons/yolov8m_expert_eff.md
```

Benchmark dataset stats (SSD):

```bash
.venv/bin/python scripts/analysis/compute_benchmark_statistics.py
# → /media/7TBSSD/data/tracking/benchmarks_statistics.md
```

Examples:

```bash
# FastTracker-Bench, reused YOLO dets
less /media/7TBSSD/data/tracking/experiments/_findings/fasttracker_bench/fasttracker/existing_det_vehicles/findings.csv

# UA-DETRAC oracle
less /media/7TBSSD/data/tracking/experiments/_findings/ua_detrac/fasttracker/gt_vehicles/findings.csv

# Enumerate all combination indexes
find /media/7TBSSD/data/tracking/experiments/_findings -name findings.csv
```

---

## 8. Batch sweeps

```bash
# Sequential GT-oracle all benches × trackers
./scripts/batch/run_all_gt_trackers.sh

# Parallel YOLO detect (multi-GPU) → track → eval
DETECTOR=yolov8 GPUS="0 1 2 3" ./scripts/batch/run_all_parallel.sh
DETECTOR=yolov8 GPUS="0 1 2 3 4 5 6 7" SHARD_SEQS=1 TRACK_JOBS=12 ./scripts/batch/run_all_parallel.sh

# Overlay the same latest findings as HTML grids (no re-track)
./scripts/batch/visualize_all.sh
./scripts/batch/serve_visualizations.sh
```

---

## 9. Quick cheat-sheet

| Goal | Command sketch |
|------|----------------|
| GT video | `scripts/visualize/visualize_{ua_detrac,trafficmot,cityflow,fasttracker,lumana}.py` |
| Ensure MOT layout | `-m mot_pipeline.run convert --benchmark …` |
| Use existing dets | `--detector existing` |
| Run YOLO | `--detector yolov8 --device cuda:0` |
| Oracle association | `--detector gt` |
| Full track+metrics | `-m mot_pipeline.run all --benchmark … --tracker {fasttracker,ocsort,hybridsort,analytics_bytetrack,analytics_bytetrack_plus,analytics_bytetrack_plus_aug19,botsort} --detector …` |
| Parallel YOLO sweep | `./scripts/batch/run_all_parallel.sh` |
| Re-eval | `-m mot_pipeline.run eval --run-id …` |
| Overlay tracks (one seq) | `scripts/visualize/visualize_mot_results.py --frames … --results … --out …` |
| Tracker grids (from findings) | `./scripts/batch/visualize_all.sh` then `./scripts/batch/serve_visualizations.sh` |
| All sequences in a grid | `N_SEQS=0 BENCHMARKS=… FPS_VALUES="5" ./scripts/batch/visualize_all.sh` |
| Compare runs | `scripts/analysis/compare_findings.py --detector-id …` |
| Findings CSV | `experiments/_findings/<bench>/<tracker>/<det_id>/findings.csv` |


--------------------------------------------
GPUS="0 1 2 3 4 5 6 7" TRACK_JOBS=16 FPS_VALUES="5" ./scripts/batch/run_all.sh

# GT oracle dets
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --format both \
  --out results/comparisons/gt_vehicles.md

# GT oracle dets FPS=5
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --target-fps 5 --format both \
  --out results/comparisons/gt_vehicles_fps5.md

# GT oracle dets FPS=10
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id gt_vehicles --target-fps 10 --format both \
  --out results/comparisons/gt_vehicles_fps10.md

# native FPS
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles --format both \
  --out results/comparisons/yolov8m_expert_eff.md

# FPS=10
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles --target-fps 10 --format both \
  --out results/comparisons/yolov8m_expert_eff_fps10.md

# FPS=5
.venv/bin/python scripts/analysis/compare_findings.py \
  --detector-id yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles --target-fps 5 --format both \
  --out results/comparisons/yolov8m_expert_eff_fps5.md