# AGENTS.md

Guidance for agents working in this vehicle-tracking project.

## Goal

Build and evaluate vehicle / traffic multi-object tracking benchmarks (UA-DETRAC, TrafficMOT, CityFlow, FastTracker-Benchmark, …). Prefer reusable scripts and clear data paths over one-off notebooks.

## Layout

| Path | Role |
|------|------|
| `/mnt/nas/Users/ibrahim/tracking/` | Project root (code, docs). NAS — slower I/O. |
| `/media/7TBSSD/data/tracking/` | Data + heavy artifacts on the fast local SSD. |
| `/media/7TBSSD/data/tracking/UA-DETRAC/` | UA-DETRAC dataset root. |
| `/media/7TBSSD/data/tracking/UA-DETRAC_visualizations/` | Rendered UA-DETRAC GT videos (sibling of the dataset). |
| `/media/7TBSSD/data/tracking/TrafficMOT/` | TrafficMOT dataset root. |
| `/media/7TBSSD/data/tracking/TrafficMOT_visualizations/` | Rendered TrafficMOT GT videos (sibling of the dataset). |
| `/media/7TBSSD/data/tracking/CityFlow/` | CityFlow / AIC22 (CityFlowV2) dataset root. |
| `/media/7TBSSD/data/tracking/CityFlow_visualizations/` | Rendered CityFlow GT videos (sibling of the dataset). |
| `/media/7TBSSD/data/tracking/FastTracker-Benchmark/` | FastTracker-Benchmark (HF) dataset root. |
| `/media/7TBSSD/data/tracking/FastTracker-Benchmark_visualizations/` | Rendered FastTracker GT videos (sibling of the dataset). |
| `/media/7TBSSD/data/tracking/venvs/tracking/` | Real Python virtualenv (fast drive). |
| `/media/7TBSSD/data/tracking/mot/` | Normalized MOTChallenge roots (converters / symlinks). |
| `/media/7TBSSD/data/tracking/detections/` | Shared detector cache (`<bench>/<split>/<detector_id>/<seq>/det.txt`). |
| `/media/7TBSSD/data/tracking/experiments/` | Track + eval runs (`<run_id>/{config,tracks,eval}/`). |
| `mot_pipeline/` | Unified detect→track→eval package (benchmarks / detectors / trackers). |
| `.venv` | Symlink from project root → that venv. |

Keep large data, model weights, and venvs on the SSD. Keep source and lightweight config in the project root.

## Environment

Always use the project venv (never system or `--user` installs for project work):

```bash
.venv/bin/python ...
.venv/bin/pip install ...
```

Dependencies are listed in `requirements.txt`. The venv lives on the fast drive and is linked as `.venv` so tools that expect a local `.venv` still work.

## UA-DETRAC

Dataset root: `/media/7TBSSD/data/tracking/UA-DETRAC`

Relevant structure (Kaggle layout nests folders once):

- `DETRAC-Images/DETRAC-Images/MVI_XXXXX/imgXXXXX.jpg` — frame sequences (100 total)
- `DETRAC-Train-Annotations-XML/.../*.xml` — 60 train sequences
- `DETRAC-Test-Annotations-XML/.../*.xml` — 40 test sequences

XML per sequence: `ignored_region` boxes; per-`frame` → `target_list` → `target` with `id`, `box` (`left`/`top`/`width`/`height`), and `attribute` (`vehicle_type`, etc.).

## Visualization

`scripts/visualize/visualize_ua_detrac.py` loads every annotated sequence and writes MP4s with GT boxes, track IDs, vehicle types, and trajectory trails.

```bash
.venv/bin/python scripts/visualize/visualize_ua_detrac.py /media/7TBSSD/data/tracking/UA-DETRAC
```

Default output: `/media/7TBSSD/data/tracking/UA-DETRAC_visualizations/{train,test}/`. Useful flags: `--split`, `--sequences`, `--show-ignored`, `--max-sequences`.

## TrafficMOT

Dataset root: `/media/7TBSSD/data/tracking/TrafficMOT`

Structure (see local `ReadMe.txt`):

- `image/Fully_annotate/<seq>/frame{N}.jpg` — fully labeled clips (~30 frames, even indices `0..58`)
- `image/FirstFrame_annotate/<seq>/...` — same frame layout; only frame 0 is labeled
- `ground_truth/{Fully_annotate,FirstFrame_annotate}/<seq>/frame{N}.csv` — per-frame CSV GT
- `ground_truth_json/*.json` — COCO-style files for the official dataloader (optional for viz)

CSV columns: `video,imagePath,imageHeight,imageWidth,classID,x,y,w,h,trackID`  
Classes (1–10): Motor_Bike, Bus, LMV, Auto, Bike, Pedestrian, LCV, E-rickshaw, Tractor, Truck.

This copy includes 27 fully annotated + 27 first-frame sequences (train subset described in the ReadMe).

### Visualization

`scripts/visualize/visualize_trafficmot.py` renders MP4s with GT boxes, track IDs, class names, and trajectory trails (trails only where multi-frame GT exists).

```bash
.venv/bin/python scripts/visualize/visualize_trafficmot.py /media/7TBSSD/data/tracking/TrafficMOT
```

Default output: `/media/7TBSSD/data/tracking/TrafficMOT_visualizations/{Fully_annotate,FirstFrame_annotate}/`. Useful flags: `--split`, `--sequences`, `--max-sequences`.

## CityFlow

Dataset root: `/media/7TBSSD/data/tracking/CityFlow` (AIC22 / CityFlowV2 MTMC vehicle tracking).

Structure (see local `ReadMe.txt`):

- `{train,validation,test}/<scenario>/<camera>/vdo.avi` — camera video (typically 10 FPS; `c015` is 8 FPS)
- `.../gt/gt.txt` — MOTChallenge GT on **train** and **validation** only: `frame,id,left,top,width,height,1,-1,-1,-1`
- `.../roi.jpg` — ROI mask for annotated regions
- `.../det/`, `.../mtsc/` — baseline detections / single-camera tracks (not used by the viz script)
- `cam_framenum/`, `cam_timestamp/`, `cam_loc/`, `list_cam.txt`, `eval/` — metadata and MTMC eval

This copy has 36 train + 23 validation cameras with GT, and 6 test cameras without public GT. Only vehicles seen in ≥2 cameras are annotated.

### Visualization

`scripts/visualize/visualize_cityflow.py` overlays global track IDs and trajectory trails on each camera video. Cameras without `gt/gt.txt` (test) are skipped.

```bash
.venv/bin/python scripts/visualize/visualize_cityflow.py /media/7TBSSD/data/tracking/CityFlow
```

Default output: `/media/7TBSSD/data/tracking/CityFlow_visualizations/{train,validation}/<scenario>/<camera>.mp4`. Useful flags: `--split`, `--sequences`, `--show-roi`, `--max-sequences`, `--max-frames`.

## FastTracker-Benchmark

Dataset root: `/media/7TBSSD/data/tracking/FastTracker-Benchmark`  
Source: [Hamidreza-Hashemp/FastTracker-Benchmark](https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark) (~19 GB of zips).

Structure after extract:

- `train/task_xxx.zip` → `train/task_xxx/{img1,gt,video,seqinfo.ini}`
- `gt/gt.txt` — MOT-style: `frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility`
- `gt/labels.txt` — class names by line order (1-indexed); class 9 is `ignore_region`
- `seqinfo.ini` — name, frameRate (typically 30), seqLength, resolution
- 12 task sequences (day/night turns, occlusions, tunnel, pedestrian crossing, …)

Download:

```bash
.venv/bin/hf download Hamidreza-Hashemp/FastTracker-Benchmark \
  --repo-type dataset \
  --local-dir /media/7TBSSD/data/tracking/FastTracker-Benchmark
```

### Visualization

`scripts/visualize/visualize_fasttracker.py` overlays track IDs, class names, and trajectory trails. Pass `--extract` once to unzip `train/*.zip` into `train/<seq>/`.

```bash
.venv/bin/python scripts/visualize/visualize_fasttracker.py /media/7TBSSD/data/tracking/FastTracker-Benchmark --extract
```

Default output: `/media/7TBSSD/data/tracking/FastTracker-Benchmark_visualizations/<seq>.mp4`. Useful flags: `--sequences`, `--show-ignore`, `--max-sequences`, `--max-frames`.

## FastTracker (tracker)

Cloned at project root: `FastTracker/` (upstream [Hamidreza-Hashempoor/FastTracker](https://github.com/Hamidreza-Hashempoor/FastTracker), YOLOX + ByteTrack + occlusion/ROI logic).

### How detections reach the tracker

The stock pipeline (`tools/track.py` → `yolox/evaluators/mot_evaluator.py`) runs YOLOX per frame, then calls:

```
online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
```

`Fasttracker.update(output_results, img_info, img_size)` (in `yolox/tracker/fasttracker.py`) is the only place detections enter. It accepts two shapes:

- **`[N, 5]` numpy** → `[x1, y1, x2, y2, score]` (no `.cpu()` — pure numpy/CPU path).
- **`[N, 7]` torch** (YOLOX) → `[x1, y1, x2, y2, obj_conf, cls_conf, cls]`; score = `obj_conf*cls_conf`.

It then rescales: `scale = min(img_size[0]/img_h, img_size[1]/img_w); bboxes /= scale`. So if you pass detections in **original image pixels** and set `img_size == (img_h, img_w)`, then `scale == 1` and boxes are used as-is. This is how we bypass YOLOX entirely.

Class-aware variant `yolox/tracker/fasttracker_cls.py` reads class from the last column; feed it a `[N, 7]` torch tensor `[x1,y1,x2,y2,score,1.0,class]` (its `[N,5]` path mislabels class as score).

### Running from precomputed detections (no YOLOX / no GPU)

`FastTracker/tools/track_from_dets.py` loads MOT-style detections and drives the tracker directly, writing MOT-format results — no detector, no weights, CPU-only.

Detection input (one line per detection): `frame, id, bb_left, bb_top, bb_width, bb_height, conf, [class], [visibility]` (`id` ignored; `conf<0` → `--conf-default`). Works with FastTracker-Benchmark `gt/gt.txt` (as oracle dets), CityFlow `det/det_*.txt`, or any MOT `det/det.txt`.

```bash
cd FastTracker
PYTHONPATH=. /mnt/nas/Users/ibrahim/tracking/.venv/bin/python tools/track_from_dets.py \
  /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
  --det-source gt --config configs/004_default.json
```

Results go to `<dataset_root>/../fasttracker_results/track_results/<seq>.txt`. Flags: `--sequences`, `--det-source {gt,det,file}` + `--det-name`, `--drop-classes` / `--keep-classes` (filter by class id before tracking), `--class-aware`, `--min-box-area`, `--drop-vertical` (OFF by default — the upstream `w/h>1.6` filter discards wide vehicles), `--conf-default`.

### Verified end-to-end on UA-DETRAC (GT boxes as oracle detections)

`scripts/convert/prepare_ua_detrac_mot.py` (legacy one-seq helper; prefer `-m mot_pipeline.run convert --benchmark ua_detrac` for bulk) converts one UA-DETRAC sequence into a MOT-style folder (`seqinfo.ini`, `gt/gt.txt`, `img1` symlink) under `/media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/<SEQ>/`:

```bash
.venv/bin/python scripts/convert/prepare_ua_detrac_mot.py MVI_39811
```

Then run the tracker (uses `configs/detrac_no_roi.json` — the bundled `004/005` configs carry ROIs tuned for a 1920×1080 scene and misfire on 960×540 DETRAC):

```bash
cd FastTracker
PYTHONPATH=. /mnt/nas/Users/ibrahim/tracking/.venv/bin/python tools/track_from_dets.py \
  /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker \
  --sequences MVI_39811 --det-source gt --config configs/detrac_no_roi.json \
  --output-dir /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/fasttracker_results
```

Overlay the tracker output on the frames with the generic MOT renderer:

```bash
.venv/bin/python scripts/visualize/visualize_mot_results.py \
  --frames /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/MVI_39811/img1 \
  --results /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/fasttracker_results/MVI_39811.txt \
  --out /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/MVI_39811_tracked.mp4
```

Note: UA-DETRAC XMLs annotate only a subset of frames (e.g. MVI_39811 covers frames 1–786 of 1070), so tracking spans the annotated range. GT-as-detections is an *oracle* test (perfect boxes) that validates the association/occlusion pipeline; swap in a real detector's `det.txt` for a fair benchmark.

### Verified end-to-end on FastTracker-Benchmark (all 12 sequences)

The benchmark ships extracted MOT layout already — `train/<seq>/{img1, gt/gt.txt, seqinfo.ini}` — so `gt/gt.txt` feeds `track_from_dets.py` directly with **no converter**. Two benchmark specifics:

- **Every sequence has a per-frame `ignore_region` (class 9)** — one box/frame. Drop it with `--drop-classes 9`, or it becomes a bogus static track.
- Scenes vary per task (day/night, tunnel, far objects up to 1920×1080), so the bundled scene-specific ROI configs don't apply. Use `configs/fasttracker_bench.json` (empty `ROIs`, lower `min_box_area: 10` to keep far objects, `track_buffer: 45` to survive occlusion).

```bash
cd FastTracker
PYTHONPATH=. /mnt/nas/Users/ibrahim/tracking/.venv/bin/python tools/track_from_dets.py \
  /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
  --det-source gt --drop-classes 9 \
  --config configs/fasttracker_bench.json
```

All 12 sequences run on CPU (~88 s total; the dense `task_day_occlusion`, ~47 obj/frame, is ~12 s). On that oracle run the tracker emits 203 IDs vs 193 GT IDs (1.05× — low fragmentation), covering all 1780 frames.

**Tuning knobs** (edit `configs/fasttracker_bench.json`): `track_thresh`, `match_thresh` (association IoU), `track_buffer` (frames a lost track survives), the occlusion block (`enlarge_bbox_occ`, `dampen_motion_occ`, `active_occ_to_lost_thresh`, `reset_*_offset_occ`), and per-scene `ROIs`. Add `--class-aware` to use the class-aware KF (`fasttracker_cls`).

**Oracle caveat (important for "the hard cases"):** feeding `gt/gt.txt` as detections gives *perfect* boxes with no misses, so association is near-trivial and this mainly validates the pipeline under high density. The benchmark's real difficulty (occlusion-induced detection gaps, night/tunnel/far-object misses) only shows up with a **real detector's `det.txt`**. To genuinely tune/evaluate, run a multi-class detector over the frames, write `det/det.txt` per sequence, then run with `--det-source det` and score against `gt/gt.txt` via TrackEval.

### Real detections with YOLO (`scripts/detect/run_yolo_dets.py`)

Legacy standalone detector. Prefer `-m mot_pipeline.run detect --detector yolov8` for the shared `detections/` cache. The helper writes per-sequence `det/det.txt` in MOT format (`frame,-1,x,y,w,h,conf,class,-1`).

```bash
.venv/bin/python scripts/detect/run_yolo_dets.py \
 /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
 --weights /media/7TBSSD/data/tracking/weights/yolov8m-expert_eff-1_2.pt \
 --device cuda:0 --imgsz 1280 --conf 0.25
```

Then track on those detections (no `--drop-classes` needed — YOLO emits only kept classes, no `ignore_region`):

```bash
cd FastTracker
PYTHONPATH=. ../.venv/bin/python tools/track_from_dets.py \
  /media/7TBSSD/data/tracking/FastTracker-Benchmark/train \
  --det-source det --config configs/fasttracker_bench.json
```

Notes / gotchas:
- **GPU**: this box has 8× V100. The venv now has CUDA torch (`torch==2.5.1+cu121`, driver 535 → cu121 wheels) plus its `nvidia-*-cu12` runtime wheels. `pip install torch` alone won't pull the CUDA build if a CPU torch is "already satisfied" — force it: `pip install --index-url .../cu121 --force-reinstall torch torchvision` **with** the `nvidia-*-cu12` + `triton==3.1.0` deps (a `--no-deps` install misses `libcudart.so.12`).
- Weights live on the SSD under `/media/7TBSSD/data/tracking/weights/` (keep them off the NAS). Default for the pipeline is `yolov8m-expert_eff-1_2.pt`.
- The expert model uses a **non-COCO** class map for bus/truck (`bus=4`, `truck=6`, plus `forklift=19`). Stock COCO `yolov8s.pt` keep-set is different — see `mot_pipeline/class_maps.py`.
- `det.txt` class column is the model class id (kept for class-aware tracking / eval mapping).

### Runtime deps (installed in the venv)

Importing/running the tracker needs: `torch` + `torchvision` (CPU wheels are enough for detection-fed tracking; `yolox/__init__.py` pulls in torchvision), `scipy`, `lap`, `cython_bbox` (built fine against numpy 2 via gcc), and `loguru`, `tabulate`, `thop` (imported by `yolox/__init__.py`). All installed. `motmetrics`/`filterpy` are only needed for the full YOLOX eval / motmetrics path.

### Evaluation

Prefer the unified pipeline below (`mot_pipeline`). The bundled TrackEval under `FastTracker/TrackEval/` is still what computes HOTA / CLEAR / Identity; `motmetrics` remains available for the upstream YOLOX path.

## MOT pipeline (`mot_pipeline/`)

Organized **benchmark × tracker × detector** runs with shared caches and TrackEval metrics.

### Artifacts (SSD)

```
/media/7TBSSD/data/tracking/
  mot/<Benchmark>/<split>/<seq>/{img1,gt/gt.txt,seqinfo.ini}
  detections/<benchmark>/<split>/<detector_id>/<seq>/det.txt
  experiments/<run_id>/{config.json,tracks/<seq>.txt,eval/summary.csv}
  experiments/_findings/<benchmark>/<tracker>/<detector_id>/{findings.csv,findings.json}
```

Detections are shared across trackers and tuning sweeps; each experiment freezes tracker config + writes tracks + metrics.
After every successful eval, the run is upserted into its combination-specific findings index. This keeps detector settings isolated (they are encoded in `detector_id`) while allowing tracker configurations to be compared as rows in one file. Re-evaluating the same `run_id` updates its row.

### Class policy (vehicles, motorcycles included)

Default keep-sets drop person / bicycle / ignore regions and **include** motorcycles. Pass `--exclude-motorcycles` to drop moto ids too. Maps live in `mot_pipeline/class_maps.py`. Eval rewrites kept GT boxes to class `1` and scores TrackEval's `person` slot as a **class-agnostic vehicle** channel (the bundled TrackEval is patched for FastTracker-Benchmark class names).

### CLI

```bash
# Ensure MOT layout (idempotent; pass --sequences to limit work)
.venv/bin/python -m mot_pipeline.run convert --benchmark fasttracker_bench

# Full run: detect → track → eval
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench \
  --tracker fasttracker \
  --detector yolov8 \
  --tracker-config mot_pipeline/configs/trackers/fasttracker/fasttracker_bench.json \
  --sequences task_day_occlusion \
  --device cuda:0

# Oracle GT detections (association-only stress test)
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench \
  --tracker fasttracker \
  --detector gt \
  --sequences task_day_occlusion

# Stage-wise
.venv/bin/python -m mot_pipeline.run detect --benchmark fasttracker_bench --detector yolov8 --device cuda:0
.venv/bin/python -m mot_pipeline.run track  --benchmark fasttracker_bench --tracker fasttracker --detector yolov8 \
  --tracker-config mot_pipeline/configs/trackers/fasttracker/fasttracker_bench.json
.venv/bin/python -m mot_pipeline.run eval --run-id <run_id>
```

Benchmarks: `fasttracker_bench`, `ua_detrac`, `trafficmot`, `cityflow`.  
Detectors: `gt`, `yolov8`, `existing` (reuse each sequence's `det/det.txt`), `yolox` (stub).  
Trackers: `fasttracker`, `ocsort`, `hybridsort`, `analytics_bytetrack` (motion-only wired); `traffictrack` (stub — see `TODOs.md`).

Default FastTracker configs (override with `--tracker-config`):
- `fasttracker_bench` → `configs/trackers/fasttracker/fasttracker_bench.json`
- `ua_detrac` → `detrac_no_roi.json`
- `trafficmot` / `cityflow` → `general_no_roi.json`

OC-SORT / HybridSORT defaults: `configs/trackers/ocsort/default.json` and `configs/trackers/hybridsort/default.json` (all benchmarks).

Analytics ByteTrack (live import from `analytics/analyzer_manager/app/tracking/`): default `configs/trackers/analytics_bytetrack/benchmark.json`; as-deployed settings in `production.json`. Shim lives in `mot_pipeline/trackers/analytics_shim.py` — no changes to the analytics clone.

Other benchmark defaults: UA-DETRAC / CityFlow → `--split train`; TrafficMOT → `--split Fully_annotate`. CityFlow sequence names are flattened (`S01_c001`).

### OC-SORT / HybridSORT (motion-only)

Clones live at project root: `OC_SORT/`, `HybridSORT/`. Same CLI as FastTracker — swap `--tracker`:

```bash
.venv/bin/python -m mot_pipeline.run all \
 --benchmark fasttracker_bench --tracker ocsort --detector existing \
 --sequences task_day_occlusion

.venv/bin/python -m mot_pipeline.run all \
 --benchmark fasttracker_bench --tracker hybridsort --detector existing \
 --sequences task_day_occlusion
```

### Analytics ByteTrack (in-house)

```bash
.venv/bin/python -m mot_pipeline.run all \
 --benchmark fasttracker_bench --tracker analytics_bytetrack --detector existing \
 --sequences task_day_occlusion
```

Hybrid-SORT-ReID, ByteSReid, and TrafficTrack are deferred — see [`TODOs.md`](TODOs.md).

### FastTracker on all four benchmarks

```bash
# 1) FastTracker-Benchmark — reuse the YOLO dets already under train/<seq>/det/det.txt
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker fasttracker --detector existing

# …or re-detect from scratch
.venv/bin/python -m mot_pipeline.run all \
  --benchmark fasttracker_bench --tracker fasttracker --detector yolov8 --device cuda:0

# 2) UA-DETRAC (oracle GT dets; no public detector cache yet)
.venv/bin/python -m mot_pipeline.run all \
  --benchmark ua_detrac --tracker fasttracker --detector gt

# …or run YOLO after convert (images are symlinked into mot/UA-DETRAC/)
.venv/bin/python -m mot_pipeline.run all \
  --benchmark ua_detrac --tracker fasttracker --detector yolov8 --device cuda:0

# 3) TrafficMOT (Fully_annotate)
.venv/bin/python -m mot_pipeline.run all \
  --benchmark trafficmot --tracker fasttracker --detector gt
.venv/bin/python -m mot_pipeline.run all \
  --benchmark trafficmot --tracker fasttracker --detector yolov8 --device cuda:0

# 4) CityFlow (convert extracts vdo.avi → img1; baseline dets linked as det/det.txt)
.venv/bin/python -m mot_pipeline.run convert --benchmark cityflow --sequences S01_c001
.venv/bin/python -m mot_pipeline.run all \
  --benchmark cityflow --tracker fasttracker --detector existing --sequences S01_c001
.venv/bin/python -m mot_pipeline.run all \
  --benchmark cityflow --tracker fasttracker --detector yolov8 --device cuda:0 --sequences S01_c001
```

`--detector existing` copies/filters sequence-local dets into `detections/.../existing_det_vehicles/` (use `--det-name det/det_yolo3.txt` for a CityFlow baseline file by name). `--detector yolov8` always writes a fresh cache under `detections/.../yolov8s_.../` and skips frames that are already cached unless `--force-detect`.

### Adding a tracker

1. Implement `mot_pipeline/trackers/<name>.py` with `track_sequence(seq_dir, det_path, out_path, config, extra=…)`.
2. Contract: MOT dets in **original image pixels** → MOT track txt `frame,id,x,y,w,h,conf,-1,-1,-1`.
3. Register in `mot_pipeline/trackers/__init__.py` (`TRACKERS` dict).
4. Drop hyperparameter JSONs under `mot_pipeline/configs/trackers/<name>/`.
5. See `mot_pipeline/trackers/fasttracker.py` as the reference adapter (no shelling out to upstream CLIs).

### Metrics

TrackEval writes `experiments/<run_id>/eval/summary.csv` + `summary.json` with per-sequence and `COMBINED` **HOTA** (DetA/AssA), **CLEAR** (MOTA/MOTP/IDSW/Frag/MT/ML/FP/FN), **Identity** (IDF1/IDP/IDR).
Cross-run indexes live at `experiments/_findings/<benchmark>/<tracker>/<detector_id>/findings.csv` (spreadsheet-friendly) and `findings.json` (full structured records).

## Conventions

- Prefer scripts under `scripts/<area>/` (`visualize/`, `convert/`, `detect/`, `batch/`, `analysis/`) with clear CLI args. Keep `mot_pipeline/` as the package; do not nest clones under `scripts/`.
- Dataset paths should be arguments, not hard-coded — defaults may point at the SSD paths above.
- Put generated videos, caches, and checkpoints next to the data on `/media/7TBSSD/data/tracking/`, not on the NAS project root.
- Batch sweeps: `./scripts/batch/run_all_gt_trackers.sh` (GT oracle) and `./scripts/batch/run_all_parallel.sh` (multi-GPU detect → track → eval).
- Findings tables: `.venv/bin/python scripts/analysis/compare_findings.py --detector-id …`.
- Do not commit secrets, raw dataset dumps, or the venv contents; `.venv` is a symlink only.
