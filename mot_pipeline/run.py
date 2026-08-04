#!/usr/bin/env python3
"""CLI: convert | detect | track | eval | all for MOT experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from mot_pipeline.class_maps import (
    class_space_for_yolo_weights,
    policy_for_benchmark,
    policy_for_coco,
    policy_for_yolo_weights,
)
from mot_pipeline.eval.prepare_trackeval import prepare_trackeval_layout
from mot_pipeline.eval.run_trackeval import run_trackeval
from mot_pipeline.findings import record_findings
from mot_pipeline.mot_io import write_json
from mot_pipeline.paths import (
    DEFAULT_YOLO_WEIGHTS,
    DETECTIONS_ROOT,
    EXPERIMENTS_ROOT,
    PIPELINE_ROOT,
)
from mot_pipeline.protocols import RunSpec
from mot_pipeline.registry import get_benchmark, get_detector, get_tracker
from mot_pipeline.trackers.analytics_bytetrack import DEFAULT_CFG as AB_DEFAULTS
from mot_pipeline.trackers.base import load_tracker_config
from mot_pipeline.trackers.fasttracker import DEFAULT_CFG as FT_DEFAULTS
from mot_pipeline.trackers.hybridsort import DEFAULT_CFG as HS_DEFAULTS
from mot_pipeline.trackers.ocsort import DEFAULT_CFG as OC_DEFAULTS


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cfg_tag(path: Optional[Path]) -> str:
    if path is None:
        return "default"
    return Path(path).stem


def make_run_id(
    benchmark: str,
    tracker: str,
    detector_id: str,
    cfg_path: Optional[Path],
    explicit: Optional[str] = None,
    *,
    target_fps: Optional[float] = None,
) -> str:
    if explicit:
        return explicit
    fps_tag = ""
    if target_fps is not None:
        # Keep the tag filesystem-friendly (10 → fps10, 7.5 → fps7p5).
        fps_txt = f"{float(target_fps):g}".replace(".", "p")
        fps_tag = f"_fps{fps_txt}"
    return f"{benchmark}_{tracker}_{detector_id}_{_cfg_tag(cfg_path)}{fps_tag}_{_ts()}"


def _default_tracker_config(tracker: str, benchmark: str) -> Optional[Path]:
    cfg_root = PIPELINE_ROOT / "configs" / "trackers"
    if tracker == "fasttracker":
        cfg_dir = cfg_root / "fasttracker"
        by_bench = {
            "fasttracker_bench": cfg_dir / "fasttracker_bench.json",
            "ua_detrac": cfg_dir / "detrac_no_roi.json",
            "trafficmot": cfg_dir / "general_no_roi.json",
            "cityflow": cfg_dir / "general_no_roi.json",
        }
        return by_bench.get(benchmark, cfg_dir / "general_no_roi.json")
    if tracker == "ocsort":
        return cfg_root / "ocsort" / "default.json"
    if tracker == "hybridsort":
        return cfg_root / "hybridsort" / "default.json"
    if tracker == "analytics_bytetrack":
        return cfg_root / "analytics_bytetrack" / "benchmark.json"
    return None


def _tracker_defaults(tracker: str) -> dict:
    if tracker == "fasttracker":
        return dict(FT_DEFAULTS)
    if tracker == "ocsort":
        return dict(OC_DEFAULTS)
    if tracker == "hybridsort":
        return dict(HS_DEFAULTS)
    if tracker == "analytics_bytetrack":
        return dict(AB_DEFAULTS)
    return {}


def _build_detector(args: argparse.Namespace, benchmark: str):
    if args.detector == "gt":
        return get_detector(
            "gt", benchmark=benchmark, exclude_motorcycles=args.exclude_motorcycles
        )
    if args.detector == "existing":
        return get_detector(
            "existing",
            benchmark=benchmark,
            exclude_motorcycles=args.exclude_motorcycles,
            det_name=getattr(args, "det_name", None) or "det/det.txt",
            class_space=getattr(args, "class_space", None) or "auto",
        )
    if args.detector == "yolov8":
        return get_detector(
            "yolov8",
            weights=Path(args.weights) if args.weights else DEFAULT_YOLO_WEIGHTS,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            batch=args.batch,
            device=args.device,
            half=args.half,
            exclude_motorcycles=args.exclude_motorcycles,
            max_frames=args.max_frames,
        )
    if args.detector == "yolox":
        return get_detector("yolox")
    raise SystemExit(f"Unknown detector: {args.detector}")


def _track_class_filter(args: argparse.Namespace, benchmark: str):
    """Class filter applied at track time (defense in depth; dets may already be filtered)."""
    if args.detector == "yolov8":
        weights = Path(args.weights) if args.weights else DEFAULT_YOLO_WEIGHTS
        return policy_for_yolo_weights(weights, args.exclude_motorcycles)
    if args.detector == "existing":
        space = getattr(args, "class_space", None) or "auto"
        if space == "coco" or (space == "auto" and benchmark == "fasttracker_bench"):
            return policy_for_coco(args.exclude_motorcycles)
        if space == "none" or (space == "auto" and benchmark == "cityflow"):
            return None, frozenset()
        return policy_for_benchmark(benchmark, args.exclude_motorcycles)
    return policy_for_benchmark(benchmark, args.exclude_motorcycles)


def _track_class_space(args: argparse.Namespace, benchmark: str) -> str:
    """Which class numbering the cached det.txt uses (see class_maps)."""
    if args.detector == "yolov8":
        weights = Path(args.weights) if args.weights else DEFAULT_YOLO_WEIGHTS
        return class_space_for_yolo_weights(weights)
    if args.detector == "existing":
        space = getattr(args, "class_space", None) or "auto"
        if space == "coco" or (space == "auto" and benchmark == "fasttracker_bench"):
            return "coco"
        if space == "none" or (space == "auto" and benchmark == "cityflow"):
            return "cityflow"
        return benchmark
    return benchmark


def _detections_dir(benchmark: str, split: str, detector_id: str) -> Path:
    return DETECTIONS_ROOT / benchmark / split / detector_id


def cmd_convert(args: argparse.Namespace) -> None:
    bench = get_benchmark(args.benchmark)
    split = args.split or bench.default_split()
    print(f"Converting {args.benchmark} split={split} ...")
    root = bench.ensure_mot(split, force=args.force, sequences=args.sequences)
    seqs = bench.sequence_dirs(split, args.sequences)
    print(f"MOT root: {root}  ({len(seqs)} sequences)")


def cmd_detect(args: argparse.Namespace) -> Path:
    bench = get_benchmark(args.benchmark)
    split = args.split or bench.default_split()
    bench.ensure_mot(split, force=False, sequences=args.sequences)
    seq_dirs = bench.sequence_dirs(split, args.sequences)
    if not seq_dirs:
        raise SystemExit("No sequences found.")
    det = _build_detector(args, args.benchmark)
    det_id = det.detector_id()
    out_root = (
        Path(args.detections_dir)
        if args.detections_dir
        else _detections_dir(args.benchmark, split, det_id)
    )
    print(f"Detect [{det.name}] → {out_root}")
    det.run(seq_dirs, out_root, force=args.force_detect)
    return out_root


def cmd_track(args: argparse.Namespace, detections_dir: Optional[Path] = None) -> Path:
    bench = get_benchmark(args.benchmark)
    split = args.split or bench.default_split()
    bench.ensure_mot(split, force=False, sequences=args.sequences)
    seq_dirs = bench.sequence_dirs(split, args.sequences)
    if not seq_dirs:
        raise SystemExit("No sequences found.")

    det = _build_detector(args, args.benchmark)
    det_id = det.detector_id()
    det_root = Path(detections_dir) if detections_dir else (
        Path(args.detections_dir)
        if args.detections_dir
        else _detections_dir(args.benchmark, split, det_id)
    )
    if not det_root.is_dir():
        raise SystemExit(
            f"Detections missing: {det_root}. Run `detect` first or use `all`."
        )

    cfg_path = (
        Path(args.tracker_config)
        if args.tracker_config
        else _default_tracker_config(args.tracker, args.benchmark)
    )
    defaults = _tracker_defaults(args.tracker)
    cfg = load_tracker_config(cfg_path, defaults)

    keep, drop = _track_class_filter(args, args.benchmark)
    target_fps = getattr(args, "fps", None)
    if target_fps is not None and float(target_fps) <= 0:
        raise SystemExit(f"--fps must be positive, got {target_fps}")
    extra = {
        "keep_classes": sorted(keep) if keep is not None else None,
        "drop_classes": sorted(drop),
        "class_space": _track_class_space(args, args.benchmark),
    }
    if target_fps is not None:
        extra["target_fps"] = float(target_fps)

    run_id = make_run_id(
        args.benchmark,
        args.tracker,
        det_id,
        cfg_path,
        args.run_id,
        target_fps=float(target_fps) if target_fps is not None else None,
    )
    exp_dir = EXPERIMENTS_ROOT / run_id
    tracks_dir = exp_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    spec_extra = {
        "detections_dir": str(det_root),
        "class_filter": {
            "keep_classes": extra["keep_classes"],
            "drop_classes": extra["drop_classes"],
            "class_space": extra["class_space"],
        },
    }
    if target_fps is not None:
        spec_extra["target_fps"] = float(target_fps)

    spec = RunSpec(
        run_id=run_id,
        benchmark=args.benchmark,
        split=split,
        tracker=args.tracker,
        detector=args.detector,
        detector_id=det_id,
        tracker_config_path=str(cfg_path) if cfg_path else None,
        tracker_config=cfg,
        sequences=[p.name for p in seq_dirs],
        exclude_motorcycles=args.exclude_motorcycles,
        extra=spec_extra,
    )
    write_json(exp_dir / "config.json", spec.to_dict())
    if cfg_path and cfg_path.is_file():
        shutil.copy2(cfg_path, exp_dir / "tracker_config.json")

    tracker_kwargs = {}
    if args.tracker == "fasttracker":
        tracker_kwargs["class_aware"] = getattr(args, "class_aware", False)
    tracker = get_tracker(args.tracker, **tracker_kwargs)
    fps_msg = f" @ {float(target_fps):g} FPS (frame subsample)" if target_fps is not None else ""
    print(f"Track [{tracker.name}]{fps_msg} → {tracks_dir}")
    for i, seq_dir in enumerate(seq_dirs, 1):
        det_path = det_root / seq_dir.name / "det.txt"
        if not det_path.is_file():
            # Also accept flat det.txt naming for convenience.
            alt = det_root / f"{seq_dir.name}.txt"
            if alt.is_file():
                det_path = alt
            else:
                raise FileNotFoundError(f"Missing detections for {seq_dir.name}: {det_path}")
        out_path = tracks_dir / f"{seq_dir.name}.txt"
        print(f"  [{i}/{len(seq_dirs)}] {seq_dir.name}")
        tracker.track_sequence(
            seq_dir,
            det_path,
            out_path,
            cfg,
            extra={k: v for k, v in extra.items() if v is not None},
        )

    print(f"Experiment: {exp_dir}")
    return exp_dir


def cmd_eval(args: argparse.Namespace, exp_dir: Optional[Path] = None) -> Path:
    exp_dir = Path(exp_dir or args.exp_dir or args.run_id or "")
    if args.run_id and not exp_dir.is_dir():
        exp_dir = EXPERIMENTS_ROOT / args.run_id
    if not exp_dir.is_dir():
        raise SystemExit(f"Experiment dir not found: {exp_dir}")

    cfg_path = exp_dir / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing config.json in {exp_dir}")
    with cfg_path.open() as f:
        spec = json.load(f)

    benchmark = args.benchmark or spec["benchmark"]
    split = args.split or spec.get("split")
    tracker_name = args.tracker or spec["tracker"]
    exclude_moto = bool(spec.get("exclude_motorcycles", False)) or bool(
        args.exclude_motorcycles
    )
    sequences = args.sequences or spec.get("sequences")

    bench = get_benchmark(benchmark)
    split = split or bench.default_split()
    bench.ensure_mot(split, force=False, sequences=sequences)
    seq_dirs = bench.sequence_dirs(split, sequences)
    tracks_dir = exp_dir / "tracks"

    eval_root = exp_dir / "eval" / "trackeval"
    print(f"Prepare TrackEval layout → {eval_root}")
    prepared = prepare_trackeval_layout(
        seq_dirs=seq_dirs,
        tracks_dir=tracks_dir,
        eval_root=eval_root,
        tracker_name=tracker_name,
        benchmark=benchmark,
        exclude_motorcycles=exclude_moto,
    )
    write_json(exp_dir / "eval" / "prepare_meta.json", {
        k: (str(v) if isinstance(v, Path) else v) for k, v in prepared.items()
    })

    out = exp_dir / "eval"
    print("Running TrackEval (HOTA / CLEAR / Identity) ...")
    summary = run_trackeval(
        gt_folder=prepared["gt_folder"],
        trackers_folder=prepared["trackers_folder"],
        tracker_name=tracker_name,
        seq_info=prepared["seq_info"],
        output_folder=out,
    )
    comb = summary.get("COMBINED", {})
    findings_dir = record_findings(
        spec=spec,
        combined_metrics=comb,
        experiment_dir=exp_dir,
    )
    print(
        "COMBINED:"
        f" HOTA={comb.get('HOTA', float('nan')):.3f}"
        f" MOTA={comb.get('MOTA', float('nan')):.3f}"
        f" IDF1={comb.get('IDF1', float('nan')):.3f}"
    )
    print(f"Wrote {out / 'summary.csv'} and {out / 'summary.json'}")
    print(f"Updated findings index: {findings_dir / 'findings.csv'}")
    return exp_dir


def cmd_all(args: argparse.Namespace) -> Path:
    det_root = cmd_detect(args)
    exp_dir = cmd_track(args, detections_dir=det_root)
    # Point eval at the new experiment.
    args.exp_dir = exp_dir
    args.run_id = exp_dir.name
    cmd_eval(args, exp_dir=exp_dir)
    return exp_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MOT pipeline: convert / detect / track / eval / all"
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, *, need_tracker: bool = False) -> None:
        sp.add_argument(
            "--benchmark",
            required=True,
            choices=["fasttracker_bench", "ua_detrac", "trafficmot", "cityflow"],
        )
        sp.add_argument("--split", default=None, help="Benchmark split (default per adapter).")
        sp.add_argument("--sequences", nargs="+", default=None)
        sp.add_argument(
            "--exclude-motorcycles",
            action="store_true",
            help="Drop motorcycles/motorbikes from vehicle keep-set.",
        )
        if need_tracker:
            sp.add_argument(
                "--tracker",
                required=True,
                choices=[
                    "fasttracker",
                    "traffictrack",
                    "ocsort",
                    "hybridsort",
                    "analytics_bytetrack",
                ],
            )
            sp.add_argument(
                "--tracker-config",
                type=Path,
                default=None,
                help="Tracker hyperparameter JSON.",
            )
            sp.add_argument("--class-aware", action="store_true")
            sp.add_argument("--run-id", default=None)
            sp.add_argument(
                "--detector",
                required=True,
                choices=["gt", "yolov8", "existing", "yolox"],
            )
            sp.add_argument("--detections-dir", type=Path, default=None)
            sp.add_argument(
                "--det-name",
                default="det/det.txt",
                help="Relative detection path inside each sequence "
                "(used by --detector existing).",
            )
            sp.add_argument(
                "--class-space",
                choices=("auto", "coco", "benchmark", "none"),
                default="auto",
                help="How to filter classes when importing existing dets.",
            )
            sp.add_argument("--weights", type=Path, default=None)
            sp.add_argument("--device", default=None)
            sp.add_argument("--imgsz", type=int, default=1280)
            sp.add_argument("--conf", type=float, default=0.25)
            sp.add_argument("--iou", type=float, default=0.7)
            sp.add_argument("--batch", type=int, default=16)
            sp.add_argument("--half", action="store_true")
            sp.add_argument("--max-frames", type=int, default=None)
            sp.add_argument("--force-detect", action="store_true")
            sp.add_argument(
                "--fps",
                type=float,
                default=None,
                help=(
                    "Target tracking FPS: subsample cached detections by "
                    "stride≈round(seq_fps/target) from seqinfo.ini frameRate. "
                    "Does not re-run detection. Omit to track every frame."
                ),
            )

    sp_c = sub.add_parser("convert", help="Ensure MOT layout for a benchmark.")
    add_common(sp_c)
    sp_c.add_argument("--force", action="store_true")
    sp_c.set_defaults(func=cmd_convert)

    sp_d = sub.add_parser("detect", help="Run / cache detections.")
    add_common(sp_d, need_tracker=False)
    sp_d.add_argument(
        "--detector",
        required=True,
        choices=["gt", "yolov8", "existing", "yolox"],
    )
    sp_d.add_argument("--detections-dir", type=Path, default=None)
    sp_d.add_argument("--det-name", default="det/det.txt")
    sp_d.add_argument(
        "--class-space",
        choices=("auto", "coco", "benchmark", "none"),
        default="auto",
    )
    sp_d.add_argument("--weights", type=Path, default=None)
    sp_d.add_argument("--device", default=None)
    sp_d.add_argument("--imgsz", type=int, default=1280)
    sp_d.add_argument("--conf", type=float, default=0.25)
    sp_d.add_argument("--iou", type=float, default=0.7)
    sp_d.add_argument("--batch", type=int, default=16)
    sp_d.add_argument("--half", action="store_true")
    sp_d.add_argument("--max-frames", type=int, default=None)
    sp_d.add_argument("--force-detect", action="store_true")
    sp_d.set_defaults(func=cmd_detect)

    sp_t = sub.add_parser("track", help="Run tracker on cached detections.")
    add_common(sp_t, need_tracker=True)
    sp_t.set_defaults(func=cmd_track)

    sp_e = sub.add_parser("eval", help="Score an experiment with TrackEval.")
    sp_e.add_argument("--run-id", default=None, help="Experiment id under experiments/.")
    sp_e.add_argument("--exp-dir", type=Path, default=None)
    sp_e.add_argument("--benchmark", default=None)
    sp_e.add_argument("--split", default=None)
    sp_e.add_argument("--tracker", default=None)
    sp_e.add_argument("--sequences", nargs="+", default=None)
    sp_e.add_argument("--exclude-motorcycles", action="store_true")
    sp_e.set_defaults(func=cmd_eval)

    sp_a = sub.add_parser("all", help="detect → track → eval")
    add_common(sp_a, need_tracker=True)
    sp_a.set_defaults(func=cmd_all)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
