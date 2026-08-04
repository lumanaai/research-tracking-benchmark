#!/usr/bin/env python3
"""Run FastTracker directly from precomputed detections (no YOLOX, no GPU needed).

Why this exists
---------------
The stock pipeline (``tools/track.py`` + ``run_mot17.sh``) runs a YOLOX detector
on every frame and feeds the detections to the tracker inside
``yolox/evaluators/mot_evaluator.py``::

    outputs = model(imgs)                      # YOLOX forward
    outputs = postprocess(outputs, ...)        # NMS -> [x1,y1,x2,y2,obj,cls_conf,cls]
    online_targets = tracker.update(outputs[0], info_imgs, self.img_size)

The only thing the tracker actually consumes is that detection array. Looking at
``Fasttracker.update(output_results, img_info, img_size)``:

    if output_results.shape[1] == 5:           # <-- numpy path, NO .cpu()
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]         # x1,y1,x2,y2
    else:                                      # torch path (YOLOX 7-col)
        output_results = output_results.cpu().numpy()
        scores = output_results[:, 4] * output_results[:, 5]
        bboxes = output_results[:, :4]
    ...
    scale = min(img_size[0] / img_h, img_size[1] / img_w)
    bboxes /= scale                            # map model-space -> image-space

So we can bypass YOLOX completely by handing the tracker a plain
``[N, 5]`` numpy array ``[x1, y1, x2, y2, score]`` in ORIGINAL image pixels, and
passing ``img_size == (img_h, img_w)`` so that ``scale == 1`` and the boxes are
used as-is. This script does exactly that.

Detection input format (MOTChallenge ``det.txt`` style, one line per detection)::

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, [class], [visibility]

- ``id`` is ignored (use -1 for pure detections).
- ``conf`` is the detection score; if missing/negative, ``--conf-default`` is used.
- ``class`` is only used in ``--class-aware`` mode.

This is compatible with:
- FastTracker-Benchmark ``gt/gt.txt`` (use ``--det-source gt`` to treat GT boxes
  as oracle detections),
- CityFlow ``det/det_*.txt``,
- any MOT17/MOT20-style ``det/det.txt``.

NOTE: This script only *builds* the pipeline; running it performs tracking.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

# Defaults mirror yolox/evaluators/mot_evaluator.py::DEFAULT_CFG
DEFAULT_CFG: Dict[str, object] = {
    "track_thresh": 0.6,
    "track_buffer": 30,
    "match_thresh": 0.9,
    "min_box_area": 100,
    "reset_velocity_offset_occ": 5,
    "reset_pos_offset_occ": 3,
    "enlarge_bbox_occ": 1.2,
    "dampen_motion_occ": 0.85,
    "active_occ_to_lost_thresh": 15,
    "init_iou_suppress": 0.8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run FastTracker from precomputed detections (bypasses YOLOX)."
    )
    p.add_argument(
        "dataset_root",
        type=Path,
        help="Folder of sequences; each <seq>/ has seqinfo.ini and a detection file.",
    )
    p.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional subset of sequence folder names to run.",
    )
    p.add_argument(
        "--det-source",
        choices=("gt", "det", "file"),
        default="det",
        help="Where per-sequence detections live: gt/gt.txt, det/det.txt, or a "
        "custom relative path given by --det-name (default: det).",
    )
    p.add_argument(
        "--det-name",
        default=None,
        help="Relative path (within each sequence) to the detection file. "
        "Defaults: gt->gt/gt.txt, det->det/det.txt.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a tracking config JSON (thresholds/ROIs). Falls back to defaults.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write MOT result txts. Default: <dataset_root>/../"
        "fasttracker_results/track_results",
    )
    p.add_argument(
        "--conf-default",
        type=float,
        default=1.0,
        help="Score to use when the detection file has no/negative conf (e.g. GT).",
    )
    p.add_argument(
        "--min-box-area",
        type=float,
        default=None,
        help="Override min_box_area result filter (default: from config).",
    )
    p.add_argument(
        "--drop-vertical",
        action="store_true",
        help="Replicate the ByteTrack pedestrian filter that drops boxes with "
        "w/h > 1.6. OFF by default (that filter throws away wide vehicles).",
    )
    p.add_argument(
        "--frame-rate",
        type=int,
        default=None,
        help="Override frame rate (default: from seqinfo.ini, else 30).",
    )
    p.add_argument(
        "--drop-classes",
        nargs="+",
        type=int,
        default=None,
        help="Class ids to exclude from detections (e.g. 9 for FastTracker-Benchmark "
        "ignore_region). Applied before tracking.",
    )
    p.add_argument(
        "--keep-classes",
        nargs="+",
        type=int,
        default=None,
        help="If set, ONLY these class ids are kept as detections (applied after "
        "--drop-classes).",
    )
    p.add_argument(
        "--class-aware",
        action="store_true",
        help="Use fasttracker_cls (class-aware KF). Requires a class column in "
        "detections; builds a 7-col torch tensor the cls tracker expects.",
    )
    p.add_argument(
        "--mot20",
        action="store_true",
        help="Disable score fusion in association (matches ByteTrack MOT20 setting).",
    )
    return p.parse_args()


def load_config(path: Optional[Path]) -> Dict[str, object]:
    cfg = dict(DEFAULT_CFG)
    if path is not None and path.is_file():
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def parse_seqinfo(seq_dir: Path) -> Dict[str, str]:
    ini = seq_dir / "seqinfo.ini"
    if not ini.is_file():
        return {}
    parser = configparser.ConfigParser()
    parser.read(ini)
    return dict(parser["Sequence"]) if "Sequence" in parser else {}


def resolve_det_path(seq_dir: Path, det_source: str, det_name: Optional[str]) -> Path:
    if det_name:
        return seq_dir / det_name
    if det_source == "gt":
        return seq_dir / "gt" / "gt.txt"
    return seq_dir / "det" / "det.txt"


def load_detections(
    det_path: Path,
    conf_default: float,
    drop_classes: Optional[List[int]] = None,
    keep_classes: Optional[List[int]] = None,
) -> Dict[int, List[Tuple[float, float, float, float, float, int]]]:
    """Return {frame: [(x1, y1, x2, y2, score, class), ...]} in image pixels."""
    drop = set(drop_classes or [])
    keep = set(keep_classes) if keep_classes else None
    per_frame: Dict[int, List[Tuple[float, float, float, float, float, int]]] = defaultdict(list)
    with det_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[2:6])
            conf = float(parts[6]) if len(parts) >= 7 else conf_default
            if conf < 0:
                conf = conf_default
            cls = int(float(parts[7])) if len(parts) >= 8 else 0
            if cls in drop:
                continue
            if keep is not None and cls not in keep:
                continue
            per_frame[frame].append((x, y, x + w, y + h, conf, cls))
    return per_frame


def build_det_array(
    dets: List[Tuple[float, float, float, float, float, int]],
    class_aware: bool,
):
    """Build the array the tracker consumes.

    - default: [N, 5] float numpy -> [x1, y1, x2, y2, score]
    - class-aware: [N, 7] float torch -> [x1, y1, x2, y2, score, 1.0, class]
      (the cls tracker computes score = col4*col5 and reads class from col[-1]).
    """
    if not class_aware:
        if not dets:
            return np.empty((0, 5), dtype=np.float32)
        return np.array([[d[0], d[1], d[2], d[3], d[4]] for d in dets], dtype=np.float32)

    import torch  # local import; only needed for class-aware mode

    if not dets:
        return torch.zeros((0, 7), dtype=torch.float32)
    arr = np.array(
        [[d[0], d[1], d[2], d[3], d[4], 1.0, d[5]] for d in dets], dtype=np.float32
    )
    return torch.from_numpy(arr)


def write_results(path: Path, results: List[Tuple[int, list, list, list]]) -> None:
    fmt = "{frame},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},{s:.2f},-1,-1,-1\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for frame_id, tlwhs, ids, scores in results:
            for tlwh, tid, score in zip(tlwhs, ids, scores):
                if tid < 0:
                    continue
                x, y, w, h = tlwh
                f.write(fmt.format(frame=frame_id, tid=tid, x=x, y=y, w=w, h=h, s=score))


def run_sequence(
    seq_dir: Path,
    out_path: Path,
    cfg: Dict[str, object],
    args: argparse.Namespace,
) -> int:
    # Imports are local so merely creating the file / --help does not require torch.
    if args.class_aware:
        from yolox.tracker.fasttracker_cls import Fasttracker
    else:
        from yolox.tracker.fasttracker import Fasttracker

    meta = parse_seqinfo(seq_dir)
    img_w = int(meta.get("imWidth", 0)) or None
    img_h = int(meta.get("imHeight", 0)) or None
    seq_len = int(meta.get("seqLength", 0)) or None
    frame_rate = args.frame_rate or int(float(meta.get("frameRate", 30)))

    det_path = resolve_det_path(seq_dir, args.det_source, args.det_name)
    if not det_path.is_file():
        raise FileNotFoundError(f"No detection file for {seq_dir.name}: {det_path}")
    per_frame = load_detections(
        det_path, args.conf_default, args.drop_classes, args.keep_classes
    )

    # Fall back to detection extent if seqinfo is missing image size.
    if img_h is None or img_w is None:
        max_x = max((d[2] for dl in per_frame.values() for d in dl), default=1.0)
        max_y = max((d[3] for dl in per_frame.values() for d in dl), default=1.0)
        img_w = img_w or int(np.ceil(max_x))
        img_h = img_h or int(np.ceil(max_y))

    frames = sorted(per_frame)
    if seq_len is None:
        seq_len = frames[-1] if frames else 0

    tracker_args = SimpleNamespace(mot20=args.mot20)
    tracker = Fasttracker(tracker_args, cfg, frame_rate=frame_rate)

    min_box_area = args.min_box_area if args.min_box_area is not None else cfg["min_box_area"]
    # scale == 1 because we pass img_size == (img_h, img_w): boxes stay in pixels.
    img_info = (img_h, img_w)
    img_size = (img_h, img_w)

    results: List[Tuple[int, list, list, list]] = []
    for frame_id in range(1, seq_len + 1):
        det_array = build_det_array(per_frame.get(frame_id, []), args.class_aware)
        online_targets = tracker.update(det_array, img_info, img_size)

        tlwhs, ids, scores = [], [], []
        for t in online_targets:
            tlwh = t.tlwh
            if tlwh[2] * tlwh[3] <= min_box_area:
                continue
            if args.drop_vertical and tlwh[2] / max(tlwh[3], 1e-6) > 1.6:
                continue
            tlwhs.append(tlwh)
            ids.append(t.track_id)
            scores.append(t.score)
        results.append((frame_id, tlwhs, ids, scores))

    write_results(out_path, results)
    return len(results)


def discover_sequences(dataset_root: Path, wanted: Optional[List[str]]) -> List[Path]:
    seqs = []
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name == "__MACOSX":
            continue
        if wanted is not None and p.name not in wanted:
            continue
        seqs.append(p)
    return seqs


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    cfg = load_config(args.config)
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root.parent / "fasttracker_results" / "track_results"
    )

    seqs = discover_sequences(dataset_root, args.sequences)
    if not seqs:
        raise SystemExit("No sequences found.")

    print(f"Dataset root : {dataset_root}")
    print(f"Detections   : {args.det_source} ({args.det_name or 'default path'})")
    print(f"Output dir   : {out_dir}")
    print(f"Class-aware  : {args.class_aware}")
    print(f"Sequences    : {len(seqs)}")

    for i, seq_dir in enumerate(seqs, start=1):
        out_path = out_dir / f"{seq_dir.name}.txt"
        print(f"[{i}/{len(seqs)}] {seq_dir.name} -> {out_path}")
        n = run_sequence(seq_dir, out_path, cfg, args)
        print(f"  wrote {n} frames")

    print("Done.")


if __name__ == "__main__":
    main()
