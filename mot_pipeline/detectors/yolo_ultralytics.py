"""Ultralytics YOLO detector writing shared MOT det.txt cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

from mot_pipeline.class_maps import policy_for_yolo_weights
from mot_pipeline.mot_io import frame_index, list_frames, read_json, write_json
from mot_pipeline.paths import DEFAULT_YOLO_WEIGHTS
from mot_pipeline.protocols import Detector

Imgsz = Union[int, Tuple[int, int]]

# In-house product YOLO settings (expert_eff). imgsz is Ultralytics (H, W).
DEFAULT_YOLO_IMGSZ: Imgsz = (704, 1280)
DEFAULT_YOLO_CONF = 0.3
DEFAULT_YOLO_IOU = 0.7  # Ultralytics NMS; production NMS IoU not confirmed


def parse_imgsz(value: object) -> Imgsz:
    """Accept 1280, (704, 1280), or '704x1280' / '704,1280'."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, int):
        return value
    text = str(value).strip().lower().replace(",", "x")
    if "x" in text:
        h, w = text.split("x", 1)
        return (int(h), int(w))
    return int(text)


def imgsz_tag(imgsz: Imgsz) -> str:
    if isinstance(imgsz, tuple):
        return f"{imgsz[0]}x{imgsz[1]}"
    return str(int(imgsz))


class YoloUltralyticsDetector(Detector):
    name = "yolov8"

    def __init__(
        self,
        weights: Optional[Path] = None,
        imgsz: Imgsz = DEFAULT_YOLO_IMGSZ,
        conf: float = DEFAULT_YOLO_CONF,
        iou: float = DEFAULT_YOLO_IOU,
        batch: int = 16,
        device: Optional[str] = None,
        half: bool = False,
        exclude_motorcycles: bool = False,
        max_frames: Optional[int] = None,
        keep_classes: Optional[Sequence[int]] = None,
    ) -> None:
        self.weights = Path(weights or DEFAULT_YOLO_WEIGHTS)
        self.imgsz = parse_imgsz(imgsz)
        self.conf = conf
        self.iou = iou
        self.batch = batch
        self.device = device
        self.half = half
        self.exclude_motorcycles = exclude_motorcycles
        self.max_frames = max_frames
        self.keep_classes = (
            [int(c) for c in keep_classes] if keep_classes is not None else None
        )

    def detector_id(self, **kwargs: Any) -> str:
        stem = self.weights.stem
        moto = "nomoto" if self.exclude_motorcycles else "vehicles"
        return f"{stem}_imgsz{imgsz_tag(self.imgsz)}_conf{self.conf:g}_{moto}"

    def _keep_drop(self) -> tuple:
        if self.keep_classes is not None:
            return frozenset(self.keep_classes), frozenset()
        return policy_for_yolo_weights(self.weights, self.exclude_motorcycles)

    def _meta(self) -> dict:
        keep, drop = self._keep_drop()
        return {
            "detector": self.name,
            "detector_id": self.detector_id(),
            "weights": str(self.weights),
            "imgsz": list(self.imgsz) if isinstance(self.imgsz, tuple) else self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "batch": self.batch,
            "device": self.device,
            "half": self.half,
            "exclude_motorcycles": self.exclude_motorcycles,
            "keep": sorted(keep) if keep is not None else None,
            "drop": sorted(drop),
        }

    def run(
        self,
        seq_dirs: Sequence[Path],
        out_root: Path,
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> Path:
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        meta = self._meta()
        meta_path = out_root / "meta.json"
        if meta_path.is_file() and not force:
            prev = read_json(meta_path)
            # Allow reuse if core knobs match. Empty prev → treat as missing.
            # Weights are compared by filename so models/foo.pt and
            # /media/.../weights/foo.pt share the same detection cache.
            if prev:
                keys = ("imgsz", "conf", "iou", "exclude_motorcycles", "keep")
                mismatch = any(prev.get(k) != meta.get(k) for k in keys)
                prev_w = Path(str(prev.get("weights") or "")).name
                cur_w = Path(str(meta.get("weights") or "")).name
                if mismatch or (prev_w and cur_w and prev_w != cur_w):
                    force = True
        # Atomic / idempotent: safe under multi-GPU sharded detect.
        if force or not meta_path.is_file() or not read_json(meta_path):
            write_json(meta_path, meta)

        keep, _drop = self._keep_drop()
        classes: Optional[List[int]] = sorted(keep) if keep is not None else None

        pending = []
        for seq_dir in seq_dirs:
            dst = out_root / seq_dir.name / "det.txt"
            tmp = dst.with_suffix(".txt.tmp")
            if tmp.is_file():
                # Killed mid-write — drop partial and redo.
                tmp.unlink()
            if dst.is_file() and not force:
                print(f"  [yolov8] skip cached {seq_dir.name}")
                continue
            pending.append(seq_dir)

        if not pending:
            return out_root

        from ultralytics import YOLO

        model = YOLO(str(self.weights))
        for seq_dir in pending:
            n = self._run_sequence(model, seq_dir, out_root / seq_dir.name / "det.txt", classes)
            print(f"  [yolov8] {seq_dir.name}: {n} dets")
        return out_root

    def _run_sequence(
        self,
        model,
        seq_dir: Path,
        out_path: Path,
        classes: Optional[List[int]],
    ) -> int:
        frames = list_frames(seq_dir / "img1")
        if not frames:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("")
            return 0
        if self.max_frames is not None:
            frames = frames[: self.max_frames]

        predict_kwargs = dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=classes,
            verbose=False,
        )
        if self.half:
            predict_kwargs["half"] = True

        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_det = 0
        tmp_path = out_path.with_suffix(".txt.tmp")
        with tmp_path.open("w") as f:
            for start in range(0, len(frames), self.batch):
                chunk = frames[start : start + self.batch]
                results = model.predict(source=[str(p) for p in chunk], **predict_kwargs)
                for fp, res in zip(chunk, results):
                    fidx = frame_index(fp)
                    boxes = res.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                        w, h = x2 - x1, y2 - y1
                        f.write(
                            f"{fidx},-1,{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{c:.4f},{int(k)},-1\n"
                        )
                        n_det += 1
        tmp_path.replace(out_path)
        return n_det
