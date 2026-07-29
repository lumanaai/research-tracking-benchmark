"""Shared MOTChallenge file helpers."""

from __future__ import annotations

import configparser
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def parse_seqinfo(seq_dir: Path) -> Dict[str, str]:
    ini = seq_dir / "seqinfo.ini"
    if not ini.is_file():
        return {}
    parser = configparser.ConfigParser()
    # Preserve MOTChallenge key casing (frameRate, imWidth, …).
    parser.optionxform = str  # type: ignore[method-assign]
    parser.read(ini)
    return dict(parser["Sequence"]) if "Sequence" in parser else {}


def write_seqinfo(
    path: Path,
    *,
    name: str,
    frame_rate: float,
    seq_length: int,
    im_width: int,
    im_height: int,
    im_ext: str = ".jpg",
    im_dir: str = "img1",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Sequence]\n"
        f"name={name}\n"
        f"imDir={im_dir}\n"
        f"frameRate={frame_rate}\n"
        f"seqLength={seq_length}\n"
        f"imWidth={im_width}\n"
        f"imHeight={im_height}\n"
        f"imExt={im_ext}\n"
    )


def discover_mot_sequences(
    root: Path, wanted: Optional[Sequence[str]] = None
) -> List[Path]:
    seqs: List[Path] = []
    if not root.is_dir():
        return seqs
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name == "__MACOSX":
            continue
        if wanted is not None and p.name not in wanted:
            continue
        if (p / "seqinfo.ini").is_file() or (p / "gt" / "gt.txt").is_file():
            seqs.append(p)
    return seqs


def frame_index(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def list_frames(img_dir: Path) -> List[Path]:
    frames = sorted(img_dir.glob("*.jpg"), key=frame_index)
    if not frames:
        frames = sorted(img_dir.glob("*.png"), key=frame_index)
    return frames


def filter_mot_lines(
    lines: Iterable[str],
    *,
    keep: Optional[Set[int]] = None,
    drop: Optional[Set[int]] = None,
    rewrite_class: Optional[int] = None,
    conf_default: float = 1.0,
    force_conf: Optional[float] = None,
) -> List[str]:
    """Filter MOT rows by class column; optionally rewrite class / conf.

    Expected columns: frame,id,x,y,w,h,conf[,class[,visibility]]
    Rows without a class column are kept (and get ``rewrite_class`` if set).
    """
    drop_set = drop or set()
    out: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 6:
            continue
        has_cls = len(parts) >= 8
        cls = int(float(parts[7])) if has_cls else None
        if cls is not None and cls in drop_set:
            continue
        if keep is not None and cls is not None and cls not in keep:
            continue
        conf = float(parts[6]) if len(parts) >= 7 else conf_default
        if conf < 0:
            conf = conf_default
        if force_conf is not None:
            conf = force_conf
        out_cls = rewrite_class if rewrite_class is not None else (cls if cls is not None else 1)
        vis = parts[8] if len(parts) >= 9 else "1"
        out.append(
            f"{int(float(parts[0]))},{int(float(parts[1]))},"
            f"{float(parts[2]):.2f},{float(parts[3]):.2f},"
            f"{float(parts[4]):.2f},{float(parts[5]):.2f},"
            f"{conf:.4f},{out_cls},{vis}"
        )
    return out


def filter_mot_file(
    src: Path,
    dst: Path,
    *,
    keep: Optional[Set[int]] = None,
    drop: Optional[Set[int]] = None,
    rewrite_class: Optional[int] = None,
    conf_default: float = 1.0,
    force_conf: Optional[float] = None,
) -> int:
    with src.open() as f:
        lines = filter_mot_lines(
            f,
            keep=keep,
            drop=drop,
            rewrite_class=rewrite_class,
            conf_default=conf_default,
            force_conf=force_conf,
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def load_mot_dets(
    det_path: Path,
    conf_default: float = 1.0,
    keep: Optional[Set[int]] = None,
    drop: Optional[Set[int]] = None,
) -> Dict[int, List[Tuple[float, float, float, float, float, int]]]:
    """Return {frame: [(x1,y1,x2,y2,score,class), ...]}."""
    drop_set = drop or set()
    per_frame: Dict[int, List[Tuple[float, float, float, float, float, int]]] = defaultdict(
        list
    )
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
            if cls in drop_set:
                continue
            if keep is not None and cls not in keep:
                continue
            per_frame[frame].append((x, y, x + w, y + h, conf, cls))
    return per_frame


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def read_json(path: Path) -> dict:
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return {}
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Concurrent writers can briefly leave an incomplete file.
        return {}
