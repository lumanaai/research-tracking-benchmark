#!/usr/bin/env python3
"""Overlay a MOTChallenge results/GT file on a folder of frames -> MP4 or JPEGs.

Generic helper to eyeball any tracker output (e.g. FastTracker) or GT in MOT
format: `frame,id,bb_left,bb_top,bb_width,bb_height,...`.

Boxes are drawn in original image pixels, then the frame may be downscaled.
If ``out`` has no video suffix, writes a JPEG sequence instead (used by
``visualize_all.py`` tracker grids).

For a synced multi-tracker HTML grid over latest findings, use
``./scripts/batch/visualize_all.sh`` then ``./scripts/batch/serve_visualizations.sh``.

Example:
    .venv/bin/python scripts/visualize/visualize_mot_results.py \
        --frames /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/MVI_39811/img1 \
        --results /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/fasttracker_results/MVI_39811.txt \
        --out /media/7TBSSD/data/tracking/UA-DETRAC_fasttracker/MVI_39811_tracked.mp4
"""

from __future__ import annotations

import argparse
import colorsys
import shutil
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a MOT results file over frames.")
    p.add_argument("--frames", type=Path, required=True, help="Directory of frame images.")
    p.add_argument("--results", type=Path, required=True, help="MOT-format txt to overlay.")
    p.add_argument("--out", type=Path, required=True, help="Output .mp4 path.")
    p.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output video FPS (use native/stride to match subsampled tracking).",
    )
    p.add_argument("--trail-length", type=int, default=60)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Cap written frames to this many seconds at --fps (after stride).",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First 1-indexed MOT frame to write (among kept frames).",
    )
    p.add_argument(
        "--start-frac",
        type=float,
        default=0.0,
        help="Skip this fraction of kept frames before writing (ignored if --start-frame is set).",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Downscale after drawing so width <= this (height follows aspect).",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth image (1-indexed MOT frames). Use with --fps to match subsampled tracking.",
    )
    p.add_argument(
        "--trail-warmup",
        action="store_true",
        help="Build trails from kept frames before the snippet without writing them.",
    )
    p.add_argument(
        "--browser-mp4",
        action="store_true",
        help="Remux/transcode to H.264 when ffmpeg is available (HTML <video> playback).",
    )
    p.add_argument("--label", type=str, default="", help="Optional HUD prefix.")
    return p.parse_args()


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def frame_index(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def list_frame_paths(frames_dir: Path) -> List[Path]:
    frames = sorted(frames_dir.glob("*.jpg"), key=frame_index)
    if not frames:
        frames = sorted(frames_dir.glob("*.png"), key=frame_index)
    return frames


def load_mot(path: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    per_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c = line.split(",")
        f = int(float(c[0]))
        tid = int(float(c[1]))
        x, y, w, h = (float(v) for v in c[2:6])
        per_frame[f].append((tid, x, y, w, h))
    return per_frame


def apply_stride(frames: Sequence[Path], stride: int) -> List[Path]:
    step = max(1, int(stride))
    if step == 1:
        return [fp for fp in frames if frame_index(fp) > 0]
    return [
        fp
        for fp in frames
        if frame_index(fp) > 0 and (frame_index(fp) - 1) % step == 0
    ]


def snippet_window(
    kept: Sequence[Path],
    *,
    start_frame: Optional[int] = None,
    start_frac: float = 0.0,
    max_frames: Optional[int] = None,
    max_seconds: Optional[float] = None,
    writer_fps: float = 25.0,
) -> Tuple[List[Path], List[Path]]:
    """Split kept frames into (warmup, write) lists."""
    if not kept:
        return [], []
    if start_frame is not None:
        write_start = next(
            (i for i, fp in enumerate(kept) if frame_index(fp) >= int(start_frame)),
            0,
        )
    else:
        frac = min(max(float(start_frac), 0.0), 0.95)
        write_start = int(len(kept) * frac)
        if write_start >= len(kept):
            write_start = max(0, len(kept) - 1)

    n_write = len(kept) - write_start
    if max_seconds is not None and float(max_seconds) > 0 and writer_fps > 0:
        n_write = min(n_write, max(1, int(round(float(max_seconds) * float(writer_fps)))))
    if max_frames is not None:
        n_write = min(n_write, max(1, int(max_frames)))
    write = list(kept[write_start : write_start + n_write])
    warmup = list(kept[:write_start])
    return warmup, write


def _output_size(h: int, w: int, max_width: Optional[int]) -> Tuple[int, int, float]:
    if max_width and int(max_width) > 0 and w > int(max_width):
        out_w = int(max_width)
        out_h = max(1, int(round(h * (out_w / float(w)))))
    else:
        out_w, out_h = w, h
    if out_h % 2:
        out_h += 1
    if out_w % 2:
        out_w += 1
    return out_w, out_h, out_w / float(w)


def _thickness(h: int, w: int) -> int:
    return max(2, int(round(min(h, w) / 400.0)))


def draw_tracks(
    img: np.ndarray,
    dets: Sequence[Tuple[int, float, float, float, float]],
    trails: Dict[int, Deque[Tuple[int, int]]],
    *,
    trail_length: int,
    label: str,
    frame_idx: int,
) -> None:
    h, w = img.shape[:2]
    thick = _thickness(h, w)
    font_scale = max(0.4, min(h, w) / 1200.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    active = set()
    for tid, x, y, bw, bh in dets:
        active.add(tid)
        cx, cy = int(x + bw / 2), int(y + bh / 2)
        trails.setdefault(tid, deque(maxlen=trail_length)).append((cx, cy))
    for tid in list(trails):
        if tid not in active:
            del trails[tid]

    for tid, pts in trails.items():
        if len(pts) >= 2:
            cv2.polylines(
                img,
                [np.array(pts, np.int32).reshape(-1, 1, 2)],
                False,
                color_for_id(tid),
                thick,
                cv2.LINE_AA,
            )
    for tid, x, y, bw, bh in dets:
        c = color_for_id(tid)
        p1, p2 = (int(x), int(y)), (int(x + bw), int(y + bh))
        cv2.rectangle(img, p1, p2, c, thick)
        lbl = f"ID {tid}"
        (tw, th), bs = cv2.getTextSize(lbl, font, font_scale, max(1, thick - 1))
        ty = max(th + 2, int(y) - 4)
        cv2.rectangle(img, (int(x), ty - th - bs), (int(x) + tw + 4, ty + 2), c, -1)
        cv2.putText(img, lbl, (int(x) + 2, ty), font, font_scale, (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)

    hud = f"{label} frame {frame_idx}  tracks {len(dets)}".strip()
    cv2.putText(img, hud, (10, 26 + thick), font, font_scale * 1.4, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, hud, (10, 26 + thick), font, font_scale * 1.4, (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)


def open_video_writer(path: Path, fps: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".webm":
        candidates = ("vp09", "VP90", "VP80")
    else:
        candidates = ("avc1", "H264", "X264", "mp4v")
    last_err = ""
    for fourcc_name in candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), float(fps), size)
        if writer.isOpened():
            return writer, fourcc_name
        writer.release()
        last_err = fourcc_name
    raise RuntimeError(f"Could not open VideoWriter for {path} (last tried {last_err})")


def ensure_browser_mp4(path: Path) -> bool:
    """Transcode to H.264 + yuv420p + faststart when ffmpeg is on PATH."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not path.is_file():
        return False
    tmp = path.with_suffix(".h264.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.replace(path)
    return True


@dataclass
class RenderResult:
    written: int
    out_path: Path
    writer_fps: float
    width: int
    height: int
    start_mot_frame: Optional[int]
    end_mot_frame: Optional[int]


def render_overlay(
    frames_dir: Path,
    results_path: Path,
    out_path: Path,
    *,
    fps: float = 25.0,
    trail_length: int = 60,
    max_frames: Optional[int] = None,
    max_seconds: Optional[float] = None,
    start_frame: Optional[int] = None,
    start_frac: float = 0.0,
    max_width: Optional[int] = None,
    frame_stride: int = 1,
    trail_warmup: bool = False,
    browser_mp4: bool = False,
    label: str = "",
) -> RenderResult:
    frames = list_frame_paths(frames_dir)
    if not frames:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    if not results_path.is_file():
        raise FileNotFoundError(f"No track file: {results_path}")

    kept = apply_stride(frames, frame_stride)
    if not kept:
        raise FileNotFoundError(f"No frames left after --frame-stride {frame_stride}")

    warmup, write = snippet_window(
        kept,
        start_frame=start_frame,
        start_frac=start_frac,
        max_frames=max_frames,
        max_seconds=max_seconds,
        writer_fps=fps,
    )
    if not write:
        raise RuntimeError(f"Snippet window is empty for {frames_dir}")

    per_frame = load_mot(results_path)
    sample = cv2.imread(str(write[0]))
    if sample is None:
        raise RuntimeError(f"Cannot read {write[0]}")
    h, w = sample.shape[:2]
    out_w, out_h, _ = _output_size(h, w, max_width)

    writer = None
    jpeg_dir: Optional[Path] = None
    if out_path.suffix.lower() in {".mp4", ".webm", ".avi"}:
        writer, _fourcc = open_video_writer(out_path, fps, (out_w, out_h))
    else:
        jpeg_dir = out_path
        if jpeg_dir.exists() and jpeg_dir.is_dir():
            for old in jpeg_dir.glob("*.jpg"):
                old.unlink()
        jpeg_dir.mkdir(parents=True, exist_ok=True)

    trails: Dict[int, Deque[Tuple[int, int]]] = {}

    if trail_warmup:
        for fp in warmup:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            fidx = frame_index(fp)
            draw_tracks(
                img,
                per_frame.get(fidx, []),
                trails,
                trail_length=trail_length,
                label=label,
                frame_idx=fidx,
            )

    written = 0
    first_mot: Optional[int] = None
    last_mot: Optional[int] = None
    for fp in write:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        fidx = frame_index(fp)
        draw_tracks(
            img,
            per_frame.get(fidx, []),
            trails,
            trail_length=trail_length,
            label=label,
            frame_idx=fidx,
        )
        if (out_w, out_h) != (w, h):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if writer is not None:
            writer.write(img)
        if jpeg_dir is not None:
            cv2.imwrite(
                str(jpeg_dir / f"{written:06d}.jpg"),
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
        written += 1
        first_mot = fidx if first_mot is None else first_mot
        last_mot = fidx

    if writer is not None:
        writer.release()
    if written == 0:
        raise RuntimeError(f"Wrote 0 frames to {out_path}")
    if writer is not None and browser_mp4 and out_path.suffix.lower() == ".mp4":
        ensure_browser_mp4(out_path)
    return RenderResult(
        written=written,
        out_path=out_path,
        writer_fps=float(fps),
        width=out_w,
        height=out_h,
        start_mot_frame=first_mot,
        end_mot_frame=last_mot,
    )


def write_placeholder_mp4(
    out_path: Path,
    *,
    n_frames: int,
    fps: float,
    width: int,
    height: int,
    message: str,
    browser_mp4: bool = False,
) -> None:
    """Grey clip (mp4/webm) or JPEG folder with a status message."""
    n_frames = max(1, int(n_frames))
    w = width if width % 2 == 0 else width + 1
    h = height if height % 2 == 0 else height + 1
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, w / 900.0)
    y = h // 2
    for i, line in enumerate(message.split("\n")[:6]):
        (tw, th), _ = cv2.getTextSize(line, font, scale, 1)
        cv2.putText(canvas, line, ((w - tw) // 2, y + i * (th + 8)), font, scale, (200, 200, 200), 1, cv2.LINE_AA)

    if out_path.suffix.lower() in {".mp4", ".webm", ".avi"}:
        writer, _ = open_video_writer(out_path, fps, (w, h))
        for _ in range(n_frames):
            writer.write(canvas)
        writer.release()
        if browser_mp4 and out_path.suffix.lower() == ".mp4":
            ensure_browser_mp4(out_path)
        return

    if out_path.exists() and out_path.is_dir():
        for old in out_path.glob("*.jpg"):
            old.unlink()
    out_path.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        cv2.imwrite(str(out_path / f"{i:06d}.jpg"), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 70])


def main() -> None:
    args = parse_args()
    result = render_overlay(
        args.frames,
        args.results,
        args.out,
        fps=args.fps,
        trail_length=args.trail_length,
        max_frames=args.max_frames,
        max_seconds=args.max_seconds,
        start_frame=args.start_frame,
        start_frac=args.start_frac,
        max_width=args.max_width,
        frame_stride=args.frame_stride,
        trail_warmup=args.trail_warmup,
        browser_mp4=args.browser_mp4,
        label=args.label,
    )
    print(f"wrote {result.written} frames -> {result.out_path}")


if __name__ == "__main__":
    main()
