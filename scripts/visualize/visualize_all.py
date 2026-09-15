#!/usr/bin/env python3
"""Render tracker-grid HTML from existing run_all.sh findings.

Never detects or tracks. Looks up the latest findings row per
benchmark × tracker × detector_id × target_fps (same rule as
compare_findings.py) and overlays that run's ``tracks/<seq>.txt`` on MOT
frames as JPEG cells plus a synced HTML viewer (Dataset + Video dropdowns).

Typical flow:

    ./scripts/batch/visualize_all.sh
    ./scripts/batch/serve_visualizations.sh

    N_SEQS=0 BENCHMARKS=lumana_benchmark FPS_VALUES="5" ./scripts/batch/visualize_all.sh

Direct:

    .venv/bin/python scripts/visualize/visualize_all.py \\
        --benchmark fasttracker_bench --detector yolov8 --fps 5
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mot_pipeline.findings import combo_dir, latest_record  # noqa: E402
from mot_pipeline.mot_io import list_frames, parse_seqinfo  # noqa: E402
from mot_pipeline.paths import DEFAULT_YOLO_WEIGHTS, FINDINGS_ROOT, SSD_ROOT  # noqa: E402
from mot_pipeline.registry import get_benchmark, get_detector  # noqa: E402
from mot_pipeline.tracker_meta import display_tracker_name  # noqa: E402
from mot_pipeline.trackers.base import resolve_tracking_schedule  # noqa: E402

import cv2  # noqa: E402

try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    pass

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
from visualize_mot_results import render_overlay, write_placeholder_mp4  # noqa: E402
from grid_page import html_page, rebuild_page_html, sibling_datasets  # noqa: E402

DEFAULT_BENCHMARKS = [
    "fasttracker_bench",
    "ua_detrac",
    "trafficmot",
    "cityflow",
    "lumana_benchmark",
]
DEFAULT_TRACKERS = [
    "fasttracker",
    "ocsort",
    "hybridsort",
    "analytics_bytetrack",
    "analytics_bytetrack_plus",
    "analytics_bytetrack_plus_aug19",
    "botsort",
]
PREFERRED_SEQUENCES = {
    "fasttracker_bench": ["task_day_occlusion", "task_night_occlusion", "task_tunnel"],
    "ua_detrac": ["MVI_20011", "MVI_39811", "MVI_40751"],
    "trafficmot": [],
    "cityflow": ["S01_c001", "S02_c006", "S03_c010"],
    "lumana_benchmark": [],
}
COMPARE_STEMS = {
    "gt_vehicles": "gt_vehicles",
    "yolov8m-expert_eff-1_2_imgsz704x1280_conf0.3_vehicles": "yolov8m_expert_eff",
    "yolov8m-expert_eff-1_2_imgsz1280_conf0.25_vehicles": "yolov8m_expert_eff",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize latest run_all.sh tracks as a synced HTML grid (no re-tracking)."
    )
    p.add_argument("--benchmarks", nargs="+", default=None)
    p.add_argument("--benchmark", default=None, help="Shorthand for a single --benchmarks value.")
    p.add_argument("--trackers", nargs="+", default=DEFAULT_TRACKERS)
    p.add_argument("--detector", default="yolov8", help="Detector name used by run_all.sh (default: yolov8).")
    p.add_argument(
        "--detector-id",
        default=None,
        help="Pin the findings detector_id (otherwise derived from --detector / --weights).",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    p.add_argument(
        "--fps",
        default="full",
        help='Target FPS matching run_all.sh FPS_VALUES: a number or "full" (native rate).',
    )
    p.add_argument("--sequences", nargs="+", default=None, help="Override sequence subset.")
    p.add_argument(
        "--n-seqs",
        type=int,
        default=3,
        help="Sequences per benchmark to render (0 = all that have img1 frames).",
    )
    p.add_argument(
        "--snippet-sec",
        type=float,
        default=8.0,
        help="Seconds of kept frames to render. 0 = full sequence (from the start).",
    )
    p.add_argument(
        "--start-frac",
        type=float,
        default=0.2,
        help="Skip this fraction of the sequence before the snippet (ignored when --snippet-sec is 0).",
    )
    p.add_argument("--max-width", type=int, default=480)
    p.add_argument(
        "--out-root",
        type=Path,
        default=SSD_ROOT / "visualizations",
        help="SSD output root (not the NAS project tree).",
    )
    p.add_argument("--run-id-substr", default=None, help="Pin to one run_all.sh timestamp / sweep.")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Re-encode cells even if the mp4 already exists.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-browser-mp4", action="store_true", help="Skip ffmpeg H.264 remux.")
    p.add_argument(
        "--refresh-index",
        action="store_true",
        help="Only rewrite parent index.html files under --out-root (no rendering).",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Start a local HTTP server for --out-root (does not re-render unless other flags need it).",
    )
    p.add_argument("--port", type=int, default=8765, help="Port for --serve (default 8765).")
    p.add_argument(
        "--serve-only",
        action="store_true",
        help="Only serve existing grids under --out-root; skip rendering.",
    )
    p.add_argument(
        "--print-urls",
        action="store_true",
        help="Print file:// and http:// paths for existing grids, then exit.",
    )
    p.add_argument(
        "--html-only",
        action="store_true",
        help="Rewrite index.html from existing cells/ + manifest.json (no re-render).",
    )
    return p.parse_args()


def _fps_arg(raw: str) -> Optional[float]:
    text = str(raw).strip().lower()
    if text in ("", "full", "native", "none"):
        return None
    return float(text)


def fps_dir_tag(target_fps: Optional[float]) -> str:
    if target_fps is None:
        return "full"
    return f"fps{float(target_fps):g}".replace(".", "p")


def compare_stem(detector_id: str) -> str:
    return COMPARE_STEMS.get(detector_id, detector_id)


def resolve_detector_id(detector: str, weights: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    kwargs: Dict[str, Any] = {}
    if detector == "yolov8":
        kwargs["weights"] = Path(weights)
    elif detector == "gt":
        kwargs["benchmark"] = "fasttracker_bench"
    return get_detector(detector, **kwargs).detector_id()


def available_detector_ids(benchmark: str, tracker: str) -> List[str]:
    root = FINDINGS_ROOT / benchmark / tracker
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "findings.json").is_file())


def has_frames(seq_dir: Path) -> bool:
    img1 = seq_dir / "img1"
    return img1.is_dir() and bool(list_frames(img1))


def pick_sequences(
    seq_dirs: Sequence[Path],
    *,
    preferred: Sequence[str],
    n: int,
    explicit: Optional[Sequence[str]] = None,
) -> List[Path]:
    by_name = {p.name: p for p in seq_dirs}
    if explicit:
        missing = [name for name in explicit if name not in by_name]
        if missing:
            raise SystemExit(f"Unknown --sequences: {missing}")
        picked = [by_name[name] for name in explicit]
        no_frames = [p.name for p in picked if not has_frames(p)]
        if no_frames:
            raise SystemExit(f"No img1 frames for sequences: {no_frames}")
        return picked

    unlimited = n is None or int(n) <= 0
    picked: List[Path] = []
    seen = set()
    for name in preferred:
        seq = by_name.get(name)
        if seq is None or not has_frames(seq):
            continue
        picked.append(seq)
        seen.add(seq.name)
        if not unlimited and len(picked) >= n:
            return picked
    for seq in seq_dirs:
        if seq.name in seen or not has_frames(seq):
            continue
        picked.append(seq)
        if not unlimited and len(picked) >= n:
            break
    return picked


def experiment_tracks_dir(record: Dict[str, Any]) -> Path:
    exp = record.get("experiment_dir")
    if exp:
        return Path(exp) / "tracks"
    run_id = record.get("run_id")
    if not run_id:
        raise ValueError("findings row has neither experiment_dir nor run_id")
    from mot_pipeline.paths import EXPERIMENTS_ROOT

    return EXPERIMENTS_ROOT / str(run_id) / "tracks"


def schedule_for_seq(
    seq_dir: Path, target_fps: Optional[float]
) -> Tuple[List[int], float, float, int]:
    meta = parse_seqinfo(seq_dir)
    seq_len = int(float(meta.get("seqLength") or 0) or 0)
    if seq_len <= 0:
        seq_len = len(list_frames(seq_dir / "img1"))
    extra: Dict[str, Any] = {}
    if target_fps is not None:
        extra["target_fps"] = float(target_fps)
    return resolve_tracking_schedule(meta, seq_len, extra)


def expected_snippet_len(
    seq_dir: Path,
    *,
    target_fps: Optional[float],
    start_frac: float,
    snippet_sec: float,
) -> Tuple[int, float, int]:
    """Return (n_write_frames, writer_fps, stride)."""
    frame_ids, writer_fps, _native, stride = schedule_for_seq(seq_dir, target_fps)
    if not frame_ids:
        return 0, writer_fps, stride
    # Full sequence: no time cap and no intro skip.
    if snippet_sec is not None and float(snippet_sec) <= 0:
        return len(frame_ids), float(writer_fps), stride
    frac = min(max(float(start_frac), 0.0), 0.95)
    start = int(len(frame_ids) * frac)
    if start >= len(frame_ids):
        start = max(0, len(frame_ids) - 1)
    n = len(frame_ids) - start
    if writer_fps > 0 and snippet_sec > 0:
        n = min(n, max(1, int(round(float(snippet_sec) * float(writer_fps)))))
    return n, float(writer_fps), stride


def lookup_runs(
    benchmark: str,
    trackers: Sequence[str],
    detector_id: str,
    target_fps: Optional[float],
    run_id_substr: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    hints: List[str] = []
    for tracker in trackers:
        if tracker == "traffictrack":
            continue
        rec = latest_record(
            benchmark=benchmark,
            tracker=tracker,
            detector_id=detector_id,
            target_fps=target_fps,
            run_id_substr=run_id_substr,
        )
        if rec is None:
            missing.append(tracker)
            alts = available_detector_ids(benchmark, tracker)
            if alts:
                hints.append(f"  {tracker}: available detector_ids={alts}")
            else:
                hints.append(f"  {tracker}: no findings under {combo_dir(benchmark, tracker, detector_id)}")
            continue
        found[tracker] = rec
    if missing:
        fps_txt = "full-rate" if target_fps is None else f"target_fps={target_fps:g}"
        extra = f" run_id_substr={run_id_substr!r}" if run_id_substr else ""
        print(
            f"[WARN] {benchmark}: no findings for {missing} "
            f"(detector_id={detector_id}, {fps_txt}{extra})",
            file=sys.stderr,
        )
        for line in hints:
            print(line, file=sys.stderr)
    return found


def _cell_task(spec: Dict[str, Any]) -> Dict[str, Any]:
    out_path = Path(spec["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_jpg = out_path / f"{max(0, int(spec.get('n_frames') or 1) - 1):06d}.jpg"
    if out_path.is_dir() and last_jpg.is_file() and not spec["force"]:
        n_existing = len(list(out_path.glob("*.jpg")))
        return {**spec["meta"], "status": "skipped", "out": str(out_path), "written": n_existing}
    if spec["placeholder"]:
        write_placeholder_mp4(
            out_path,
            n_frames=spec["n_frames"],
            fps=spec["writer_fps"],
            width=spec["width"],
            height=spec["height"],
            message=spec["message"],
            browser_mp4=spec["browser_mp4"],
        )
        return {
            **spec["meta"],
            "status": "placeholder",
            "out": str(out_path),
            "reason": spec["message"],
            "written": spec["n_frames"],
        }
    result = render_overlay(
        Path(spec["frames_dir"]),
        Path(spec["tracks_path"]),
        out_path,
        fps=spec["writer_fps"],
        trail_length=max(8, int(round(2.0 * spec["writer_fps"]))),
        max_seconds=spec["snippet_sec"],
        start_frac=spec["start_frac"],
        max_width=spec["max_width"],
        frame_stride=spec["stride"],
        trail_warmup=True,
        browser_mp4=spec["browser_mp4"],
        label=spec["label"],
    )
    return {
        **spec["meta"],
        "status": "ok",
        "out": str(out_path),
        "written": result.written,
        "start_mot_frame": result.start_mot_frame,
        "end_mot_frame": result.end_mot_frame,
        "width": result.width,
        "height": result.height,
    }


def discover_grid_pages(stem_root: Path) -> List[Dict[str, str]]:
    pages: List[Dict[str, str]] = []
    if not stem_root.is_dir():
        return pages
    for index in sorted(stem_root.glob("*/*/index.html")):
        fps_tag = index.parent.parent.name
        bench = index.parent.name
        pages.append({"path": str(index), "label": f"{bench} · {fps_tag}"})
    return pages


def write_parent_index(stem_root: Path) -> Path:
    stem_root.mkdir(parents=True, exist_ok=True)
    pages = discover_grid_pages(stem_root)
    items = []
    for page in pages:
        rel = Path(page["path"]).relative_to(stem_root).as_posix()
        items.append(f'<li><a href="{html.escape(rel)}">{html.escape(page["label"])}</a></li>')
    stem = stem_root.name
    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Tracker grids — {html.escape(stem)}</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;padding:24px}} a{{color:#9cf}}</style>
</head><body>
<h1>Tracker visualization grids</h1>
<p>Latest run_all.sh tracks · {html.escape(stem)}</p>
<ul>
{"".join(items) if items else "<li>No pages yet</li>"}
</ul>
</body></html>
"""
    path = stem_root / "index.html"
    path.write_text(body)
    return path


def write_root_index(out_root: Path) -> Path:
    """Landing page at the serve root so http://127.0.0.1:PORT/ is useful."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    items: List[str] = []
    for stem_root in sorted(p for p in out_root.iterdir() if p.is_dir()):
        for page in discover_grid_pages(stem_root):
            rel = Path(page["path"]).relative_to(out_root).as_posix()
            label = f"{stem_root.name} · {page['label']}"
            items.append(f'<li><a href="{html.escape(rel)}">{html.escape(label)}</a></li>')
    body = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>Tracker visualizations</title>
<link rel="icon" href="data:,"/>
<style>body{{font-family:system-ui;background:#111;color:#eee;padding:24px}} a{{color:#9cf}}</style>
</head><body>
<h1>Tracker visualizations</h1>
<p>Open a grid (one video at a time, all trackers on screen).</p>
<ul>
{"".join(items) if items else "<li>No grids yet — run visualize_all.sh first</li>"}
</ul>
</body></html>
"""
    path = out_root / "index.html"
    path.write_text(body)
    return path


def refresh_all_indexes(out_root: Path) -> List[Path]:
    """Rewrite the serve-root landing page and each detector-stem catalog."""
    out_root = Path(out_root)
    written: List[Path] = []
    if not out_root.is_dir():
        return written
    written.append(write_root_index(out_root))
    for stem_root in sorted(p for p in out_root.iterdir() if p.is_dir()):
        if discover_grid_pages(stem_root):
            written.append(write_parent_index(stem_root))
    return written


def print_open_urls(out_root: Path, port: int = 8765) -> List[Path]:
    pages: List[Path] = []
    if not out_root.is_dir():
        return pages
    # visualizations/<stem>/<fpsTag>/<bench>/index.html
    for index in sorted(out_root.glob("*/*/*/index.html")):
        pages.append(index)
    catalogs = sorted(out_root.glob("*/index.html"))
    print("")
    print("=== Open the HTML viewer in a browser ===")
    print("Step 1 already wrote these files. Step 2 is a separate server:")
    print("  ./scripts/batch/serve_visualizations.sh")
    for cat in catalogs:
        print(f"  catalog: file://{cat}")
        print(f"           http://127.0.0.1:{port}/{cat.relative_to(out_root).as_posix()}")
    for page in pages:
        rel = page.relative_to(out_root).as_posix()
        print(f"  grid:    file://{page}")
        print(f"           http://127.0.0.1:{port}/{rel}")
    if not pages and not catalogs:
        print(f"  (no index.html under {out_root})")
    print("")
    return pages


def serve_visualizations(out_root: Path, port: int) -> None:
    import http.server
    import os
    import socketserver

    out_root = Path(out_root).resolve()
    if not out_root.is_dir():
        raise SystemExit(f"Nothing to serve: {out_root}")
    pages = print_open_urls(out_root, port=port)
    write_root_index(out_root)

    class QuietFavicon(http.server.SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            super().do_GET()

        def log_message(self, fmt: str, *args: object) -> None:
            if args and str(args[0]).startswith("GET /favicon.ico"):
                return
            super().log_message(fmt, *args)

    handler = QuietFavicon

    class ReuseServer(socketserver.TCPServer):
        allow_reuse_address = True

    os.chdir(out_root)
    with ReuseServer(("0.0.0.0", int(port)), handler) as httpd:
        print(f"Serving {out_root}", flush=True)
        print(f"  http://127.0.0.1:{port}/", flush=True)
        for page in pages:
            rel = page.relative_to(out_root).as_posix()
            print(f"  http://127.0.0.1:{port}/{rel}", flush=True)
        print("Ctrl+C to stop.", flush=True)
        httpd.serve_forever()


def render_benchmark(
    *,
    benchmark: str,
    trackers: Sequence[str],
    detector_id: str,
    target_fps: Optional[float],
    n_seqs: int,
    sequences: Optional[Sequence[str]],
    snippet_sec: float,
    start_frac: float,
    max_width: int,
    out_root: Path,
    run_id_substr: Optional[str],
    jobs: int,
    force: bool,
    dry_run: bool,
    browser_mp4: bool,
) -> Optional[Path]:
    bench = get_benchmark(benchmark)
    split = bench.default_split()
    seq_dirs = bench.sequence_dirs(split)
    if not seq_dirs:
        print(f"[WARN] {benchmark}: no MOT sequences under split={split}", file=sys.stderr)
        return None

    picked = pick_sequences(
        seq_dirs,
        preferred=PREFERRED_SEQUENCES.get(benchmark, []),
        n=n_seqs,
        explicit=sequences,
    )
    if not picked:
        print(f"[WARN] {benchmark}: no sequences with img1/ frames", file=sys.stderr)
        return None

    runs = lookup_runs(benchmark, trackers, detector_id, target_fps, run_id_substr)
    active_trackers = [t for t in trackers if t != "traffictrack"]
    fps_label = "full / native" if target_fps is None else f"{target_fps:g} FPS"
    stem = compare_stem(detector_id)
    page_dir = out_root / stem / fps_dir_tag(target_fps) / benchmark
    cells_dir = page_dir / "cells"

    print(f"=== {benchmark}  detector_id={detector_id}  fps={fps_label} ===")
    print(f"  sequences: {', '.join(p.name for p in picked)}")
    for tracker in active_trackers:
        rec = runs.get(tracker)
        if rec:
            print(f"  {tracker}: run_id={rec.get('run_id')}  tracks={experiment_tracks_dir(rec)}")
        else:
            print(f"  {tracker}: MISSING findings (placeholder cells)")

    if dry_run:
        return None

    use_frac = 0.0 if snippet_sec is not None and float(snippet_sec) <= 0 else start_frac

    tasks: List[Dict[str, Any]] = []
    cell_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for seq_dir in picked:
        n_frames, writer_fps, stride = expected_snippet_len(
            seq_dir,
            target_fps=target_fps,
            start_frac=use_frac,
            snippet_sec=snippet_sec,
        )
        # Even width for yuv420p / libx264.
        width = max_width if max_width % 2 == 0 else max_width + 1
        height = 270
        sample = list_frames(seq_dir / "img1")
        if sample:
            img = cv2.imread(str(sample[0]))
            if img is not None:
                h, w = img.shape[:2]
                height = max(2, int(round(h * (width / float(w)))))
                if height % 2:
                    height += 1
        for tracker in active_trackers:
            rec = runs.get(tracker)
            out_path = cells_dir / tracker / seq_dir.name
            rel = out_path.relative_to(page_dir).as_posix()
            meta = {
                "tracker": tracker,
                "seq": seq_dir.name,
                "run_id": (rec or {}).get("run_id"),
                "rel": rel,
            }
            if rec is None:
                tasks.append(
                    {
                        "placeholder": True,
                        "out_path": str(out_path),
                        "n_frames": max(1, n_frames),
                        "writer_fps": writer_fps or 10.0,
                        "width": width,
                        "height": height,
                        "message": f"{tracker}\nno findings for this detector/fps",
                        "browser_mp4": browser_mp4,
                        "force": force,
                        "meta": meta,
                    }
                )
                continue
            tracks_path = experiment_tracks_dir(rec) / f"{seq_dir.name}.txt"
            if not tracks_path.is_file():
                tasks.append(
                    {
                        "placeholder": True,
                        "out_path": str(out_path),
                        "n_frames": max(1, n_frames),
                        "writer_fps": writer_fps or 10.0,
                        "width": width,
                        "height": height,
                        "message": f"{tracker}\nmissing {tracks_path.name}",
                        "browser_mp4": browser_mp4,
                        "force": force,
                        "meta": meta,
                    }
                )
                continue
            tasks.append(
                {
                    "placeholder": False,
                    "out_path": str(out_path),
                    "frames_dir": str(seq_dir / "img1"),
                    "tracks_path": str(tracks_path),
                    "writer_fps": writer_fps or 10.0,
                    "n_frames": max(1, n_frames),
                    "snippet_sec": snippet_sec,
                    "start_frac": use_frac,
                    "max_width": max_width,
                    "stride": stride,
                    "label": tracker,
                    "browser_mp4": browser_mp4,
                    "force": force,
                    "meta": meta,
                }
            )

    n_jobs = max(1, int(jobs))
    results: List[Dict[str, Any]] = []
    if n_jobs == 1 or len(tasks) <= 1:
        for spec in tasks:
            results.append(_cell_task(spec))
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futs = [pool.submit(_cell_task, spec) for spec in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())

    for item in results:
        key = (item["tracker"], item["seq"])
        cell_meta[key] = item
        print(f"  [{item.get('status')}] {item['tracker']} / {item['seq']}")

    page_dir.mkdir(parents=True, exist_ok=True)
    written_counts = [int(item.get("written") or 0) for item in results if int(item.get("written") or 0) > 0]
    page_n = min(written_counts) if written_counts else 1
    page_fps = 5.0
    if results:
        # Use the first task's writer fps from picked seqs (approx. target fps).
        page_fps = float(tasks[0]["writer_fps"]) if tasks else 5.0
    snippet_label = "full sequence" if snippet_sec is not None and float(snippet_sec) <= 0 else (
        f"snippets {snippet_sec:g}s from {start_frac:.0%}"
    )
    subtitle = f"{benchmark} / {split} · {snippet_label} · max width {max_width}px"
    title = f"{benchmark} tracker grid"
    page = html_page(
        title=title,
        subtitle=subtitle,
        trackers=active_trackers,
        sequences=[p.name for p in picked],
        cells=cell_meta,
        detector_id=detector_id,
        fps_label=fps_label,
        n_frames=page_n,
        play_fps=page_fps,
        dataset_id=benchmark,
        datasets=sibling_datasets(page_dir),
    )
    index_path = page_dir / "index.html"
    index_path.write_text(page)

    manifest = {
        "benchmark": benchmark,
        "split": split,
        "detector_id": detector_id,
        "target_fps": target_fps,
        "sequences": [p.name for p in picked],
        "trackers": active_trackers,
        "snippet_sec": snippet_sec,
        "start_frac": start_frac,
        "max_width": max_width,
        "run_id_substr": run_id_substr,
        "runs": {
            tracker: {
                "run_id": rec.get("run_id"),
                "evaluated_at": rec.get("evaluated_at"),
                "experiment_dir": rec.get("experiment_dir"),
                "HOTA": rec.get("HOTA"),
            }
            for tracker, rec in runs.items()
        },
        "cells": results,
    }
    (page_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Refresh sibling pages so the Dataset dropdown includes this benchmark.
    for child in page_dir.parent.iterdir():
        if child.is_dir() and (child / "manifest.json").is_file():
            rebuild_page_html(child)
    print(f"  wrote {index_path}")
    return index_path


def main() -> None:
    args = parse_args()
    if args.print_urls:
        print_open_urls(args.out_root, port=args.port)
        return
    if args.html_only:
        n = 0
        for manifest in sorted(args.out_root.glob("*/*/*/manifest.json")):
            path = rebuild_page_html(manifest.parent)
            if path is not None:
                print(f"html: {path}")
                n += 1
        if n == 0:
            print(f"No manifest.json grids under {args.out_root}", file=sys.stderr)
        return
    if args.serve_only:
        serve_visualizations(args.out_root, args.port)
        return
    if args.refresh_index:
        written = refresh_all_indexes(args.out_root)
        for path in written:
            print(f"index: {path}")
        if not written:
            print(f"No grid pages under {args.out_root}", file=sys.stderr)
        print_open_urls(args.out_root, port=args.port)
        return

    benches = list(args.benchmarks or [])
    if args.benchmark:
        benches = [args.benchmark]
    if not benches:
        benches = list(DEFAULT_BENCHMARKS)
    target_fps = _fps_arg(str(args.fps))
    detector_id = resolve_detector_id(args.detector, args.weights, args.detector_id)
    browser_mp4 = not args.no_browser_mp4

    wrote_any = False
    for bench in benches:
        path = render_benchmark(
            benchmark=bench,
            trackers=args.trackers,
            detector_id=detector_id,
            target_fps=target_fps,
            n_seqs=args.n_seqs,
            sequences=args.sequences,
            snippet_sec=args.snippet_sec,
            start_frac=args.start_frac,
            max_width=args.max_width,
            out_root=args.out_root,
            run_id_substr=args.run_id_substr,
            jobs=args.jobs,
            force=args.force,
            dry_run=args.dry_run,
            browser_mp4=browser_mp4,
        )
        if path is not None:
            wrote_any = True

    if args.dry_run:
        return
    for path in refresh_all_indexes(args.out_root):
        print(f"index: {path}")
    print_open_urls(args.out_root, port=args.port)
    if not wrote_any:
        print(
            "No grids written. Check that findings exist for "
            f"detector_id={detector_id} and that MOT seqs have img1/ frames.",
            file=sys.stderr,
        )
    if args.serve:
        print("Note: prefer ./scripts/batch/serve_visualizations.sh as a separate step.", flush=True)
        serve_visualizations(args.out_root, args.port)


if __name__ == "__main__":
    main()
