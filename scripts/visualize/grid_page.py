"""HTML tracker comparison page: one sequence at a time, all trackers on screen.

Dataset dropdown lists sibling ``<fps>/<benchmark>/`` grids that already have
``manifest.json``. Video dropdown lists sequences rendered into ``cells/``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mot_pipeline.tracker_meta import display_tracker_name

DATASET_LABELS = {
    "fasttracker_bench": "FastTracker-Bench",
    "ua_detrac": "UA-DETRAC",
    "trafficmot": "TrafficMOT",
    "cityflow": "CityFlow",
    "lumana_benchmark": "Lumana",
}


def dataset_label(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def sibling_datasets(page_dir: Path) -> List[Tuple[str, str]]:
    parent = Path(page_dir).resolve().parent
    found: List[Tuple[str, str]] = []
    if not parent.is_dir():
        return found
    for child in sorted(parent.iterdir()):
        if child.is_dir() and (child / "manifest.json").is_file():
            found.append((child.name, dataset_label(child.name)))
    return found


def short_seq_label(name: str) -> str:
    if len(name) <= 36:
        return name
    return f"{name[:14]}…{name[-16:]}"


def tile_shape(n: int) -> Tuple[int, int]:
    """Columns, rows so n tracker tiles fill one screen."""
    n = max(1, int(n))
    if n <= 3:
        return n, 1
    if n == 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    if n <= 8:
        return 4, 2
    if n <= 9:
        return 3, 3
    cols = int(n**0.5 + 0.999)
    rows = (n + cols - 1) // cols
    return cols, rows


def html_page(
    *,
    title: str,
    subtitle: str,
    trackers: Sequence[str],
    sequences: Sequence[str],
    cells: Dict[Tuple[str, str], Dict[str, Any]],
    detector_id: str,
    fps_label: str,
    n_frames: int,
    play_fps: float,
    dataset_id: str = "",
    datasets: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    play_fps = float(play_fps) if play_fps and play_fps > 0 else 5.0
    fallback_n = max(1, int(n_frames) or 1)
    seq_ns: Dict[str, int] = {}
    for seq in sequences:
        counts = [int((cells.get((t, seq)) or {}).get("written") or 0) for t in trackers]
        counts = [c for c in counts if c > 0]
        seq_ns[seq] = max(1, min(counts) if counts else fallback_n)

    cols, rows = tile_shape(len(trackers))
    ds_id = dataset_id or ""
    ds_list = list(datasets or ((ds_id, dataset_label(ds_id)),) if ds_id else ())
    ds_options = []
    for did, dlabel in ds_list:
        sel = " selected" if did == ds_id else ""
        ds_options.append(
            f'<option value="{html.escape(did, quote=True)}"{sel}>'
            f"{html.escape(dlabel)}</option>"
        )
    ds_label_html = ""
    if ds_options:
        ds_label_html = (
            "<label>Dataset "
            f'<select id="dataset-select">{"".join(ds_options)}</select>'
            "</label>"
        )

    options = []
    for seq in sequences:
        options.append(
            f'<option value="{html.escape(seq, quote=True)}">'
            f"{html.escape(short_seq_label(seq))}</option>"
        )

    tiles = []
    for tracker in trackers:
        dirs = {
            seq: str((cells.get((tracker, seq)) or {}).get("rel") or "").rstrip("/")
            for seq in sequences
        }
        run_id = "—"
        for seq in sequences:
            rid = (cells.get((tracker, seq)) or {}).get("run_id")
            if rid:
                run_id = str(rid)
                break
        label = display_tracker_name(tracker)
        dirs_attr = html.escape(json.dumps(dirs, separators=(",", ":")), quote=True)
        tiles.append(
            '<div class="cell">'
            f'<div class="tlabel" title="{html.escape(str(run_id))}">{html.escape(label)}</div>'
            f'<img class="clip" alt="{html.escape(label)}" data-dirs="{dirs_attr}"/>'
            "</div>"
        )

    data = {
        "fps": play_fps,
        "datasetId": ds_id,
        "sequences": [{"id": s, "n": seq_ns[s]} for s in sequences],
    }
    data_json = json.dumps(data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  html, body {{ height: 100%; margin: 0; }}
  body {{
    font-family: system-ui, sans-serif; background: #111; color: #eee;
    display: flex; flex-direction: column; overflow: hidden;
  }}
  header {{ flex: 0 0 auto; padding: 8px 12px 6px; background: #111; z-index: 2; }}
  h1 {{ font-size: 1rem; margin: 0 0 2px; }}
  .sub {{ color: #888; font-size: 0.75rem; margin-bottom: 6px; }}
  .controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  button, select {{ background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 4px 10px; cursor: pointer; }}
  #dataset-select, #seq-select {{ max-width: min(420px, 42vw); }}
  input[type=range] {{ flex: 1 1 220px; min-width: 140px; cursor: pointer; }}
  .viewer {{
    flex: 1 1 auto; min-height: 0;
    display: grid; gap: 6px; padding: 4px 8px 8px;
    grid-template-columns: repeat({cols}, 1fr);
    grid-template-rows: repeat({rows}, 1fr);
  }}
  .cell {{
    display: flex; flex-direction: column; min-width: 0; min-height: 0;
    background: #1a1a1a; border-radius: 6px; overflow: hidden;
  }}
  .tlabel {{
    flex: 0 0 auto; font-size: 0.75rem; font-weight: 600;
    padding: 4px 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .cell img {{
    flex: 1 1 auto; min-height: 0; width: 100%; object-fit: contain; background: #000;
  }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="sub">{html.escape(subtitle)} · detector_id={html.escape(detector_id)} · {html.escape(fps_label)}</div>
  <div class="controls">
    {ds_label_html}
    <label>Video
      <select id="seq-select">{"".join(options)}</select>
    </label>
    <button id="play" type="button">Play</button>
    <button id="pause" type="button">Pause</button>
    <button id="prev" type="button">-1</button>
    <button id="next" type="button">+1</button>
    <input id="scrub" type="range" min="0" max="0" value="0" step="1"/>
    <span id="clock">0.00s</span>
    <select id="rate">
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="2">2×</option>
    </select>
    <label><input type="checkbox" id="loop" checked/> loop</label>
  </div>
</header>
<div class="viewer">
{"".join(tiles)}
</div>
<script>
const DATA = {data_json};
const imgs = [...document.querySelectorAll("img.clip")];
const scrub = document.getElementById("scrub");
const clock = document.getElementById("clock");
const loopBox = document.getElementById("loop");
const rateSel = document.getElementById("rate");
const seqSel = document.getElementById("seq-select");
let seqId = (DATA.sequences[0] || {{}}).id || "";
let N = (DATA.sequences[0] || {{}}).n || 1;
let frame = 0;
let playing = false;
let timer = null;
let rate = 1;

function pad(i) {{ return String(i).padStart(6, "0"); }}
function seqMeta(id) {{
  return DATA.sequences.find(s => s.id === id) || {{id, n: 1}};
}}
function dirFor(img) {{
  try {{ return JSON.parse(img.getAttribute("data-dirs") || "{{}}")[seqId] || ""; }}
  catch (e) {{ return ""; }}
}}
function srcFor(img, i) {{
  const dir = dirFor(img);
  return dir ? (dir + "/" + pad(i) + ".jpg") : "";
}}
function show(i) {{
  frame = Math.max(0, Math.min(N - 1, i | 0));
  for (const img of imgs) {{
    const next = srcFor(img, frame);
    if (next && img.getAttribute("src") !== next) img.src = next;
  }}
  scrub.max = String(Math.max(0, N - 1));
  scrub.value = String(frame);
  clock.textContent = (frame / DATA.fps).toFixed(2) + "s  f" + frame;
  prefetch(frame);
}}
function prefetch(i) {{
  const hi = Math.min(N - 1, i + 12);
  for (const img of imgs) {{
    for (let k = i + 1; k <= hi; k++) {{
      const url = srcFor(img, k);
      if (!url) continue;
      const pre = new Image();
      pre.src = url;
    }}
  }}
}}
function pause() {{
  playing = false;
  if (timer) {{ clearTimeout(timer); timer = null; }}
}}
function tick() {{
  if (!playing) return;
  if (frame >= N - 1) {{
    if (loopBox.checked) show(0);
    else {{ pause(); return; }}
  }} else {{
    show(frame + 1);
  }}
  timer = setTimeout(tick, 1000 / (DATA.fps * rate));
}}
function play() {{
  if (playing) return;
  if (frame >= N - 1) show(0);
  playing = true;
  timer = setTimeout(tick, 1000 / (DATA.fps * rate));
}}
function setSeq(id) {{
  pause();
  seqId = id;
  N = seqMeta(id).n || 1;
  show(0);
}}
document.getElementById("play").onclick = play;
document.getElementById("pause").onclick = pause;
document.getElementById("prev").onclick = () => {{ pause(); show(frame - 1); }};
document.getElementById("next").onclick = () => {{ pause(); show(frame + 1); }};
rateSel.onchange = () => {{ rate = parseFloat(rateSel.value) || 1; }};
seqSel.onchange = () => setSeq(seqSel.value);
const datasetSel = document.getElementById("dataset-select");
if (datasetSel) {{
  datasetSel.onchange = () => {{
    const id = datasetSel.value;
    if (id && id !== (DATA.datasetId || "")) {{
      window.location.href = "../" + encodeURIComponent(id) + "/index.html";
    }}
  }};
}}
scrub.addEventListener("pointerdown", pause);
scrub.addEventListener("input", () => {{ pause(); show(parseInt(scrub.value, 10)); }});
scrub.addEventListener("change", () => {{ pause(); show(parseInt(scrub.value, 10)); }});
document.addEventListener("keydown", (e) => {{
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (e.key === " ") {{ e.preventDefault(); playing ? pause() : play(); }}
  if (e.key === "ArrowLeft") {{ pause(); show(frame - 1); }}
  if (e.key === "ArrowRight") {{ pause(); show(frame + 1); }}
}});
if (DATA.sequences.length) setSeq(DATA.sequences[0].id);
</script>
</body>
</html>
"""


def rebuild_page_html(page_dir: Path) -> Optional[Path]:
    """Rewrite index.html from an existing cells/ + manifest.json (no re-render)."""
    page_dir = Path(page_dir)
    manifest_path = page_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    trackers = list(manifest.get("trackers") or [])
    sequences = list(manifest.get("sequences") or [])
    runs = manifest.get("runs") or {}
    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for tracker in trackers:
        for seq in sequences:
            rel = f"cells/{tracker}/{seq}"
            jpg_dir = page_dir / "cells" / tracker / seq
            n = len(list(jpg_dir.glob("*.jpg"))) if jpg_dir.is_dir() else 0
            run_id = (runs.get(tracker) or {}).get("run_id")
            cells[(tracker, seq)] = {
                "rel": rel,
                "written": n,
                "status": "ok" if n else "missing",
                "run_id": run_id,
            }
    fps = manifest.get("target_fps")
    play_fps = float(fps) if fps not in (None, "", "None") else 5.0
    written = [int(c["written"]) for c in cells.values() if int(c.get("written") or 0) > 0]
    n_frames = min(written) if written else 1
    bench = manifest.get("benchmark") or page_dir.name
    html_text = html_page(
        title=f"{dataset_label(bench)} tracker grid",
        subtitle=f"{bench} / {manifest.get('split') or ''}",
        trackers=trackers,
        sequences=sequences,
        cells=cells,
        detector_id=str(manifest.get("detector_id") or ""),
        fps_label=f"{play_fps:g} FPS" if fps not in (None, "", "None") else "full / native",
        n_frames=n_frames,
        play_fps=play_fps,
        dataset_id=bench,
        datasets=sibling_datasets(page_dir),
    )
    out = page_dir / "index.html"
    out.write_text(html_text)
    return out
