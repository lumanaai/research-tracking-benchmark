"""Persistent per-combination indexes of experiment findings."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mot_pipeline.paths import FINDINGS_ROOT


PREFERRED_COLUMNS = [
    "run_id",
    "evaluated_at",
    "benchmark",
    "split",
    "tracker",
    "detector",
    "detector_id",
    "tracker_config",
    "tracker_config_sha256",
    "target_fps",
    "sequence_count",
    "exclude_motorcycles",
    "avg_ms_per_frame",
    "track_total_seconds",
    "track_total_frames",
    "experiment_dir",
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDP",
    "IDR",
    "IDCons",
    "IDSW",
    "Frag",
    "MT",
    "ML",
    "FP",
    "FN",
]


def _safe_component(value: object) -> str:
    text = str(value).strip()
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def _load_timing(experiment_dir: Path) -> Dict[str, Any]:
    path = Path(experiment_dir) / "timing.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _config_hash(config: object) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _ordered_columns(records: Iterable[Dict[str, Any]]) -> list[str]:
    present = {key for record in records for key in record}
    columns = [key for key in PREFERRED_COLUMNS if key in present]
    columns.extend(sorted(present - set(columns)))
    return columns


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def record_findings(
    *,
    spec: Dict[str, Any],
    combined_metrics: Dict[str, Any],
    experiment_dir: Path,
) -> Path:
    """Upsert one evaluated run into its benchmark/tracker/detector index.

    Layout:
      findings/<benchmark>/<tracker>/<detector_id>/{findings.csv,findings.json}
    """
    combo_dir = (
        FINDINGS_ROOT
        / _safe_component(spec["benchmark"])
        / _safe_component(spec["tracker"])
        / _safe_component(spec["detector_id"])
    )
    combo_dir.mkdir(parents=True, exist_ok=True)

    json_path = combo_dir / "findings.json"
    csv_path = combo_dir / "findings.csv"
    if json_path.is_file():
        records = json.loads(json_path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"Expected a list in findings index: {json_path}")
    else:
        records = []

    tracker_config = spec.get("tracker_config", {})
    extra = spec.get("extra") or {}
    timing = _load_timing(experiment_dir)
    record: Dict[str, Any] = {
        "run_id": spec["run_id"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": spec["benchmark"],
        "split": spec.get("split"),
        "tracker": spec["tracker"],
        "detector": spec["detector"],
        "detector_id": spec["detector_id"],
        "tracker_config": spec.get("tracker_config_path") or "default",
        "tracker_config_sha256": _config_hash(tracker_config),
        "target_fps": extra.get("target_fps"),
        "sequence_count": len(spec.get("sequences") or []),
        "exclude_motorcycles": bool(spec.get("exclude_motorcycles", False)),
        "experiment_dir": str(experiment_dir.resolve()),
        **combined_metrics,
    }
    if timing:
        for key in ("avg_ms_per_frame", "track_total_seconds", "track_total_frames"):
            if key in timing and timing[key] is not None:
                record[key] = timing[key]

    by_run_id = {
        existing["run_id"]: existing
        for existing in records
        if isinstance(existing, dict) and existing.get("run_id")
    }
    by_run_id[record["run_id"]] = record
    records = sorted(by_run_id.values(), key=lambda item: item["run_id"])

    _atomic_write(json_path, json.dumps(records, indent=2, sort_keys=True) + "\n")

    columns = _ordered_columns(records)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    _atomic_write(csv_path, buffer.getvalue())
    return combo_dir


def combo_dir(benchmark: str, tracker: str, detector_id: str) -> Path:
    return (
        FINDINGS_ROOT
        / _safe_component(benchmark)
        / _safe_component(tracker)
        / _safe_component(detector_id)
    )


def normalize_target_fps(val: object) -> Optional[float]:
    """Return float FPS or None for full-rate / missing (same rule as compare_findings)."""
    if val is None:
        return None
    text = str(val).strip()
    if text in ("", "None", "null", "—", "-"):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_combo_records(benchmark: str, tracker: str, detector_id: str) -> list[Dict[str, Any]]:
    path = combo_dir(benchmark, tracker, detector_id) / "findings.json"
    if not path.is_file():
        return []
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in findings index: {path}")
    return [r for r in records if isinstance(r, dict)]


def latest_record(
    *,
    benchmark: str,
    tracker: str,
    detector_id: str,
    target_fps: Optional[float] = None,
    run_id_substr: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Newest evaluated run for one combo, matching compare_findings latest-row rules.

    ``target_fps=None`` selects full-rate rows only (rows with no target_fps).
    """
    matched: list[Dict[str, Any]] = []
    for record in load_combo_records(benchmark, tracker, detector_id):
        run_id = str(record.get("run_id") or "")
        if run_id_substr and run_id_substr not in run_id:
            continue
        row_fps = normalize_target_fps(record.get("target_fps"))
        if target_fps is None:
            if row_fps is not None:
                continue
        else:
            if row_fps is None or abs(row_fps - float(target_fps)) > 1e-6:
                continue
        matched.append(record)
    if not matched:
        return None
    return max(
        matched,
        key=lambda item: (str(item.get("evaluated_at") or ""), str(item.get("run_id") or "")),
    )
