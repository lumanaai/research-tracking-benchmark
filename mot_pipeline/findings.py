"""Persistent per-combination indexes of experiment findings."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

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
