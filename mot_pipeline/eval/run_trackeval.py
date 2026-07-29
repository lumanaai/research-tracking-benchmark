"""Run bundled TrackEval (HOTA / CLEAR / Identity) on a prepared layout."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mot_pipeline.paths import TRACKEREVAL_ROOT


def _ensure_trackeval_on_path() -> None:
    root = str(TRACKEREVAL_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _patch_numpy_aliases() -> None:
    """TrackEval still uses removed np.float / np.int aliases."""
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined, assignment]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined, assignment]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined, assignment]


def run_trackeval(
    *,
    gt_folder: Path,
    trackers_folder: Path,
    tracker_name: str,
    seq_info: Dict[str, int],
    output_folder: Path,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate ``tracker_name``; write summary.csv / summary.json under output_folder."""
    _ensure_trackeval_on_path()
    _patch_numpy_aliases()
    import trackeval  # noqa: WPS433

    metrics = metrics or ["HOTA", "CLEAR", "Identity"]
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            "BREAK_ON_ERROR": True,
        }
    )

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(Path(gt_folder).resolve()),
            "TRACKERS_FOLDER": str(Path(trackers_folder).resolve()),
            "OUTPUT_FOLDER": str(output_folder.resolve()),
            "TRACKERS_TO_EVAL": [tracker_name],
            # FastTracker's bundled TrackEval is patched for benchmark class names
            # (person=1, car=6, …). We rewrite filtered vehicle GT to class 1 and
            # evaluate the 'person' slot as a class-agnostic vehicle channel.
            "CLASSES_TO_EVAL": ["person"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "data",
            "PRINT_CONFIG": False,
            "SEQ_INFO": {k: int(v) for k, v in seq_info.items()},
        }
    )

    metrics_config = {"METRICS": metrics, "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric_cls in (
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
        trackeval.metrics.VACE,
    ):
        if metric_cls.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric_cls(metrics_config))
    if not metrics_list:
        raise RuntimeError("No TrackEval metrics selected.")

    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    summary = _flatten_results(output_res, tracker_name)
    summary_json = output_folder / "summary.json"
    summary_csv = output_folder / "summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_summary_csv(summary_csv, summary)

    status = output_msg.get("MotChallenge2DBox", {}).get(tracker_name, "unknown")
    if status != "Success":
        raise RuntimeError(f"TrackEval failed for {tracker_name}: {status}")

    return summary


_HEADLINE = {
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDP",
    "IDR",
    "IDSW",
    "Frag",
    "MT",
    "ML",
    "FP",
    "FN",
    "Recall",
    "Precision",
}


def _coerce_metric_value(metric_name: str, key: str, value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if hasattr(value, "tolist"):
        try:
            import numpy as np

            arr = np.asarray(value)
            if arr.ndim == 0:
                return float(arr)
            if metric_name == "HOTA" and key in ("HOTA", "DetA", "AssA", "LocA"):
                return float(arr.mean())
        except Exception:
            return None
    return None


def _flatten_results(output_res: dict, tracker_name: str) -> Dict[str, Any]:
    """Convert nested TrackEval output into {seq: {metric: value}} + COMBINED."""
    dataset = output_res.get("MotChallenge2DBox", {})
    tracker_res = dataset.get(tracker_name) or {}
    flat: Dict[str, Any] = {"sequences": {}, "COMBINED": {}}

    for seq, cls_map in tracker_res.items():
        # Prefer the class-agnostic slot we evaluate under ('person'); fall back.
        cls_res = cls_map.get("person") or cls_map.get("pedestrian") or cls_map
        row: Dict[str, Any] = {}
        for metric_name, fields in cls_res.items():
            if not isinstance(fields, dict):
                continue
            for k, v in fields.items():
                num = _coerce_metric_value(metric_name, k, v)
                if num is None:
                    continue
                # Prefer short headline names (MOTA, IDF1, …) when unambiguous.
                if k in _HEADLINE:
                    row[k] = num
                else:
                    row[f"{metric_name}_{k}"] = num
        if seq == "COMBINED_SEQ":
            flat["COMBINED"] = row
        else:
            flat["sequences"][seq] = row
    return flat


def _write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    rows = []
    for seq, metrics in summary.get("sequences", {}).items():
        rows.append({"seq": seq, **metrics})
    if summary.get("COMBINED"):
        rows.append({"seq": "COMBINED", **summary["COMBINED"]})
    if not rows:
        path.write_text("")
        return
    # Stable column order: seq first, then preferred metrics, then the rest.
    preferred = [
        "HOTA",
        "DetA",
        "AssA",
        "LocA",
        "MOTA",
        "MOTP",
        "IDF1",
        "IDP",
        "IDR",
        "IDSW",
        "Frag",
        "MT",
        "ML",
        "FP",
        "FN",
        "Recall",
        "Precision",
    ]
    keys = ["seq"]
    seen = set(keys)
    for p in preferred:
        if any(p in r for r in rows):
            keys.append(p)
            seen.add(p)
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
