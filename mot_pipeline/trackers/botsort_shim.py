"""Import BoT-SORT motion tracker without FastReID / clone edits.

``tracker/bot_sort.py`` always imports ``FastReIDInterface`` at module load even
when ``with_reid=False``. We register a stub module first so motion-only runs
need none of FastReID's deps (yacs, termcolor, …). Also restores ``np.float``
for the upstream clone under NumPy 2.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

from mot_pipeline.paths import BOTSORT_ROOT


def _ensure_numpy_compat() -> None:
    # BoT-SORT still uses ``dtype=np.float`` (removed in NumPy 1.24+).
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]


def _ensure_fast_reid_stub() -> None:
    if "fast_reid.fast_reid_interfece" in sys.modules:
        return

    class FastReIDInterface:  # pragma: no cover - only if with_reid=True
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "BoT-SORT ReID is not wired in this pipeline. "
                "Keep with_reid=false (default), or see TODOs.md for appearance work."
            )

        def inference(self, *args: Any, **kwargs: Any):
            raise RuntimeError("BoT-SORT ReID is not wired in this pipeline.")

    pkg = types.ModuleType("fast_reid")
    iface = types.ModuleType("fast_reid.fast_reid_interfece")
    iface.FastReIDInterface = FastReIDInterface  # type: ignore[attr-defined]
    sys.modules.setdefault("fast_reid", pkg)
    sys.modules["fast_reid.fast_reid_interfece"] = iface


def ensure_botsort_importable() -> None:
    """Put BoT-SORT on ``sys.path`` and stub FastReID so ``tracker.bot_sort`` imports."""
    root = str(BOTSORT_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    _ensure_numpy_compat()
    _ensure_fast_reid_stub()
