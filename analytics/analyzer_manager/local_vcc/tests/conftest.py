"""Pytest bootstrap for local_vcc tests.

Adds ``local_vcc/`` to ``sys.path`` so the worker source imports as ``app.*``
(matching the container layout where ``PYTHONPATH=/opt/local_vcc``), and adds
``analyzer_manager/`` so ``general.*`` shared modules resolve.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_VCC_DIR = os.path.dirname(_HERE)                       # .../local_vcc
_ANALYZER_MANAGER_DIR = os.path.dirname(_LOCAL_VCC_DIR)       # .../analyzer_manager

for path in (
    _LOCAL_VCC_DIR,
    os.path.join(_LOCAL_VCC_DIR, "app"),        # for bare ``from alert_types import ...``
    _ANALYZER_MANAGER_DIR,
    os.path.join(_ANALYZER_MANAGER_DIR, "app"), # for ``from general.alert_types import ...``
):
    if path not in sys.path:
        sys.path.insert(0, path)

# Keep worker logs out of the repo during tests.
os.environ.setdefault("VCC_LOG_DIR", "/tmp/local_vcc_test_logs/")
os.environ.setdefault("LOG_LEVEL", "WARNING")

