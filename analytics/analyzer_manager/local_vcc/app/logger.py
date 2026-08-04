"""Logger for local_vcc.

Reuses the shared ``get_logger`` from ``analyzer_manager/app/general/logger.py``
so the JSON log format / rotation matches the rest of the analyzer stack.

At container build time the shared module is copied to
``/opt/local_vcc/general/logger.py`` (see local_vcc/Dockerfile), which puts it
on ``PYTHONPATH`` alongside this package.
"""

import os
import traceback

from general.logger import get_logger as _shared_get_logger

_LOG_DIR = os.environ.get("VCC_LOG_DIR", "/usr/src/app/logs/")
_LOG_NAME = os.environ.get("VCC_LOG_NAME", "local_vcc")
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
_LOG_DAYS = int(os.environ.get("VCC_LOG_DAYS", "14"))

logger = _shared_get_logger(
    log_dir=_LOG_DIR, log_file_name=_LOG_NAME, log_level=_LOG_LEVEL, logger_days=_LOG_DAYS, app_name="local_vcc"
)


def log_exception(log, message: str, ex: Exception) -> None:
    """Mirror of ``general.analyzer_general.log_exception``."""
    exception_traceback = traceback.format_exc()
    log.error(f"{message}: {str(ex)}, traceback: {exception_traceback}")
