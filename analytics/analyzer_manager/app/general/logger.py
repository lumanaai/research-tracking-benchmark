import argparse
import glob
import json
import logging
import logging.handlers
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np


class ExtTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def __init__(self, dir_log, log_file_name, when="midnight", interval=1, backup_count=14):

        self.dir_log = dir_log
        self.filename_prefix = log_file_name
        self.logs_history_count = backup_count
        filename = self.generate_new_name()
        super().__init__(filename, when=when, interval=interval, backupCount=0)

    def generate_new_name(self) -> str:
        date = datetime.today().strftime("%Y-%m-%d")
        return os.path.join(self.dir_log, f"{self.filename_prefix}-{date}.log")

    def doRollover(self):
        """
        TimedRotatingFileHandler remix - rotates logs on daily basis, and filename of current logfile is with the
        current date
        """
        self.stream.close()
        self.baseFilename = self.generate_new_name()
        self.stream = open(self.baseFilename, "w")
        self.rolloverAt = self.computeRollover(int(time.time()))

        # handle deletion
        tod = datetime.now()
        old_logs = glob.glob(os.path.join(self.dir_log, self.filename_prefix + "*.log"))
        dates_str = [os.path.basename(old_log).split(".")[0].split("-")[1:] for old_log in old_logs]
        dates = [datetime(int(date_s[0]), int(date_s[1]), int(date_s[2])) for date_s in dates_str]
        delta_days = [(tod - date).days for date in dates]
        to_delete = [old_logs[i] for i in range(len(delta_days)) if delta_days[i] > self.logs_history_count]
        for file_name in to_delete:
            try:
                os.remove(file_name)
            except OSError as e:
                print("error occured when deleting old log files")

    def getFilesToDelete(self) -> List[str]:
        return []


HTTP_LEVEL_NUM = int(logging.INFO + 5)

name_to_level = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
    "HTTP": HTTP_LEVEL_NUM,
}


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):  # Combines all np integer types
            return int(obj)

        elif isinstance(obj, (np.floating,)):  # Combines all np float types
            return float(obj)

        elif isinstance(obj, (np.ndarray,)):  # No change here
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):  # No change here
            return bool(obj)

        elif isinstance(obj, (np.void)):  # No change here
            return None

        elif isinstance(obj, argparse.Namespace):  # No change here
            return str(obj)

        elif isinstance(obj, (np.complexfloating,)):  # Combines all np complex types
            return {"real": obj.real, "imag": obj.imag}

        return super().default(obj)


class SpecificTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = ...) -> str:
        ct = time.localtime(record.created)
        s = time.strftime("%Y-%m-%dT%H:%M:%S", ct)
        msec = int(record.msecs)
        utc_off = time.strftime("%z", ct)
        return f"{s}.{msec:03}{utc_off[0:-2]}:{utc_off[-2:]}"

    def format(self, record):

        # Convert the log level to lowercase
        record.levelname = record.levelname.lower()

        # Call the original formatter to do the sprintf-style formatting
        return super(SpecificTimeFormatter, self).format(record)


class JsonifyLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        # Add your custom logic here
        # For example, you can modify the message or add extra information
        msg = json.dumps(msg, cls=NumpyEncoder)

        # Make sure to call the superclass method so that the message still gets logged
        super()._log(level, msg, args, exc_info, extra, stack_info)


def get_logger(
    log_dir: str,
    log_file_name: str,
    log_level: str = "noset",
    logger_days: int = 14,
    app_name: str = "analyzer_manager",
):
    # create the root dir
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    logging.addLevelName(HTTP_LEVEL_NUM, "HTTP")

    def http(self, message, *args, **kwargs):
        if self.isEnabledFor(HTTP_LEVEL_NUM):
            self._log(HTTP_LEVEL_NUM, message, args, **kwargs)

    # Step 3: Update the Logger class with the new method
    logging.Logger.http = http

    # logger setup
    logging_level = name_to_level[log_level.upper()]

    formatter = SpecificTimeFormatter(
        f'{{"app": "{app_name}", "level": "%(levelname)s", "cameraId": "%(threadName)s", "threadID": "%(thread)s", '
        '"message": "%(message)s", "timestamp":"%(asctime)s"}} '
    )

    # utc time instead of local time
    formatter.converter = time.gmtime

    filehandler = ExtTimedRotatingFileHandler(log_dir, log_file_name, when="midnight", backup_count=logger_days)
    filehandler.setFormatter(formatter)
    filehandler.setLevel(logging_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logging.setLoggerClass(JsonifyLogger)
    logger = logging.getLogger("LocalLogger")
    logger.setLevel(logging_level)

    logger.handlers = []
    logger.propagate = False

    logger.addHandler(filehandler)
    logger.addHandler(console_handler)
    logger.setLevel(logging_level)
    return logger
