import logging
from typing import Optional
from pathlib import Path


# re-export
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

LOG_FILE: Optional[Path] = None
LOG_LEVEL = DEBUG


# adapted from:
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class StreamFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    reset = "\x1b[0m"
    _format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        DEBUG: green + _format + reset,
        INFO: purple + _format + reset,
        WARNING: yellow + _format + reset,
        ERROR: red + _format + reset,
        CRITICAL: bold_red + _format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_handler(handler, level):
    handler.setLevel(level)
    handler.setFormatter(StreamFormatter())
    return handler


def getLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(setup_handler(logging.StreamHandler(), LOG_LEVEL))
    if LOG_FILE is not None:
        logger.addHandler(setup_handler(logging.FileHandler(LOG_FILE), LOG_LEVEL))
    return logger
