# utils/logger.py
# Minimal logging helper: writes timestamps + messages to logs/run-YYYYMMDD.log
# and also prints them to the terminal.

from __future__ import annotations
import logging, os, time
from typing import Optional

def get_logger(name: str = "agent", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, time.strftime("run-%Y%m%d.log"))

    logger = logging.getLogger(name)
    if logger.handlers:        # already configured once
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

def log_kv(logger: logging.Logger, **kwargs) -> None:
    """Convenience: log key=value pairs on one line."""
    parts = [f"{k}={repr(v)}" for k, v in kwargs.items()]
    logger.info(" | ".join(parts))
