"""Structured logging setup for the Ontario Grid Mapper."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a named logger configured with console + rotating file handlers."""
    from .config_loader import load_settings
    cfg = load_settings()
    log_cfg = cfg.get("logging", {})

    effective_level = level or log_cfg.get("level", "INFO")
    fmt = log_cfg.get(
        "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    log_file = log_cfg.get("file", "data/outputs/pipeline.log")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(effective_level)
    formatter = logging.Formatter(fmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
