"""
Author : Mohit Patle

Description:
Shared utilities: project paths, configuration loading and logging.
Every other module in src/ builds on these helpers so that paths,
config and log formatting stay consistent across the project.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """
    Return the absolute path of the project root directory.

    The root is resolved relative to this file (src/utils.py), so it
    works no matter where the interpreter was started from
    (notebooks/, tests/, app/ or the command line).

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    config_path : str | Path | None
        Path to a YAML config. Defaults to ``configs/config.yaml``
        under the project root.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(config_path) if config_path else get_project_root() / "configs" / "config.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Create (or fetch) a logger that writes to both console and logs/.

    Parameters
    ----------
    name : str
        Logger name, usually ``__name__`` of the calling module.
    log_file : str
        File name inside the logs/ directory.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:  # already configured -> avoid duplicate handlers
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    log_dir = get_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger
