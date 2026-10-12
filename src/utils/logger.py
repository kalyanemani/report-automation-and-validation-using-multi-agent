from __future__ import annotations

"""
QDR·AI — Pipeline Logging Utility
──────────────────────────────────
Sets up dedicated file logger for pipeline-wide logs.
Logs are saved to data/logs/pipeline.log.
"""

import logging
import sys
from pathlib import Path
from src.config import DATA_DIR

LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_LOG_FILE = LOGS_DIR / "pipeline.log"


def setup_pipeline_logger() -> logging.Logger:
    """Configures and returns the pipeline logger."""
    logger = logging.getLogger("qdr_ai.pipeline")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # File Handler
    file_handler = logging.FileHandler(PIPELINE_LOG_FILE, encoding="utf-8")
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Force UTF-8 on Windows console stdout if supported
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


pipeline_logger = setup_pipeline_logger()
