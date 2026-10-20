import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Log Directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent   # go up 3(parent) folders
LOG_DIR  = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "rag_pipeline.log"

def get_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)  # it will from Debug to Critical All FIve Levels

    # Console Handler (INFO Level)
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)

    # File Handler (DEBUG Level)
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(lineno)d] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Max -> 5MB, Keep the Last 3 Backup
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes= 5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        delay=True
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)

    # Attach Handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger