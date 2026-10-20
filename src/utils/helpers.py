import hashlib
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Any
# import torch
from src.utils.logger import get_logger
logger = get_logger(__name__)

def time_it(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:   
        # all Values to one tuple --> *args
        # all dict Values into One Dict --> *kwargs
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()  
            elapsed_time = end_time - start_time
        logger.info(
                f"Function [{func.__name__}] executed in "
                f"{elapsed_time:.4f} seconds"
            )
    return wrapper

# Generates an MD5 hash for a given file. Used to track which PDFs have already been processed and embedded
# to prevent duplicate data in the FAISS index.
def generate_file_hash(file_path: Path | str) -> str:
    file_path = Path(file_path)
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()
        logger.debug(
            f"Generated file hash for [{file_path.name}]: {file_hash}"
        )
        return file_hash
    except FileNotFoundError:
        logger.error(f"Cannot generate hash. File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error Generating Hash File for {file_path}: {e}")
        raise

# def get_optimal_device() -> str:
#     """
#     Detects the best available hardware device for Hugging Face models.
#     Returns 'cuda' for NVIDIA GPUs, 'mps' for Mac M-series, or 'cpu'.
#     """
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
#     logger.info(f"Hardware detection complete. Optimal device set to: {device.upper()}")
#     return device

