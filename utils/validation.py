"""
Input validation and path security for CLI and API.

- Prevents path traversal (output dir, data_path).
- Enforces numeric and string bounds for reproducibility and safety.
"""
from __future__ import annotations

import re
from pathlib import Path


# Bounds used across the pipeline (single source of truth)
MAX_TRANSACTIONS_PER_REQUEST = 50_000
MAX_LIST_TRANSACTIONS_API = 10_000
MAX_STRING_LENGTH = 256
MAX_AMOUNT = 1e12
MIN_AMOUNT = 0.0


def safe_output_path(path: str | Path, base: Path | None = None) -> Path:
    """
    Resolve output path and ensure it is under base (default: current cwd).
    Prevents path traversal (e.g. --output ../../../etc).
    """
    base = base or Path.cwd()
    base = base.resolve()
    p = (base / path).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Output path must be under project directory: {p} is not under {base}"
        ) from None
    return p


def safe_data_path(path: str | Path, base: Path | None = None) -> Path:
    """
    Resolve data file path: must exist and be under base (default: cwd).
    Prevents path traversal when loading external CSV.
    """
    base = base or Path.cwd()
    base = base.resolve()
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Data file not found or not a file: {p}")
    try:
        p.relative_to(base)
    except ValueError:
        # Allow absolute path outside cwd only if file exists (user explicitly points to CSV)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}") from None
    return p


def validate_test_size(value: float) -> float:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"test_size must be in [0, 1), got {value}")
    return value


def validate_positive_int(value: int, name: str, max_val: int = 50_000_000) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    if value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    return value


def validate_seed(value: int) -> int:
    if value < 0 or value > 2**31 - 1:
        raise ValueError(f"seed must be in [0, 2^31-1], got {value}")
    return value


def sanitize_string(s: str, max_len: int = MAX_STRING_LENGTH) -> str:
    """Return string stripped and truncated to max_len (no control chars)."""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    return s[:max_len].strip()
