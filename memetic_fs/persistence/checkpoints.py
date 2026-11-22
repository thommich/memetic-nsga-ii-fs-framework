"""Portable, non-executable experiment serialization."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Normalize common scientific Python values without lossy string coercion."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted((json_safe(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}; supply structured JSON-compatible configuration")


def stable_hash(value: Any) -> str:
    """SHA-256 of a deterministic, canonical JSON representation."""
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, data: Any) -> None:
    """Write and fsync a sibling temporary file, then atomically replace *path*."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(json_safe(data), stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        if os.name != "nt":
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path) -> Any:
    """Read a structured checkpoint; this never deserializes executable objects."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)
