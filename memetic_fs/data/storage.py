"""Persist immutable numerical inputs once for reusable process workers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa

from .validation import Dataset


def _encode_label(value):
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": str(value)}
    if type(value) in (str, int, float, bool):
        return {"type": type(value).__name__, "value": value}
    raise ValueError(f"Unsupported segment label type {type(value).__name__}; use strings, numeric values or dates")


def _decode_label(label):
    kind, value = label["type"], label["value"]
    converters = {"str": str, "int": int, "float": float, "bool": bool,
                  "timestamp": pd.Timestamp, "datetime": datetime.fromisoformat,
                  "date": date.fromisoformat, "decimal": Decimal}
    if kind not in converters:
        raise ValueError(f"Unsupported persisted segment label type {kind!r}")
    return converters[kind](value)


def _atomic_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_dataset(dataset: Dataset, directory: str | Path) -> Path:
    """Write full arrays to NPY files, preserving their existing numeric dtypes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in [("X", dataset.X), ("y", dataset.y), ("months", dataset.months)]:
        _atomic_array(directory / f"{name}.npy", value)
    segment_encoding = None
    if dataset.segments:
        temporary = directory / "segments.tmp.parquet"
        try:
            pd.DataFrame(dataset.segments).to_parquet(temporary, index=False)
            os.replace(temporary, directory / "segments.parquet")
        except (pa.ArrowException, ValueError, TypeError):
            # Arrow has no common scalar type for values such as 1 and "unknown".
            # Persist those labels safely rather than coercing them to strings,
            # which would change segment equality and constraint semantics.
            temporary.unlink(missing_ok=True)
            segment_encoding = {}
            for i, (name, values) in enumerate(dataset.segments.items()):
                codes, labels = pd.factorize(values, use_na_sentinel=True)
                filename = f"segment_{i}_codes.npy"
                _atomic_array(directory / filename, codes.astype(np.int32))
                segment_encoding[name] = {"file": filename, "labels": [_encode_label(v) for v in labels]}
    metadata = {"schema": 1, "feature_names": dataset.feature_names,
                "categorical_indices": dataset.categorical_indices,
                "categories": dataset.categories, "fingerprint": dataset.fingerprint,
                "category_encoding": {name: [_encode_label(v) for v in labels] for name, labels in dataset.categories.items()},
                "category_ordered": dataset.category_ordered,
                "segment_columns": list(dataset.segments), "segment_encoding": segment_encoding}
    temporary = directory / "metadata.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, sort_keys=True, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / "metadata.json")
    return directory


def load_dataset(directory: str | Path) -> Dataset:
    """Load read-only memory maps; no pickle deserialization is performed."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != 1:
        raise ValueError("Unsupported persisted dataset schema")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in ["X", "y", "months"]}
    segments = {}
    if metadata.get("segment_encoding"):
        for name, encoding in metadata["segment_encoding"].items():
            path = (directory / encoding["file"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Persisted segment path escapes the dataset directory")
            codes = np.load(path, mmap_mode="r", allow_pickle=False)
            labels = np.asarray([_decode_label(v) for v in encoding["labels"]], dtype=object)
            values = labels[np.maximum(codes, 0)] if len(labels) else np.full(len(codes), None, dtype=object)
            values[codes < 0] = None
            segments[name] = values
    elif metadata["segment_columns"]:
        frame = pd.read_parquet(directory / "segments.parquet")
        segments = {name: frame[name].to_numpy() for name in metadata["segment_columns"]}
    categories = {name: [_decode_label(v) for v in labels] for name, labels in metadata["category_encoding"].items()} if "category_encoding" in metadata else metadata["categories"]
    return Dataset(**arrays, segments=segments, feature_names=metadata["feature_names"],
                   categorical_indices=metadata["categorical_indices"], fingerprint=metadata["fingerprint"],
                   categories=categories, category_ordered=metadata.get("category_ordered", {}))
