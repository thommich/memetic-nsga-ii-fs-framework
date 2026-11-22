"""Validate development data without dropping or sampling observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from memetic_fs.cv.temporal import normalize_months


@dataclass
class Dataset:
    """A normalized predictor matrix and separately held metadata.

    Categorical predictors contain pandas category codes; missing categories are
    represented by -1, which LightGBM treats as missing. Categories and their
    order remain available in ``categories``.
    """

    X: np.ndarray
    y: np.ndarray
    months: np.ndarray
    segments: dict[str, np.ndarray]
    feature_names: list[str]
    categorical_indices: list[int]
    fingerprint: str
    categories: dict[str, list[Any]] = field(default_factory=dict)
    category_ordered: dict[str, bool] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Number of candidate predictors."""
        return self.X.shape[1]

    @property
    def n_rows(self) -> int:
        """Number of development observations."""
        return self.X.shape[0]


def _aligned(values: Any, frame: pd.DataFrame | None, n: int, name: str) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.DataFrame)) and frame is not None:
        if not values.index.equals(frame.index):
            raise ValueError(f"{name} index must exactly match X.index, including order")
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != n:
        raise ValueError(f"{name} must have one value per predictor row ({n} rows)")
    return result


def _hash_array(hasher: Any, value: np.ndarray) -> None:
    """Hash all data in bounded chunks, including noncontiguous arrays."""
    hasher.update(str(value.shape).encode())
    hasher.update(str(value.dtype).encode())
    for start in range(0, len(value), 65536):
        block = value[start:start + 65536]
        if block.dtype.kind in "OUS":
            hasher.update(pd.util.hash_pandas_object(pd.Series(block), index=False).values.tobytes())
            if block.dtype.kind == "O":
                # pandas intentionally hashes object 1 and object "1" identically;
                # segment constraints distinguish these values, so include types.
                types = pd.Series([f"{type(v).__module__}.{type(v).__qualname__}" for v in block])
                hasher.update(pd.util.hash_pandas_object(types, index=False).values.tobytes())
        else:
            hasher.update(np.ascontiguousarray(block).tobytes())


def _check_integer_precision(values: np.ndarray, dtype: np.dtype, name: str) -> None:
    """Reject lossy integer-to-float promotion on Windows as well as Linux."""
    if values.dtype.kind not in "iu" or dtype.kind != "f":
        return
    exact_limit = 2 ** (np.finfo(dtype).nmant + 1)
    if np.any(values > exact_limit) or (values.dtype.kind == "i" and np.any(values < -exact_limit)):
        for start in range(0, len(values), 65536):
            block = values[start:start + 65536]
            if not np.array_equal(block.astype(object), block.astype(dtype).astype(object)):
                raise ValueError(f"Predictor {name!r} loses integer precision in a mixed numeric matrix; explicitly convert it to a meaningful numeric or categorical representation")


def _metadata_aliases(frame: pd.DataFrame, values: np.ndarray) -> set[str]:
    """Find exact column aliases when external metadata has lost its Series name.

    This prevents a target, applicant ID or callable-CV time vector supplied as a
    NumPy array from accidentally leaving its original DataFrame column among
    the predictors. The equality check uses every row and never matches a
    partially similar column.
    """
    aliases = set()
    right = pd.Series(values)
    for name in frame.columns:
        left = frame[name].reset_index(drop=True)
        equal = left.eq(right).fillna(False) | (left.isna() & right.isna())
        if equal.all():
            aliases.add(name)
    return aliases


def prepare_data(
    X: pd.DataFrame | np.ndarray,
    y: str | Sequence[int] | np.ndarray,
    time_cv: str | Callable,
    id_col: str | None = None,
    segment_cols: Sequence[str] | None = None,
    allow_repeated_ids: bool = False,
    time_values: Sequence[Any] | None = None,
    ids: Sequence[Any] | None = None,
    feature_names: Sequence[str] | None = None,
) -> Dataset:
    """Separate predictors from metadata and validate aligned binary data.

    For arrays, supply ``time_values`` and ``ids`` (or explicitly permit repeated
    IDs). For a callable CV generator, ``time_values`` is required even when X is
    a DataFrame. Pass a temporal Series to preserve its name and exclude that
    named column from DataFrame predictors.
    """
    frame = X if isinstance(X, pd.DataFrame) else None
    if frame is not None and not frame.columns.is_unique:
        raise ValueError("X contains duplicate feature/metadata column names")
    if getattr(X, "ndim", None) != 2 or len(X) == 0:
        raise ValueError("X must be a nonempty two-dimensional DataFrame or NumPy array")
    n = len(X)
    segment_cols = list(segment_cols or [])
    if len(segment_cols) != len(set(segment_cols)):
        raise ValueError("segment_cols contains duplicate column names")

    def column(name: str) -> pd.Series:
        if frame is None or name not in frame.columns:
            raise ValueError(f"Required metadata column {name!r} is missing")
        return frame[name]

    metadata: set[str] = set(segment_cols)
    target = column(y) if isinstance(y, str) else y
    if isinstance(y, str):
        metadata.add(y)
    elif isinstance(y, pd.Series) and y.name in (frame.columns if frame is not None else []):
        metadata.add(y.name)
    target_array = _aligned(target, frame, n, "target")
    if pd.isna(target_array).any() or not np.isin(target_array, [0, 1]).all():
        raise ValueError("Target must contain only nonmissing binary values 0 and 1")
    if np.unique(target_array).size != 2:
        raise ValueError("Target must contain both classes 0 and 1")
    target_array = target_array.astype(np.int8, copy=False)
    if frame is not None and not isinstance(y, str) and not (isinstance(y, pd.Series) and y.name in frame.columns):
        metadata.update(_metadata_aliases(frame, target_array))

    if isinstance(time_cv, str):
        metadata.add(time_cv)
        dates = column(time_cv) if time_values is None else time_values
    elif callable(time_cv):
        if time_values is None:
            raise ValueError("Callable time_cv requires explicit time_values for chronological validation")
        dates = time_values
        if isinstance(dates, pd.Series) and frame is not None and dates.name in frame.columns:
            metadata.add(dates.name)
    else:
        raise ValueError("time_cv must be a temporal column name or callable fold generator")
    months = normalize_months(_aligned(dates, frame, n, "time_values"))
    if frame is not None and time_values is not None:
        if not (isinstance(dates, pd.Series) and dates.name in frame.columns):
            metadata.update(_metadata_aliases(frame, np.asarray(dates)))
        elif dates.name in frame.columns:
            metadata.add(dates.name)

    if id_col is not None:
        metadata.add(id_col)
        identifier = column(id_col)
        if ids is not None:
            raise ValueError("Supply id_col or ids, not both")
    else:
        identifier = ids
        if isinstance(ids, pd.Series) and frame is not None and ids.name in frame.columns:
            metadata.add(ids.name)
    identifier_array = None
    if identifier is None and not allow_repeated_ids:
        raise ValueError("Provide id_col (DataFrame) or ids (array): applicant uniqueness cannot be verified; alternatively set allow_repeated_ids=True")
    if identifier is not None:
        identifier_array = _aligned(identifier, frame, n, "applicant IDs")
        if frame is not None and id_col is None and not (isinstance(ids, pd.Series) and ids.name in frame.columns):
            metadata.update(_metadata_aliases(frame, identifier_array))
        if pd.isna(identifier_array).any():
            raise ValueError("Applicant IDs contain missing values")
        if not allow_repeated_ids and pd.Series(identifier_array).duplicated().any():
            raise ValueError("Repeated applicant IDs are prohibited; set allow_repeated_ids=True to permit them without dropping rows")
    segments = {name: _aligned(column(name), frame, n, name) for name in segment_cols}

    categories: dict[str, list[Any]] = {}
    category_ordered: dict[str, bool] = {}
    categorical_indices: list[int] = []
    if frame is not None:
        names = [c for c in frame.columns if c not in metadata]
        if not all(isinstance(c, str) for c in names):
            raise ValueError("Predictor names must be strings")
        if feature_names is not None and list(feature_names) != names:
            raise ValueError("feature_names must match the DataFrame predictor columns in order")
        columns = []
        for i, name in enumerate(names):
            series = frame[name]
            if isinstance(series.dtype, pd.CategoricalDtype):
                categorical_indices.append(i)
                categories[name] = [v.item() if isinstance(v, np.generic) else v for v in series.cat.categories]
                category_ordered[name] = series.cat.ordered
                values = series.cat.codes.to_numpy()
            elif pd.api.types.is_numeric_dtype(series.dtype):
                if pd.api.types.is_extension_array_dtype(series.dtype):
                    dtype = series.dtype.numpy_dtype
                    if series.isna().any() and dtype.kind not in "f":
                        _check_integer_precision(series.dropna().to_numpy(dtype=dtype), np.dtype("float64"), name)
                        dtype = np.dtype("float64")
                    values = series.to_numpy(dtype=dtype, na_value=np.nan)
                else:
                    values = series.to_numpy(copy=False)
            else:
                raise ValueError(f"Predictor {name!r} has unsupported dtype {series.dtype}; explicitly convert categorical predictors to pandas category")
            columns.append(values)
        if not columns:
            raise ValueError("No predictor columns remain after excluding metadata")
        dtype = np.result_type(*(a.dtype for a in columns))
        # Integer-only matrices keep integer precision; do not silently round large
        # integer identifiers or amounts when another column forces a float matrix.
        if dtype.kind == "f":
            for name, values in zip(names, columns):
                _check_integer_precision(values, dtype, name)
        matrix = np.column_stack(columns)
    else:
        matrix = np.asanyarray(X)
        if matrix.dtype.kind not in "biuf":
            raise ValueError("Array predictors must have a numeric dtype")
        names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(matrix.shape[1])]
        if len(names) != matrix.shape[1] or len(set(names)) != len(names) or not all(isinstance(s, str) for s in names):
            raise ValueError("feature_names must contain one unique string per predictor")
        if matrix.shape[1] == 0:
            raise ValueError("X must contain at least one predictor")
    if matrix.dtype.kind == "f" and np.isinf(matrix).any():
        raise ValueError("Predictors contain infinite values; replace them explicitly before fitting")
    fingerprint = sha256(b"memetic-fs-dataset-v1")
    for array in [matrix, target_array, months]:
        _hash_array(fingerprint, array)
    for name, values in segments.items():
        fingerprint.update(name.encode())
        _hash_array(fingerprint, values)
    if identifier_array is not None:
        _hash_array(fingerprint, identifier_array)
    fingerprint.update(json.dumps({"features": names, "categories": categories, "category_ordered": category_ordered}, sort_keys=True, default=str).encode())
    return Dataset(matrix, target_array, months, segments, names, categorical_indices, fingerprint.hexdigest(), categories, category_ordered)
