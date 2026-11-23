"""Single-file temporal memetic NSGA-II feature selection.

Copy this file beside a notebook and import it. Third-party dependencies must be
installed separately. Generated from the maintained memetic_fs package; all
runtime framework code is in this module. Experiment artifacts remain in
checkpoint_dir.
"""
from __future__ import annotations


# === Source: memetic_fs/config.py ===
"""Typed, validated experiment settings. All advanced parameters are optional."""
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar
import math

T = TypeVar("T")

def from_dict(cls: type[T], values: dict | T | None) -> T:
    """Construct settings, rejecting unknown keys instead of silently ignoring typos."""
    if isinstance(values, cls):
        return values
    values = {} if values is None else values
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)

def positive(name: str, value: float, minimum: float = 0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")

@dataclass
class CVConfig:
    """Windows count distinct observed months; missing calendar months are skipped."""
    min_train_months: int = 3
    validation_months: int = 1
    step_months: int = 1
    min_total_months: int = 6
    gap_months: int = 0

    def __post_init__(self):
        for name in ("min_train_months", "validation_months", "step_months", "min_total_months"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            positive(name, value)
        if self.min_train_months < 3 or self.min_total_months < 6:
            raise ValueError("At least three training months and six total months are required")
        if self.gap_months != 0:
            raise ValueError("Version 1 supports gap_months=0 only")
        if self.step_months < self.validation_months:
            raise ValueError("step_months must be >= validation_months to avoid repeated OOF observations")

@dataclass
class OptimizationConfig:
    population_size: int = 50
    max_generations: int = 100
    min_generations: int = 10
    patience: int = 12
    min_improvement: float = 0.0001
    local_search_interval: int = 4
    local_search_budget: int = 40
    local_search_candidates: int = 4
    exploration_probability: float = 0.3
    inverse_exponent: float = 1.3
    inverse_smoothing: float = 1.0
    uniform_cardinality_probability: float = 0.15
    rfe_seed_sizes: tuple[int, ...] = (20, 40, 60, 80, 100, 140, 180, 200)
    reduced_generations: int = 5
    reduced_fold_count: int = 1

    def __post_init__(self):
        for name in ("population_size", "max_generations", "min_generations", "patience", "local_search_interval", "local_search_candidates", "reduced_fold_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            positive(name, value)
        if self.population_size < 2:
            raise ValueError("population_size must be >=2")
        for name in ("local_search_budget", "reduced_generations"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("exploration_probability", "uniform_cardinality_probability"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("inverse_exponent", "inverse_smoothing"):
            positive(name, getattr(self, name))
        if not isinstance(self.min_improvement, (int, float)) or not math.isfinite(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("min_improvement must be nonnegative")
        if not isinstance(self.rfe_seed_sizes, (list, tuple)) or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in self.rfe_seed_sizes):
            raise ValueError("rfe_seed_sizes must be a sequence of positive integers")

@dataclass
class ResourceConfig:
    memory_reserve_fraction: float = 0.25
    max_ram_gb: float | None = None
    max_workers: int | None = None
    threads_per_worker: int | None = None
    max_retries: int = 2
    memory_wait_seconds: float = 30

    def __post_init__(self):
        if not 0 < self.memory_reserve_fraction < 1:
            raise ValueError("memory_reserve_fraction must be between 0 and 1")
        for name in ("max_ram_gb", "max_workers", "threads_per_worker"):
            if getattr(self, name) is not None:
                positive(name, getattr(self, name))
        for name in ("max_workers", "threads_per_worker", "max_retries"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{name} must be an integer")
        if self.max_retries < 0 or not math.isfinite(self.memory_wait_seconds) or self.memory_wait_seconds < 0:
            raise ValueError("Retry and wait settings must be nonnegative")

@dataclass
class SegmentConstraint:
    column: str
    value: Any
    max_gini_drop: float = 0.01
    reference: str = "full_feature_model"

    def __post_init__(self):
        if self.reference != "full_feature_model":
            raise ValueError("Only full_feature_model is supported as a constraint reference")
        if not math.isfinite(self.max_gini_drop) or self.max_gini_drop < 0:
            raise ValueError("max_gini_drop must be finite and nonnegative")

@dataclass
class EvaluationConfig:
    min_segment_events: int = 30
    min_segment_nonevents: int = 30
    estimator_seed: int = 1729

    def __post_init__(self):
        positive("min_segment_events", self.min_segment_events)
        positive("min_segment_nonevents", self.min_segment_nonevents)
        for name in ("min_segment_events", "min_segment_nonevents", "estimator_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

@dataclass
class SelectionConfig:
    policy: str = "tolerance"
    gini_tolerance: float = 0.005
    bootstrap_repeats: int = 1000
    confidence: float = 0.95

    def __post_init__(self):
        if self.policy not in {"tolerance", "statistical", "knee", "manual"}:
            raise ValueError("Unknown selection policy")
        if self.gini_tolerance < 0 or not math.isfinite(self.gini_tolerance):
            raise ValueError("gini_tolerance must be finite and >=0 on the Gini coefficient scale")
        positive("bootstrap_repeats", self.bootstrap_repeats)
        if not isinstance(self.bootstrap_repeats, int):
            raise ValueError("bootstrap_repeats must be an integer")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be in (0,1)")


# === Source: memetic_fs/cv/temporal.py ===
"""Full-row monthly folds, independent of the optimizer and estimator."""


from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def normalize_months(values: Sequence[Any]) -> np.ndarray:
    """Return chronological monthly period ordinals from dates or integer YYYYMM.

    Missing calendar months are allowed. Window sizes count distinct *observed*
    months, never infer or manufacture observations for absent months.
    """
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or pd.isna(values).any():
        raise ValueError("Temporal values must be a nonempty, nonmissing one-dimensional vector")
    if values.dtype.kind in "iu":
        years, months = values // 100, values % 100
        if ((years < 1) | (years > 9999) | (months < 1) | (months > 12)).any():
            raise ValueError("Integer temporal values must be calendar-valid YYYYMM (months 01–12)")
        return (years.astype(np.int64) - 1970) * 12 + months.astype(np.int64) - 1
    if values.dtype.kind == "M" or all(isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)) for value in values):
        try:
            dates = pd.DatetimeIndex(values)
            return np.asarray((dates.year - 1970) * 12 + dates.month - 1, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Temporal datetime values could not be normalized to months") from error
    raise ValueError("Temporal values must be datetime values or integer YYYYMM; parse date strings explicitly")


@dataclass
class Fold:
    """Immutable-in-practice positions defining one full training/validation fold."""

    train: np.ndarray
    valid: np.ndarray
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe exact row indices, suitable for reproducible fold persistence."""
        return {"train": self.train.tolist(), "valid": self.valid.tolist(), "label": self.label}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Fold":
        """Restore persisted row indices without executing Python code."""
        return cls(np.asarray(value["train"], dtype=np.int64), np.asarray(value["valid"], dtype=np.int64), value["label"])


def _indices(value: Any, n: int, name: str) -> np.ndarray:
    indices = np.asarray(value)
    if indices.ndim != 1 or not len(indices):
        raise ValueError(f"{name} indices must be nonempty and one-dimensional")
    if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
        raise ValueError(f"{name} indices must be integer row positions, not masks or floats")
    if (indices < 0).any() or (indices >= n).any():
        raise ValueError(f"{name} indices are out of bounds for {n} observations")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} indices contain duplicate rows")
    return indices.astype(np.int64, copy=False)


def build_folds(
    months: np.ndarray,
    y: np.ndarray,
    config: Any,
    custom: Callable[[np.ndarray], Iterable[tuple[Sequence[int], Sequence[int]] | Fold]] | None = None,
) -> list[Fold]:
    """Build and validate folds before any model executes.

    Custom protocol: ``generator(month_ordinals) -> iterable[(train, valid)]``.
    Indices are integer row positions. Every month in the chosen training and
    validation intervals must be complete; custom training windows may roll.
    The final validation interval must reach the maximum development month.
    """
    months, y = np.asarray(months), np.asarray(y)
    if months.ndim != 1 or y.ndim != 1 or len(months) != len(y):
        raise ValueError("months and y must be aligned one-dimensional vectors")
    observed = np.unique(months)
    if config.min_train_months < 3 or config.min_total_months < 6:
        raise ValueError("CV requires at least three training months and six total observed months")
    if config.validation_months < 1 or config.step_months < 1 or config.gap_months < 0:
        raise ValueError("Validation horizon and step must be positive; gap must be nonnegative")
    if len(observed) < config.min_total_months:
        raise ValueError(f"At least {config.min_total_months} distinct observed months are required; found {len(observed)}")
    if custom is None:
        first = config.min_train_months + config.gap_months
        last = len(observed) - config.validation_months
        if first > last:
            raise ValueError("Insufficient observed months for the configured training minimum, gap and validation horizon")
        starts = list(range(first, last + 1, config.step_months))
        if starts[-1] != last:
            if last - starts[-1] < config.validation_months:
                starts[-1] = last
            else:
                starts.append(last)
        raw = [(np.flatnonzero(months <= observed[start - config.gap_months - 1]),
                np.flatnonzero(np.isin(months, observed[start:start + config.validation_months]))) for start in starts]
    else:
        raw = list(custom(months.copy()))
    if not raw:
        raise ValueError("Temporal fold generator returned no folds")
    folds = []
    previous_validation_end = None
    for i, item in enumerate(raw):
        if isinstance(item, Fold):
            train, valid = item.train, item.valid
        else:
            try:
                train, valid = item
            except (TypeError, ValueError) as error:
                raise ValueError("Each custom fold must contain training and validation index vectors") from error
        train, valid = _indices(train, len(months), "Training"), _indices(valid, len(months), "Validation")
        if np.intersect1d(train, valid).size:
            raise ValueError(f"Fold {i}: training and validation rows must be disjoint")
        train_months, valid_months = months[train], months[valid]
        if train_months.max() >= valid_months.min():
            raise ValueError(f"Fold {i}: temporal leakage; every training month must precede every validation month")
        if len(np.unique(train_months)) < config.min_train_months:
            raise ValueError(f"Fold {i}: training must include at least {config.min_train_months} distinct observed months")
        for name, indices, selected in [("training", train, train_months), ("validation", valid, valid_months)]:
            eligible = np.flatnonzero((months >= selected.min()) & (months <= selected.max()))
            if not np.array_equal(np.sort(indices), eligible):
                raise ValueError(f"Fold {i}: {name} must include every eligible observation in its monthly interval; partial-row/month sampling is prohibited")
        if np.unique(y[valid]).size != 2 or np.unique(y[train]).size != 2:
            raise ValueError(f"Fold {i}: both training and validation must contain target classes 0 and 1")
        if previous_validation_end is not None and valid_months.min() <= previous_validation_end:
            raise ValueError("Validation folds must be chronological and nonoverlapping, without repeated OOF observations")
        previous_validation_end = valid_months.max()
        first_label, last_label = pd.Period(ordinal=int(valid_months.min()), freq="M"), pd.Period(ordinal=int(valid_months.max()), freq="M")
        label = str(first_label) if first_label == last_label else f"{first_label}/{last_label}"
        folds.append(Fold(train, valid, label))
    if months[folds[-1].valid].max() != observed[-1]:
        raise ValueError("Last validation fold must extend to the maximum observed development month")
    return folds


# === Source: memetic_fs/data/validation.py ===
"""Validate development data without dropping or sampling observations."""


from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd




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


# === Source: memetic_fs/data/storage.py ===
"""Persist immutable numerical inputs once for reusable process workers."""


import json
import os
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa




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


# === Source: memetic_fs/execution/resources.py ===
"""Conservative admission based on currently available host/container resources."""


from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    """Effective CPU and memory limits at an admission boundary, in bytes."""
    cpus: int
    capacity_bytes: int
    available_bytes: int
    private_bytes: int
    temporary_free_bytes: int


@dataclass(frozen=True)
class ResourcePlan:
    """Safe next-dispatch limit; zero workers means wait for memory."""
    workers: int
    threads_per_worker: int
    memory_budget_bytes: int
    estimated_task_bytes: int
    snapshot: ResourceSnapshot


def _read_number(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _cgroup_paths(root: Path) -> list[Path]:
    """Include delegated cgroup directories and every enclosing limit."""
    paths = [root, root / "memory", root / "cpu", root / "cpu,cpuacct"]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            bases = [root] if not controllers else [root / controllers, *[root / name for name in controllers.split(",")]]
            for base in bases:
                candidate = base / relative.lstrip("/")
                if candidate.is_relative_to(root):
                    paths.extend([candidate, *[p for p in candidate.parents if p.is_relative_to(root)]])
    except (OSError, ValueError):
        pass
    return list(dict.fromkeys(paths))


def private_process_bytes(process: psutil.Process | None = None) -> int:
    """Use USS (excluding shared mapped pages); RSS is only a fallback."""
    process = process or psutil.Process()
    try:
        info = process.memory_full_info()
        return int(getattr(info, "uss", getattr(info, "pss", info.rss)))
    except (psutil.Error, OSError):
        try:
            return int(process.memory_info().rss)
        except psutil.Error:
            return 0


def detect_resources(temp_dir: str | Path | None = None, cgroup_root: str | Path = "/sys/fs/cgroup") -> ResourceSnapshot:
    """Inspect affinity, CPU quotas, host availability and cgroup v1/v2 limits."""
    memory = psutil.virtual_memory()
    capacity, available = int(memory.total), int(memory.available)
    process = psutil.Process()
    cpus = os.cpu_count() or 1
    try:
        cpus = min(cpus, len(process.cpu_affinity()))
    except (AttributeError, psutil.Error, OSError):
        pass
    for path in _cgroup_paths(Path(cgroup_root)):
        limit = _read_number(path / "memory.max")
        used = _read_number(path / "memory.current")
        if limit is None:
            limit = _read_number(path / "memory.limit_in_bytes")
            used = _read_number(path / "memory.usage_in_bytes")
        if limit is not None and limit > 0:
            capacity = min(capacity, limit)
            # Missing usage is not evidence the cgroup is empty.
            available = min(available, max(0, limit - (used if used is not None else memory.total - memory.available)))
        quota, period = None, None
        try:
            quota_text, period_text = (path / "cpu.max").read_text().split()
            quota = None if quota_text == "max" else int(quota_text)
            period = int(period_text)
        except (OSError, ValueError):
            quota = _read_number(path / "cpu.cfs_quota_us")
            period = _read_number(path / "cpu.cfs_period_us")
        if quota is not None and quota > 0 and period is not None and period > 0:
            cpus = min(cpus, max(1, math.floor(quota / period)))
    private = private_process_bytes(process)
    try:
        private += sum(private_process_bytes(child) for child in process.children(recursive=True))
    except psutil.Error:
        pass
    location = Path(temp_dir or tempfile.gettempdir())
    location.mkdir(parents=True, exist_ok=True)
    return ResourceSnapshot(max(1, cpus), capacity, max(0, min(capacity, available)), private, shutil.disk_usage(location).free)


class ResourceManager:
    """Estimate task admission, then refine it with full-row measured memory."""

    def __init__(self, config: Any, estimated_task_bytes: int, temp_dir: str | Path | None = None):
        self.config = config
        self.estimated_task_bytes = max(1, int(estimated_task_bytes))
        self.temp_dir = temp_dir
        self.worker_cap: int | None = None
        self.bound_threads: int | None = None

    def calibrate(self, measured_private_peak: int) -> None:
        """Reserve a 30% margin above observed private working allocation."""
        self.estimated_task_bytes = max(self.estimated_task_bytes, int(measured_private_peak * 1.3))

    def backoff(self, previous_workers: int) -> None:
        """Reduce future concurrency and increase memory estimate after failure."""
        self.worker_cap = max(1, previous_workers // 2)
        self.estimated_task_bytes = max(1, int(self.estimated_task_bytes * 1.5))

    def plan(self) -> ResourcePlan:
        """Return safe concurrency using available memory after the safety reserve."""
        snapshot = detect_resources(self.temp_dir)
        capacity, available = snapshot.capacity_bytes, snapshot.available_bytes
        max_ram = getattr(self.config, "max_ram_gb", None)
        if max_ram is not None:
            cap = int(max_ram * 1024**3)
            capacity = min(capacity, cap)
            available = min(available, max(0, cap - snapshot.private_bytes))
        reserve = float(getattr(self.config, "memory_reserve_fraction", 0.25))
        budget = max(0, int(available - capacity * reserve))
        specified_threads = getattr(self.config, "threads_per_worker", None)
        configured_workers = getattr(self.config, "max_workers", None) or snapshot.cpus
        # Worker admission/backoff must not change the estimator's thread budget:
        # references and candidates use the same effective training settings.
        threads = min(snapshot.cpus, int(specified_threads or max(1, snapshot.cpus // min(4, configured_workers))))
        if self.bound_threads is not None:
            threads = self.bound_threads
            if snapshot.cpus < threads:
                raise RuntimeError(f"Available CPU capacity ({snapshot.cpus}) fell below this run's fixed LightGBM thread budget ({threads}); restore CPU capacity before resuming")
        max_workers = configured_workers
        if self.worker_cap is not None:
            max_workers = min(max_workers, self.worker_cap)
        workers = min(int(max_workers), max(1, snapshot.cpus // threads), budget // self.estimated_task_bytes)
        return ResourcePlan(max(0, workers), max(1, threads), budget, self.estimated_task_bytes, snapshot)


# === Source: memetic_fs/progress.py ===
"""Timestamped progress using standard Python logging, without global setup.

The parent writes ``logs/run.log``. Every worker writes its own process log so
multiple processes never share a FileHandler. Console output is concise at
verbosity 1 and includes candidate/fold details at verbosity 2. Record fields
are available as ``record.progress_fields`` to existing logging integrations.
"""


from functools import lru_cache
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import threading
from typing import Any

from tqdm.auto import tqdm


class _ProgressFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record)


def _ancestor_has_output(logger: logging.Logger) -> bool:
    current = logger.parent
    while current is not None:
        if any(not isinstance(handler, logging.NullHandler) for handler in current.handlers):
            return True
        current = current.parent
    return False


class ProgressReporter:
    """Per-experiment progress logger, safe for notebooks and reusable workers.

    ``verbose=0`` shows warnings/errors, 1 shows milestones and a generation bar,
    and 2 additionally shows routine progress and candidate/fold detail. The files
    retain DEBUG details regardless of console verbosity. Existing ancestor
    logging handlers receive records; an extra console handler is added only
    when the application has not configured logging already.
    """

    def __init__(self, directory: str | Path, verbose: int = 1, parent_pid: int | None = None):
        if not isinstance(verbose, int) or isinstance(verbose, bool) or verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0 (quiet), 1 (progress), or 2 (detailed)")
        self.directory = Path(directory).resolve()
        self.verbose = verbose
        self.parent_pid = os.getpid() if parent_pid is None else parent_pid
        self._bar = None
        self._worker_status = {"workers_total": 0, "running": 0, "completed": 0, "failed": 0}
        logs = self.directory / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        process = os.getpid()
        self.path = logs / ("run.log" if process == self.parent_pid else f"worker-{process}.log")
        identity = hashlib.sha256(str(self.directory).encode()).hexdigest()[:12]
        self.logger = logging.getLogger(f"memetic_fs.progress.{identity}.{self.parent_pid}.{process}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = True
        formatter = _ProgressFormatter("%(asctime)sZ %(levelname)s pid=%(process)d %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        own_files = [handler for handler in self.logger.handlers if getattr(handler, "_memetic_file", False)]
        if not own_files:
            handler = logging.FileHandler(self.path, encoding="utf-8")
            handler._memetic_file = True
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        console = [handler for handler in self.logger.handlers if getattr(handler, "_memetic_console", False)]
        if process == self.parent_pid and not _ancestor_has_output(self.logger) and not console:
            handler = logging.StreamHandler()
            handler._memetic_console = True
            handler.setFormatter(formatter)
            handler.addFilter(lambda record: not _ancestor_has_output(self.logger))
            self.logger.addHandler(handler)
            console = [handler]
        for handler in console:
            handler.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[verbose])

    @property
    def descriptor(self) -> dict:
        """Small serializable worker settings, containing no data or handlers."""
        return {"directory": str(self.directory), "verbose": self.verbose, "parent_pid": self.parent_pid}

    def _emit(self, level: int, message: str, fields: dict[str, Any], *, screen: bool = True) -> None:
        if fields:
            message += " | " + " ".join(f"{key}={json.dumps(value, default=str, ensure_ascii=True)}" for key, value in sorted(fields.items()))
        # Parent logger levels are not automatically applied to propagated
        # records. Keep DEBUG detail in our files without flooding an existing
        # application's INFO console handler.
        process_is_worker = os.getpid() != self.parent_pid
        if (not screen or process_is_worker
                or (_ancestor_has_output(self.logger) and level < logging.getLogger("memetic_fs").getEffectiveLevel())):
            record = self.logger.makeRecord(self.logger.name, level, __file__, 0, message, (), None,
                                            extra={"progress_fields": fields})
            for handler in self.logger.handlers:
                if level >= handler.level and (screen and not process_is_worker or getattr(handler, "_memetic_file", False)):
                    handler.handle(record)
            return
        self.logger.log(level, message, extra={"progress_fields": fields})

    def info(self, message: str, **fields: Any) -> None:
        """Record routine progress in files and at console verbosity 2."""
        self._emit(logging.INFO, message, fields, screen=self.verbose == 2)

    def milestone(self, message: str, **fields: Any) -> None:
        """Show a concise milestone at verbosity 1 or 2."""
        self._emit(logging.INFO, message, fields, screen=self.verbose > 0)

    def debug(self, message: str, **fields: Any) -> None:
        """Record detailed candidate, fold or cache progress."""
        self._emit(logging.DEBUG, message, fields, screen=self.verbose == 2)

    def start_generations(self, total: int, completed: int = 0) -> None:
        if self.verbose == 1 and os.getpid() == self.parent_pid:
            self._bar = tqdm(total=total, initial=completed, desc="Generations", unit="gen",
                             dynamic_ncols=True, leave=True)
            self._refresh_bar()

    def generation(self, number: int) -> None:
        if self._bar is not None:
            self._bar.set_description_str(f"Running generation {number}/{self._bar.total}")
            self._refresh_bar()

    def generation_completed(self, number: int) -> None:
        if self._bar is not None:
            self._bar.update(max(0, number - self._bar.n))
            self._refresh_bar()

    def worker_status(self, total: int, running: int, completed: int, failed: int) -> None:
        self._worker_status = dict(workers_total=total, running=running, completed=completed, failed=failed)
        self._refresh_bar()

    def _refresh_bar(self) -> None:
        if self._bar is not None:
            self._bar.set_postfix(self._worker_status, refresh=True)

    def finish_generations(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def warning(self, message: str, **fields: Any) -> None:
        """Record operational warnings and recoverable failures."""
        self._emit(logging.WARNING, message, fields)

    def error(self, message: str, **fields: Any) -> None:
        """Record an actionable failure."""
        self._emit(logging.ERROR, message, fields)

    def close(self) -> None:
        """Flush and release log files; later messages reopen append handlers."""
        for handler in self.logger.handlers:
            if getattr(handler, "_memetic_file", False):
                handler.flush()
                handler.close()


class _NullProgress:
    descriptor = None

    def info(self, message: str, **fields: Any) -> None:
        pass

    debug = info
    milestone = info
    warning = info
    error = info
    def start_generations(self, *args: Any, **fields: Any) -> None:
        pass

    generation = start_generations
    generation_completed = start_generations
    worker_status = start_generations
    finish_generations = start_generations

    def close(self) -> None:
        pass


NULL_PROGRESS = _NullProgress()


def get_progress(owner):
    """Use an object's reporter when present; preserve lightweight test doubles."""
    return getattr(owner, "progress", None) or NULL_PROGRESS


@lru_cache(maxsize=32)
def _process_reporter(directory: str, verbose: int, parent_pid: int) -> ProgressReporter:
    return ProgressReporter(directory, verbose, parent_pid)


def progress_from_descriptor(descriptor: dict | None):
    """Open the correct per-process logger once for a reusable worker."""
    return _process_reporter(**descriptor) if descriptor else NULL_PROGRESS


@contextmanager
def progress_activity(reporter, message: str, interval_seconds: float = 30, **fields: Any):
    """Emit a heartbeat during a long native fit or a pending model batch.

    The timer never changes model parameters or inspects observations. It stops
    immediately when the surrounding activity finishes or raises.
    """
    if reporter is NULL_PROGRESS:
        yield
        return
    stopped = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not stopped.wait(interval_seconds):
            reporter.info(message, elapsed_seconds=round(time.monotonic() - started, 1), **fields)

    thread = threading.Thread(target=heartbeat, name="memetic-fs-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


# === Source: memetic_fs/execution/scheduler.py ===
"""Bounded batch scheduling with memory admission and recoverable retries."""


import logging
import time
from concurrent.futures import as_completed
from typing import Callable, Iterable, TypeVar

from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.process_executor import TerminatedWorkerError
from threadpoolctl import threadpool_limits




T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger(__name__)


def _limited_call(fn: Callable[[T], R], task: T, threads: int) -> R:
    with threadpool_limits(limits=threads):
        return fn(task)


class ResourceScheduler:
    """Reusable loky execution with resource checks before each admitted batch.

    Tasks must carry lightweight read-only mmap descriptors rather than full
    datasets. Workers should commit results to the cache before returning, so
    completed evaluations survive a different worker terminating in that batch.
    """

    def __init__(self, manager: ResourceManager, cancelled: Callable[[], bool] | None = None, progress=None):
        self.manager = manager
        self.cancelled = cancelled or (lambda: False)
        self.progress = progress or NULL_PROGRESS

    def _admit(self):
        deadline = time.monotonic() + float(getattr(self.manager.config, "memory_wait_seconds", 30))
        while True:
            if self.cancelled():
                self.progress.warning("Worker dispatch cancelled")
                raise KeyboardInterrupt("Feature-selection run cancelled before dispatch")
            plan = self.manager.plan()
            if plan.workers > 0:
                return plan
            if time.monotonic() >= deadline:
                self.progress.error("Worker admission failed under memory pressure", memory_budget_bytes=plan.memory_budget_bytes,
                                    estimated_task_bytes=plan.estimated_task_bytes)
                raise MemoryError("No full-fold worker fits the current memory budget. Free memory, increase the RAM limit, or use a larger instance; observations will not be sampled.")
            logger.warning("Pausing new model dispatch while available memory is below the admission reserve")
            self.progress.warning("Worker dispatch paused under memory pressure", memory_budget_bytes=plan.memory_budget_bytes,
                                  estimated_task_bytes=plan.estimated_task_bytes)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    def map(self, fn: Callable[[T], R], tasks: Iterable[T]) -> list[R]:
        """Evaluate tasks in order, retrying memory failures with safer concurrency."""
        task_list = list(tasks)
        if not task_list:
            return []
        results: list[R] = []
        cursor = 0
        retries = 0
        while cursor < len(task_list):
            plan = self._admit()
            workers = min(plan.workers, len(task_list) - cursor)
            batch = task_list[cursor:cursor + workers]
            started = time.perf_counter()
            self.progress.info("Worker batch admitted", tasks=len(batch), workers=workers,
                               threads_per_worker=plan.threads_per_worker, completed_tasks=cursor,
                               total_tasks=len(task_list), memory_budget_bytes=plan.memory_budget_bytes,
                               estimated_task_bytes=plan.estimated_task_bytes)
            self.progress.worker_status(total=len(batch), running=len(batch), completed=0, failed=0)
            try:
                if workers == 1:
                    try:
                        batch_results = [_limited_call(fn, batch[0], plan.threads_per_worker)]
                    except BaseException:
                        self.progress.worker_status(total=1, running=0, completed=0, failed=1)
                        raise
                    self.progress.worker_status(total=1, running=0, completed=1, failed=0)
                else:
                    # Descriptors contain paths, not matrices. Using loky
                    # directly avoids creating redundant joblib temporary
                    # memmapping-folder contexts for every batch.
                    environment = {name: str(plan.threads_per_worker) for name in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS")}
                    executor = get_reusable_executor(max_workers=workers, timeout=300, env=environment)
                    futures = [executor.submit(_limited_call, fn, task, plan.threads_per_worker) for task in batch]
                    batch_results, errors = [None] * len(futures), []
                    completed, failed = 0, 0
                    try:
                        # Drain the bounded batch before retrying: surviving
                        # workers can commit their completed cache records.
                        with progress_activity(self.progress, "Model batch still running", workers=workers,
                                               completed_tasks=cursor, total_tasks=len(task_list)):
                            positions = {future: index for index, future in enumerate(futures)}
                            for future in as_completed(futures):
                                try:
                                    batch_results[positions[future]] = future.result()
                                    completed += 1
                                except Exception as error:
                                    errors.append(error)
                                    failed += 1
                                self.progress.worker_status(total=len(batch),
                                    running=len(batch)-completed-failed, completed=completed, failed=failed)
                        if errors:
                            raise errors[0]
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise
                results.extend(batch_results)
                cursor += len(batch)
                retries = 0
                # A successful safer batch may regain concurrency at later
                # boundaries as resource availability improves.
                self.manager.worker_cap = None
                for result in batch_results:
                    if isinstance(result, dict) and result.get("memory_peak_bytes"):
                        self.manager.calibrate(int(result["memory_peak_bytes"]))
                self.progress.info("Worker batch completed", tasks=len(batch_results), workers=workers,
                                   completed_tasks=cursor, total_tasks=len(task_list),
                                   elapsed_seconds=time.perf_counter() - started)
            except (MemoryError, TerminatedWorkerError) as error:
                retries += 1
                self.manager.backoff(workers)
                if retries > int(getattr(self.manager.config, "max_retries", 2)):
                    self.progress.error("Worker batch retries exhausted", attempts=retries,
                                        error_type=type(error).__name__, completed_tasks=cursor)
                    raise RuntimeError(f"Full-row worker evaluation failed after {retries} attempts; completed cached results remain reusable") from error
                logger.warning("Retrying model batch after %s with reduced concurrency", type(error).__name__)
                self.progress.warning("Retrying worker batch with reduced concurrency", attempt=retries,
                                      error_type=type(error).__name__, previous_workers=workers,
                                      estimated_task_bytes=self.manager.estimated_task_bytes)
        return results


# === Source: memetic_fs/persistence/checkpoints.py ===
"""Portable, non-executable experiment serialization."""


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


# === Source: memetic_fs/persistence/cache.py ===
"""SQLite cache with durable per-evaluation completion records."""


import json
from pathlib import Path
import sqlite3
import time
from typing import Any




class EvaluationCache:
    """Process-safe evaluation results and retry state.

    Connections are short lived and never serialized to workers. A result only
    becomes reusable in the same transaction that marks it complete. Pending
    records are diagnostic and can always be rescheduled after interruption.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS evaluations (
                key TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, error TEXT, updated REAL NOT NULL)""")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def get(self, key: str) -> dict[str, Any] | None:
        """Return only a committed complete result."""
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM evaluations WHERE key=? AND status='complete'", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Commit compact metrics for one completed evaluation."""
        payload = json.dumps(json_safe(value), sort_keys=True, allow_nan=False)
        with self._connect() as connection:
            connection.execute("""INSERT INTO evaluations(key,status,payload,updated)
                VALUES (?,'complete',?,?) ON CONFLICT(key) DO UPDATE SET
                status='complete',payload=excluded.payload,error=NULL,updated=excluded.updated""", (key, payload, time.time()))

    def pending(self, key: str) -> None:
        """Record an attempt without overwriting an already completed result."""
        with self._connect() as connection:
            connection.execute("""INSERT INTO evaluations(key,status,attempts,updated)
                VALUES (?,'pending',1,?) ON CONFLICT(key) DO UPDATE SET
                status='pending',attempts=attempts+1,updated=excluded.updated
                WHERE evaluations.status!='complete'""", (key, time.time()))

    def failed(self, key: str, error: str) -> None:
        """Retain failure details for interrupted or retryable work."""
        with self._connect() as connection:
            connection.execute("""UPDATE evaluations SET status='failed',error=?,updated=?
                WHERE key=? AND status!='complete'""", (str(error), time.time(), key))

    mark_failed = failed

    def completed(self) -> list[dict[str, Any]]:
        """Read all completed result payloads in stable key order."""
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM evaluations WHERE status='complete' ORDER BY key").fetchall()
        return [json.loads(row[0]) for row in rows]

    def incomplete(self) -> list[dict[str, Any]]:
        """Describe pending/failed evaluations for checkpoint recovery."""
        with self._connect() as connection:
            rows = connection.execute("SELECT key,status,attempts,error FROM evaluations WHERE status!='complete' ORDER BY key").fetchall()
        return [dict(zip(("key", "status", "attempts", "error"), row)) for row in rows]


# === Source: memetic_fs/evaluation/lightgbm.py ===
"""Fixed-round LightGBM evaluation, with explicit sampling restrictions."""


from dataclasses import asdict
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any
import hashlib
import time
import warnings

import lightgbm as lgb
from lightgbm.basic import _ConfigAliases
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits









EVALUATOR_VERSION = "1.0.1"

def selected_matrix(X: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Copy only selected rows/columns into a LightGBM-compatible numeric dtype.

    Integer input is retained on disk; LightGBM requires float32/float64 matrices.
    Integers outside the exact float64 range are rejected rather than rounded.
    """
    values = X[np.ix_(rows, columns)]
    if values.dtype.kind in "iu":
        if np.any(values > 2**53) or (values.dtype.kind == "i" and np.any(values < -(2**53))):
            raise ValueError("Integer predictor values exceed the exact float64 range (2**53); explicitly encode or rescale these features without information loss before fitting")
        return values.astype(np.float64)
    if values.dtype.kind == "b" or values.dtype == np.float16:
        return values.astype(np.float32)
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        converted = values.astype(np.float64)
        if not np.array_equal(values, converted.astype(values.dtype), equal_nan=True):
            raise ValueError("Predictor dtype loses information when converted to LightGBM float64")
        return converted
    return values

def strict_params(params: dict | None, *, seed: int, threads: int, n_rows: int) -> dict:
    """Reject sampling, early stopping and overrides of managed training settings.

    All LightGBM aliases are checked, including aliases hidden by another key.
    Histogram construction uses every training row, rather than the 200k default.
    """
    supplied = dict(params or {})
    required = {"bagging_fraction": 1.0, "bagging_freq": 0,
                "pos_bagging_fraction": 1.0, "neg_bagging_fraction": 1.0,
                "feature_fraction": 1.0, "feature_fraction_bynode": 1.0,
                "data_sample_strategy": "bagging", "boosting": "gbdt",
                "objective": "binary", "device_type": "cpu",
                "deterministic": True, "force_col_wise": True, "force_row_wise": False,
                "tree_learner": "serial", "pred_early_stop": False}
    managed = {"num_iterations", "early_stopping_round", "bin_construct_sample_cnt",
               "num_threads", "seed", "data_random_seed", "feature_fraction_seed",
               "bagging_seed", "drop_seed", "extra_seed", "config"}
    for canonical, expected in required.items():
        for alias in _ConfigAliases.get(canonical):
            if alias in supplied and supplied[alias] != expected:
                raise ValueError(f"{alias}={supplied[alias]!r} conflicts with strict full-row setting {canonical}={expected!r}")
            supplied.pop(alias, None)
    for canonical in managed:
        bad = set(_ConfigAliases.get(canonical)) & supplied.keys()
        if bad:
            raise ValueError(f"Managed LightGBM parameter {sorted(bad)}: use the dedicated framework setting")
    unsupported = {"monotone_constraints", "feature_contri", "max_bin_by_feature",
                   "categorical_feature", "forcedsplits_filename", "forcedbins_filename",
                   "interaction_constraints", "cegb_penalty_feature_coupled", "cegb_penalty_feature_lazy",
                   "init_score", "weight_column", "label_column", "group_column", "ignore_column",
                   "data", "valid", "input_model", "num_machines", "machines", "machine_list_filename",
                   "parser_config_file", "predict_contrib", "predict_leaf_index"}
    for canonical in unsupported:
        bad = set(_ConfigAliases.get(canonical)) & supplied.keys()
        if bad:
            raise ValueError(f"{sorted(bad)} is not supported across changing feature subsets and framework-managed data")
    supplied.update(required)
    supplied.update(seed=seed, data_random_seed=seed, feature_fraction_seed=seed,
                    bagging_seed=seed, num_threads=threads,
                    bin_construct_sample_cnt=n_rows, verbosity=-1)
    return supplied

def score(y: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute binary AUC/Gini using every supplied observation."""
    if len(y) != len(predictions) or len(np.unique(y)) != 2:
        raise ValueError("AUC requires aligned predictions and both target classes")
    auc = float(roc_auc_score(y, predictions))
    return {"auc": auc, "gini": 2 * auc - 1, "n": len(y),
            "events": int(np.sum(y)), "nonevents": int(len(y) - np.sum(y))}

@lru_cache(maxsize=4)
def _load(directory: str, fingerprint: str):
    return load_dataset(directory)

def _segment_mask(values, value):
    if value is None or pd.isna(value):
        return np.asarray(pd.isna(values))
    return pd.Series(values).eq(value).fillna(False).to_numpy(dtype=bool)


def _worker(task: dict) -> dict:
    """Record failed attempts; successful tasks commit before any batch completes."""
    progress = progress_from_descriptor(task.get("progress"))
    try:
        return _evaluate_task(task, progress=progress)
    except Exception as error:
        EvaluationCache(task["cache_path"]).failed(task["key"], f"{type(error).__name__}: {error}")
        progress.error("Candidate evaluation failed", candidate=task["key"][:12],
                       fidelity=task["fidelity"], error_type=type(error).__name__)
        if isinstance(error, lgb.basic.LightGBMError) and any(
                marker in str(error).lower() for marker in ("bad_alloc", "bad allocation", "cannot allocate memory", "out of memory")):
            raise MemoryError(f"LightGBM full-fold allocation failed: {error}") from error
        raise
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()

def _evaluate_task(task: dict, progress=None) -> dict:
    """Load shared arrays by path; commit each successful task before returning."""
    progress = progress or NULL_PROGRESS
    started = time.perf_counter()
    cache = EvaluationCache(task["cache_path"])
    existing = cache.get(task["key"])
    if existing is not None:
        progress.debug("Worker evaluation cache hit", candidate=task["key"][:12], fidelity=task["fidelity"])
        return existing
    pass
    data = _load(task["data_path"], task["dataset_fingerprint"])
    mask = np.asarray(task["mask"], dtype=bool)
    cols = np.flatnonzero(mask)
    progress.debug("Candidate evaluation started", candidate=task["key"][:12], fidelity=task["fidelity"],
                   n_features=len(cols), folds=len(task["folds"]), num_boost_round=task["num_boost_round"],
                   threads=task["threads"], reference=task.get("reference", False))
    cats = [i for i, c in enumerate(cols) if c in data.categorical_indices]
    fold_results, segment_results, ys, ps = [], [], [], []
    importance = np.zeros(data.X.shape[1], dtype=float)
    segment_values = {column: pd.unique(pd.Series(values, dtype=object).where(pd.notna(values), None))
                      for column, values in data.segments.items()}
    with MemoryMonitor() as monitor, threadpool_limits(limits=1):
        for fold in task["folds"]:
            fold_started = time.perf_counter()
            train = np.load(fold["train_path"], mmap_mode="r")
            valid = np.load(fold["valid_path"], mmap_mode="r")
            fold_log = progress.info if task.get("reference", False) else progress.debug
            fold_log("Temporal fold started", candidate=task["key"][:12], fold=fold["label"],
                     fidelity=task["fidelity"], train_n=len(train), validation_n=len(valid),
                     n_features=len(cols), num_boost_round=task["num_boost_round"])
            params = strict_params(task["model_params"], seed=task["seed"],
                                   threads=task["threads"], n_rows=len(train))
            # np.ix_ makes only the selected fold/columns copy, not all predictors.
            with progress_activity(progress, "LightGBM fold fit still running", phase="model",
                                   candidate=task["key"][:12], fold=fold["label"],
                                   n_features=len(cols), train_n=len(train)):
                model = lgb.train(params, lgb.Dataset(selected_matrix(data.X, train, cols),
                                  label=data.y[train], categorical_feature=cats),
                                  num_boost_round=task["num_boost_round"])
            predictions = model.predict(selected_matrix(data.X, valid, cols), num_threads=task["threads"])
            yy = data.y[valid]
            metrics = score(yy, predictions)
            metrics.update(fold=fold["label"], train_n=len(train), validation_n=len(valid))
            fold_results.append(metrics)
            ys.append(yy)
            ps.append(predictions)
            importance[cols] += model.feature_importance(importance_type="gain")
            for column, values in data.segments.items():
                # Different missing sentinels represent one descriptive segment.
                for raw_value in segment_values[column]:
                    value = None if pd.isna(raw_value) else raw_value
                    selected = _segment_mask(values[valid], value)
                    sy = yy[selected]
                    counts = {"n": len(sy), "events": int(np.sum(sy)), "nonevents": int(len(sy) - np.sum(sy))}
                    sm = score(sy, predictions[selected]) if len(np.unique(sy)) == 2 else {"auc": None, "gini": None, **counts}
                    segment_results.append({"column": column, "value": value,
                                            "fold": fold["label"], **sm})
            fold_log("Temporal fold completed", candidate=task["key"][:12], fold=fold["label"],
                     fidelity=task["fidelity"], train_n=len(train), validation_n=len(valid),
                     gini=metrics["gini"], elapsed_seconds=time.perf_counter() - fold_started)
    ginis = [f["gini"] for f in fold_results]
    constraints = []
    for constraint in task["active_constraints"]:
        gaps = []
        for sm in segment_results:
            if sm["column"] == constraint["column"] and sm["value"] == constraint["value"]:
                ref = next(r for r in task["reference_segments"] if r["column"] == sm["column"] and r["value"] == sm["value"] and r["fold"] == sm["fold"])
                if sm["gini"] is None or ref["gini"] is None:
                    raise ValueError("Eligible constrained segment has undefined Gini")
                gaps.append(ref["gini"] - sm["gini"] - constraint["max_gini_drop"])
        if not gaps:
            raise ValueError("Constrained segment is missing from evaluation")
        constraints.append(float(max(gaps)))
    result = {"key": task["key"], "mask": mask.tolist(), "n_features": int(mask.sum()),
              "fidelity": task["fidelity"], "mean_gini": float(np.mean(ginis)),
              "std_gini": float(np.std(ginis)), "min_gini": float(min(ginis)),
              "pooled_gini": score(np.concatenate(ys), np.concatenate(ps))["gini"],
              "folds": fold_results, "segments": segment_results, "constraints": constraints,
              "importance": importance.tolist(), "memory_peak_bytes": monitor.peak_private_bytes}
    cache.put(task["key"], result)
    progress.debug("Candidate evaluation completed", candidate=task["key"][:12], fidelity=task["fidelity"],
                   n_features=result["n_features"], mean_gini=result["mean_gini"],
                   elapsed_seconds=time.perf_counter() - started)
    return result

class Evaluator:
    """Cached evaluator with canonical reference, fidelity and resource admission."""
    def __init__(self, data_path: str | Path, folds: list, directory: str | Path,
                 model_params: dict | None = None, num_boost_round: int = 100,
                 config: EvaluationConfig | None = None, resources: ResourceConfig | None = None,
                 segment_constraints: list[SegmentConstraint] | None = None, cancelled=None,
                 expected_identity: str | None = None, progress=None):
        self.progress = progress or NULL_PROGRESS
        self.directory = Path(directory).resolve()
        self.data_path = str(Path(data_path).resolve())
        self.data = load_dataset(self.data_path)
        self.config = config or EvaluationConfig()
        self.num_boost_round = num_boost_round
        if not isinstance(num_boost_round, int) or isinstance(num_boost_round, bool) or num_boost_round < 1:
            raise ValueError("num_boost_round must be a positive integer")
        self.model_params = model_params
        strict_params(model_params, seed=self.config.estimator_seed, threads=1, n_rows=len(self.data.y))
        self.cache_path = str(self.directory / "evaluations.sqlite")
        self.cache = EvaluationCache(self.cache_path)
        self.all_results = {}
        self.fold_specs = []
        fold_dir = self.directory / "folds"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for i, fold in enumerate(folds):
            paths = {"label": str(fold.label)}
            for kind in ("train", "valid"):
                path = fold_dir / f"{i}_{kind}.npy"
                values = np.asarray(getattr(fold, kind))
                if path.exists():
                    if not np.array_equal(np.load(path, mmap_mode="r"), values):
                        raise ValueError(f"Persisted fold differs from the requested definition: {path.name}")
                else:
                    np.save(path, values)
                paths[f"{kind}_path"] = str(path)
            self.fold_specs.append(paths)
        self.fold_identity = [{kind: hashlib.sha256(np.asarray(getattr(f, kind), dtype="<i8").tobytes()).hexdigest()
                               for kind in ("train", "valid")} | {"label": str(f.label)} for f in folds]
        # Hash once; per-candidate keys use this compact digest.
        self.active_constraints = []
        self.excluded_constraints = []
        for constraint in segment_constraints or []:
            if constraint.column not in self.data.segments:
                raise ValueError(f"Constraint column {constraint.column!r} must be in segment_cols")
            if not pd.api.types.is_scalar(constraint.value):
                raise ValueError("Segment constraint value must be a scalar category, or None for missing values")
            normalized_constraint = asdict(constraint)
            if pd.isna(constraint.value):
                normalized_constraint["value"] = None
            selected = _segment_mask(self.data.segments[constraint.column], normalized_constraint["value"])
            failures = []
            for fold in folds:
                yy = self.data.y[fold.valid][selected[fold.valid]]
                ev = int(yy.sum())
                if ev < self.config.min_segment_events or len(yy) - ev < self.config.min_segment_nonevents:
                    failures.append(str(fold.label))
            if failures:
                info = {**normalized_constraint, "failed_folds": failures,
                        "min_events": self.config.min_segment_events, "min_nonevents": self.config.min_segment_nonevents}
                self.excluded_constraints.append(info)
                self.progress.warning("Segment constraint disabled for insufficient validation support",
                                      failed_folds=len(failures), min_events=self.config.min_segment_events,
                                      min_nonevents=self.config.min_segment_nonevents)
                warnings.warn(f"Disabling segment constraint {constraint.column}={constraint.value!r}: requires >= {self.config.min_segment_events} events and >= {self.config.min_segment_nonevents} non-events in folds {failures}", UserWarning, stacklevel=2)
            else:
                self.active_constraints.append(normalized_constraint)
        # Resolve estimator threads once. Admission may vary worker count, but
        # reference and candidate training configurations remain identical.
        estimate = int(self.data.X.nbytes * 5 + 128 * 1024**2)
        self.manager = ResourceManager(resources or ResourceConfig(), estimate, temp_dir=self.directory)
        self.training_threads = self.manager.plan().threads_per_worker
        self.manager.bound_threads = self.training_threads
        self.versions = {name: version(name) for name in ("lightgbm", "numpy", "pandas", "scikit-learn", "pymoo", "joblib", "pyarrow")}
        self.identity = stable_hash({"dataset": self.data.fingerprint, "folds": self.fold_identity,
                                    "params": model_params, "rounds": num_boost_round, "evaluation": asdict(self.config),
                                    "constraints": self.active_constraints, "versions": self.versions,
                                    "training_threads": self.training_threads, "implementation": EVALUATOR_VERSION})
        if expected_identity is not None and self.identity != expected_identity:
            raise ValueError("Checkpoint incompatible with current evaluator, threads, data or settings")
        self.progress.info("Evaluator initialized", identity=self.identity[:12], folds=len(self.fold_specs),
                           n_rows=len(self.data.y), candidate_features=self.data.X.shape[1],
                           training_threads=self.training_threads, active_constraints=len(self.active_constraints),
                           excluded_constraints=len(self.excluded_constraints))
        self.scheduler = ResourceScheduler(self.manager, cancelled=cancelled, progress=self.progress)
        calibration_started = time.perf_counter()
        self.progress.milestone("Full-feature reference calibration started", n_features=self.data.X.shape[1],
                           folds=len(self.fold_specs), num_boost_round=self.num_boost_round)
        self.reference = self.evaluate_many([np.ones(self.data.X.shape[1], dtype=bool)], "full", reference=True)[0]
        self.manager.calibrate(self.reference["memory_peak_bytes"])
        self.progress.milestone("Full-feature reference calibration completed", mean_gini=self.reference["mean_gini"],
                           measured_private_peak_bytes=self.reference["memory_peak_bytes"],
                           estimated_task_bytes=self.manager.estimated_task_bytes,
                           elapsed_seconds=time.perf_counter() - calibration_started)

    def evaluate_many(self, masks, fidelity: str = "full", reference: bool = False) -> list[dict]:
        """Evaluate unique valid masks, with final fold always included at low fidelity."""
        started = time.perf_counter()
        if fidelity == "full":
            folds = self.fold_specs
        elif fidelity.startswith("last_"):
            count = int(fidelity[5:])
            if count < 1 or count > len(self.fold_specs):
                raise ValueError("Reduced fidelity must use between one and all available folds")
            folds = self.fold_specs[-count:]
        else:
            raise ValueError(f"Unknown evaluation fidelity {fidelity!r}")
        tasks, keys, cache_hits, reference_reuses, feature_counts = {}, [], set(), set(), set()
        pending_duplicates = 0
        plan = self.manager.plan()
        for values in masks:
            mask = np.asarray(values)
            if mask.shape != (self.data.X.shape[1],) or not np.isin(mask, [0, 1]).all() or not mask.any():
                raise ValueError("Feature mask must be nonempty binary vector matching predictor count")
            mask = mask.astype(bool)
            feature_counts.add(int(mask.sum()))
            # Reference calculations have no self-comparison constraints.
            key = stable_hash({"evaluation": self.identity, "mask": mask.tolist(), "fidelity": fidelity,
                               "reference": reference, "threads": self.training_threads})
            keys.append(key)
            if key in tasks:
                pending_duplicates += 1
                continue
            if key in reference_reuses:
                continue
            value = self.cache.get(key)
            if value is not None:
                self.all_results[key] = value
                cache_hits.add(key)
                self.progress.debug("Evaluation cache hit", candidate=key[:12], fidelity=fidelity,
                                    n_features=int(mask.sum()))
                continue
            if not reference and fidelity == "full" and mask.all():
                # This is the same set of full-row model fits as the calibrated
                # reference; only the self-comparison constraint vector differs.
                value = deepcopy(self.reference)
                value.update(key=key, constraints=[-c["max_gini_drop"] for c in self.active_constraints])
                self.cache.put(key, value)
                self.all_results[key] = value
                reference_reuses.add(key)
                self.progress.debug("Full-feature candidate reused reference", candidate=key[:12], fidelity=fidelity)
                continue
            self.cache.pending(key)
            tasks[key] = {"key": key, "mask": mask.tolist(), "data_path": self.data_path,
                          "dataset_fingerprint": self.data.fingerprint,
                          "cache_path": self.cache_path, "folds": folds, "fidelity": fidelity,
                          "threads": self.training_threads, "model_params": self.model_params,
                          "reference": reference, "progress": self.progress.descriptor,
                          "seed": self.config.estimator_seed, "num_boost_round": self.num_boost_round,
                          "active_constraints": [] if reference else self.active_constraints,
                          "reference_segments": [] if reference else self.reference["segments"]}
        # A single reference task is serial while still passing resource admission.
        # Every successful task commits its cache entry before returning.
        self.progress.info("Evaluation batch prepared", requested_candidates=len(keys), unique_candidates=len(set(keys)),
                           new_evaluations=len(tasks), cache_hits=len(cache_hits), reference_reuses=len(reference_reuses),
                           duplicate_requests=len(keys) - len(set(keys)), pending_duplicates=pending_duplicates,
                           fidelity=fidelity, feature_counts=sorted(feature_counts))
        values = self.scheduler.map(_worker, list(tasks.values()))
        for value in values:
            self.all_results[value["key"]] = value
        self.progress.info("Evaluation batch completed", requested_candidates=len(keys), evaluated_candidates=len(values),
                           cache_hits=len(cache_hits), fidelity=fidelity, elapsed_seconds=time.perf_counter() - started)
        return [self.all_results[key] for key in keys]


# === Source: memetic_fs/optimization/archives.py ===
"""Common-fidelity feasible Pareto archives backed by pymoo."""

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV


def pareto_front(results, fidelity="full"):
    """Return unique feasible nondominated results for one exact fidelity."""
    unique = {}
    for result in results:
        if result["fidelity"] != fidelity or any(value > 0 for value in result.get("constraints", [])):
            continue
        if not np.isfinite(result["mean_gini"]):
            raise ValueError("Undefined Gini must not enter a Pareto archive")
        unique[tuple(result["mask"])] = result
    values = list(unique.values())
    if not values:
        return []
    objectives = np.array([[-r["mean_gini"], r["n_features"]] for r in values])
    front = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    return sorted([values[i] for i in front], key=lambda r: (r["n_features"], -r["mean_gini"]))


def normalized_hypervolume(front, n_features):
    """Fixed-bounds hypervolume: Gini in [-1,1], cardinality in [0,p]."""
    if not front:
        return 0.0
    objectives = np.array([[(1-r["mean_gini"])/2, r["n_features"]/n_features] for r in front])
    return float(HV(ref_point=np.array([1.01, 1.01]))(objectives))


# === Source: memetic_fs/optimization/operators.py ===
"""Cardinality-aware binary proposals used by initialization and local search."""


from collections.abc import Mapping, Sequence
import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair


def validate_mask(mask, n_features: int, bounds: tuple[int, int]) -> np.ndarray:
    """Validate a mask without silently accepting arbitrary numeric values."""
    values = np.asarray(mask)
    if values.ndim != 1 or len(values) != n_features or not np.isin(values, [0, 1]).all():
        raise ValueError(f"Feature mask must contain exactly {n_features} binary values")
    result = values.astype(bool)
    if not bounds[0] <= result.sum() <= bounds[1]:
        raise ValueError(f"Feature mask cardinality must be within {bounds}")
    return result


def cardinality_probabilities(bounds, exponent=1.3, smoothing=1.0, uniform_probability=0.15):
    """Inverse-cardinality distribution with a strictly positive uniform floor."""
    counts = np.arange(bounds[0], bounds[1] + 1)
    weights = (counts + smoothing).astype(float) ** (-exponent)
    weights /= weights.sum()
    return counts, (1 - uniform_probability) * weights + uniform_probability / len(counts)


def repair_mask(mask, bounds, rng):
    """Change only enough randomly chosen bits to satisfy cardinality bounds."""
    result = np.asarray(mask, dtype=bool).copy()
    if result.sum() < bounds[0]:
        result[rng.choice(np.flatnonzero(~result), bounds[0] - int(result.sum()), replace=False)] = True
    elif result.sum() > bounds[1]:
        result[rng.choice(np.flatnonzero(result), int(result.sum()) - bounds[1], replace=False)] = False
    return result


def propose_neighbor(mask, bounds, rng, operation=None, importance=None, exploration=0.3, groups=None):
    """Add, remove or exchange features, occasionally using supplied feature groups.

    The random branch gives every available feature a nonzero probability even
    when importance-guided proposals are enabled.
    """
    result = np.asarray(mask, dtype=bool).copy()
    selected, absent = np.flatnonzero(result), np.flatnonzero(~result)
    operation = operation or rng.choice(["add", "remove", "swap", "swap2", "group"])
    guided = importance is not None and rng.random() >= exploration
    scores = np.maximum(np.asarray(importance, dtype=float), 0) if importance is not None else None

    def choose(indices, size, add):
        size = min(size, len(indices))
        if size == 0:
            return np.array([], dtype=int)
        probability = None
        if guided:
            weights = scores[indices] + 1e-12
            if not add:
                weights = 1 / weights
            probability = weights / weights.sum()
        return rng.choice(indices, size=size, replace=False, p=probability)

    if operation == "add" and len(absent) and len(selected) < bounds[1]:
        result[choose(absent, 1, True)] = True
    elif operation == "remove" and len(selected) > bounds[0]:
        result[choose(selected, 1, False)] = False
    else:
        size = 2 if operation == "swap2" else (3 if operation == "group" else 1)
        if operation == "group" and groups:
            candidates = [np.asarray(g, dtype=int) for g in groups if len(g)]
            if candidates:
                group = candidates[int(rng.integers(len(candidates)))]
                group_absent = group[~result[group]]
                size = min(len(group_absent), len(selected), 3)
                if size:
                    result[choose(selected, size, False)] = False
                    result[rng.choice(group_absent, size, replace=False)] = True
                    return repair_mask(result, bounds, rng)
        size = min(size, len(selected), len(absent))
        result[choose(selected, size, False)] = False
        result[choose(absent, size, True)] = True
    return repair_mask(result, bounds, rng)


class CardinalityRepair(Repair):
    """pymoo repair preserving valid binary subsets."""

    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng(0)
        return np.asarray([repair_mask(mask, self.bounds, rng) for mask in X])


class SubsetMutation(Mutation):
    """Guided/random exchanges plus inverse-cardinality reset proposals."""

    def __init__(self, bounds, config, importance=None, groups=None):
        super().__init__(prob=1.0)
        self.bounds, self.config = bounds, config
        self.importance, self.groups = importance, groups

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng(0)
        result = []
        for mask in X:
            if rng.random() < 0.15:
                counts, probability = cardinality_probabilities(
                    self.bounds, self.config.inverse_exponent, self.config.inverse_smoothing,
                    self.config.uniform_cardinality_probability)
                k = int(rng.choice(counts, p=probability))
                proposal = np.zeros(len(mask), dtype=bool)
                proposal[rng.choice(len(mask), k, replace=False)] = True
            else:
                proposal = propose_neighbor(mask, self.bounds, rng, importance=self.importance,
                                            exploration=self.config.exploration_probability, groups=self.groups)
            result.append(proposal)
        return np.asarray(result)


# === Source: memetic_fs/optimization/initialization.py ===
"""Diverse random, importance, family and recursive-elimination seeds."""

from math import comb
import time
import numpy as np




def normalize_subsets(subsets, names, bounds):
    """Accept feature-name subsets or binary masks and reject invalid seeds."""
    result, seen = [], set()
    for subset in subsets or []:
        if all(isinstance(value, str) for value in subset):
            unknown = set(subset) - set(names)
            if unknown:
                raise ValueError(f"Unknown features in initial subset: {sorted(unknown)}")
            if len(set(subset)) != len(subset):
                raise ValueError("Initial subset contains duplicated feature names")
            mask = validate_mask([name in subset for name in names], len(names), bounds)
        else:
            mask = validate_mask(subset, len(names), bounds)
        key = tuple(mask)
        if key not in seen:
            seen.add(key)
            result.append(mask)
    return result


def rfe_seeds(evaluator, n_features, bounds, sizes):
    """Batch RFE, evaluating each retained checkpoint under full temporal CV."""
    targets = sorted({int(k) for k in sizes if bounds[0] <= k <= bounds[1]}, reverse=True)
    if not targets:
        targets = [bounds[1], bounds[0]]
    current = np.ones(n_features, dtype=bool)
    evaluation = evaluator.reference
    seeds, trajectory = [], []
    progress = get_progress(evaluator)
    started = time.perf_counter()
    progress.milestone("RFE initialization started", targets=targets, initial_features=n_features)
    for target in targets:
        progress.info("RFE target started", target_features=target, current_features=int(current.sum()))
        while current.sum() > target:
            # Large steps far from checkpoints; every checkpoint gets its own fit.
            step = min(int(current.sum()) - target, max(1, int(current.sum() * 0.2)))
            selected = np.flatnonzero(current)
            importance = np.asarray(evaluation["importance"])
            before = int(current.sum())
            current[selected[np.argsort(importance[selected], kind="stable")[:step]]] = False
            progress.info("RFE batch started", target_features=target, features_before=before,
                          features_after=int(current.sum()), removed_features=step, fidelity="full")
            batch_started = time.perf_counter()
            evaluation = evaluator.evaluate_many([current], fidelity="full")[0]
            progress.info("RFE batch completed", selected_features=int(current.sum()),
                          mean_gini=evaluation["mean_gini"],
                          elapsed_seconds=time.perf_counter()-batch_started)
            trajectory.append(evaluation)
        seeds.append(current.copy())
        if not trajectory or trajectory[-1]["mask"] != evaluation["mask"]:
            trajectory.append(evaluation)
        progress.info("RFE target completed", target_features=target, mean_gini=evaluation["mean_gini"])
    progress.milestone("RFE initialization completed", seeds=len(seeds), evaluation_steps=len(trajectory),
                  elapsed_seconds=time.perf_counter()-started)
    return seeds, trajectory


def initial_population(n_features, bounds, count, rng, config, importance, user=None, rfe=None, groups=None):
    """Mix cardinality bands, importance ranks, family coverage and random subsets."""
    possible = sum(comb(n_features, k) for k in range(bounds[0], bounds[1]+1))
    count = min(count, possible)
    seeds, seen = [], set()

    def add(mask, diverse=False):
        mask = validate_mask(mask, n_features, bounds)
        key = tuple(mask)
        if key in seen:
            return
        if diverse and seeds and min(np.mean(mask != other) for other in seeds[-20:]) < 1/n_features:
            return
        seen.add(key)
        seeds.append(mask.copy())

    for mask in list(user or []) + list(rfe or []):
        add(mask)
    ranked = np.argsort(-np.asarray(importance), kind="stable")
    for k in sorted(set(np.linspace(bounds[0], bounds[1], min(count, 8), dtype=int))):
        mask = np.zeros(n_features, dtype=bool)
        mask[ranked[:k]] = True
        add(mask)
        # Rotate through supplied feature families, breaking importance ties randomly.
        order = []
        if groups:
            remaining = [list(rng.permutation(group)) for group in groups]
            while any(remaining):
                for group in remaining:
                    if group:
                        feature = group.pop()
                        if feature not in order:
                            order.append(feature)
        order += [i for i in rng.permutation(n_features) if i not in order]
        mask = np.zeros(n_features, dtype=bool)
        mask[order[:k]] = True
        add(mask)
    counts, probability = cardinality_probabilities(bounds, config.inverse_exponent,
        config.inverse_smoothing, config.uniform_cardinality_probability)
    attempts = 0
    while len(seeds) < count and attempts < max(500, 100*count):
        k = int(rng.choice(counts, p=probability))
        mask = np.zeros(n_features, dtype=bool)
        mask[rng.choice(n_features, k, replace=False)] = True
        add(mask, diverse=True)
        attempts += 1
    # For small spaces enumerate as a bounded fallback; never hang on duplicates.
    if len(seeds) < count and n_features <= 16:
        from itertools import combinations
        for k in range(bounds[0], bounds[1]+1):
            for selected in combinations(range(n_features), k):
                mask = np.zeros(n_features, dtype=bool)
                mask[list(selected)] = True
                add(mask)
                if len(seeds) >= count:
                    break
            if len(seeds) >= count:
                break
    return np.asarray(seeds[:max(count, len(user or []))], dtype=bool)


# === Source: memetic_fs/optimization/local_search.py ===
"""Budgeted diverse local refinement allowing removed features to return."""

import numpy as np



def local_neighbors(front, bounds, rng, budget, candidate_count, importance, exploration, groups=None):
    """Generate distinct neighborhoods spread over frontier cardinalities."""
    if not front or budget <= 0:
        return np.empty((0, len(importance)), dtype=bool)
    indices = np.unique(np.linspace(0, len(front)-1, min(candidate_count, len(front)), dtype=int))
    seeds = [np.asarray(front[i]["mask"], dtype=bool) for i in indices]
    seen = {tuple(mask) for mask in seeds}
    neighbors = []
    operations = ["add", "remove", "swap", "swap2", "group"]
    for attempt in range(max(100, budget*30)):
        if len(neighbors) >= budget:
            break
        seed = seeds[attempt % len(seeds)]
        mask = propose_neighbor(seed, bounds, rng, operations[(attempt//len(seeds)) % len(operations)],
                                importance, exploration, groups)
        if tuple(mask) not in seen:
            seen.add(tuple(mask))
            neighbors.append(mask)
    return np.asarray(neighbors, dtype=bool).reshape((-1, len(importance)))


# === Source: memetic_fs/optimization/problem.py ===
"""Vectorized pymoo problem; evaluation and scheduling belong to the evaluator."""

import numpy as np
from pymoo.core.problem import Problem


class FeatureSubsetProblem(Problem):
    """Minimize negative fold-mean Gini and selected-feature count."""

    def __init__(self, evaluator, n_features, n_constraints, on_results=None):
        super().__init__(n_var=n_features, n_obj=2, n_ieq_constr=n_constraints,
                         xl=0, xu=1, vtype=bool)
        self.fitness_evaluator = evaluator
        self.fidelity = "full"
        self.on_results = on_results

    def _evaluate(self, X, out, *args, **kwargs):
        results = self.fitness_evaluator.evaluate_many(X, fidelity=self.fidelity)
        if self.on_results:
            self.on_results(results)
        out["F"] = np.array([[-r["mean_gini"], r["n_features"]] for r in results])
        if self.n_ieq_constr:
            out["G"] = np.array([r["constraints"] for r in results])


# === Source: memetic_fs/optimization/algorithm.py ===
"""Restartable pymoo NSGA-II with full-fidelity confirmation and local search."""


from collections.abc import Callable
from math import comb
import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.pntx import TwoPointCrossover









class SearchEngine:
    """Single-machine memetic search, with no optimizer-object deserialization.

    Evaluators implement ``evaluate_many(masks, fidelity)`` and expose the
    full-feature ``reference`` result. Checkpoint callbacks receive JSON state
    only at completed generation boundaries. Cached interrupted work is replayed
    deterministically from the preceding boundary.
    """

    STATE_VERSION = 1

    def __init__(self, evaluator, n_features, feature_range, config, random_state=42,
                 use_rfe_initialization=True, initial_subsets=None, feature_names=None,
                 feature_groups=None):
        self.evaluator = evaluator
        self.progress = get_progress(evaluator)
        self.n_features = n_features
        self.bounds = tuple(feature_range)
        self.config = config
        self.seed = random_state
        self.use_rfe = use_rfe_initialization
        self.names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(n_features)]
        self.user_seeds = normalize_subsets(initial_subsets, self.names, self.bounds)
        self.groups = self._groups(feature_groups)
        self.rng = np.random.default_rng(random_state)
        self.results = {}
        self.rfe_trajectory = []
        self.history = []
        self.algorithm = None
        self.problem = None
        self.last_state = None

    def _groups(self, groups):
        if groups is None:
            return None
        if isinstance(groups, dict):
            # Mapping feature -> family is convenient for both reporting/search.
            if all(name in self.names for name in groups):
                families = {}
                for name, family in groups.items():
                    families.setdefault(str(family), []).append(self.names.index(name))
                return list(families.values())
            groups = list(groups.values())
        normalized = []
        for group in groups:
            indices = [self.names.index(i) if isinstance(i, str) else int(i) for i in group]
            if any(i < 0 or i >= self.n_features for i in indices):
                raise ValueError("Feature-group index is out of bounds")
            normalized.append(list(dict.fromkeys(indices)))
        return normalized

    def _record(self, results):
        for result in results:
            self.results[(result["fidelity"], tuple(result["mask"]))] = result

    def _evaluate(self, masks, fidelity="full"):
        results = self.evaluator.evaluate_many(masks, fidelity=fidelity)
        self._record(results)
        return results

    def _front(self, fidelity="full"):
        results = [r for r in self.results.values() if self.bounds[0] <= r["n_features"] <= self.bounds[1]]
        return pareto_front(results, fidelity)

    def _confirm(self, fidelity):
        if fidelity == "full":
            return
        front = self._front(fidelity)
        if not front:
            # A provisional constraint failure must not prevent discovering full-CV feasibility.
            candidates = [r for r in self.results.values() if r["fidelity"] == fidelity]
            front = sorted(candidates, key=lambda r: sum(max(0., g) for g in r.get("constraints", [])))[:self.config.population_size]
        if front:
            self.progress.info("Confirming provisional candidates with full temporal CV",
                               candidates=len(front), provisional_fidelity=fidelity,
                               confirmed_before=len(self._front("full")))
            self._evaluate([r["mask"] for r in front], "full")
            self.progress.debug("Provisional candidate confirmation completed",
                                confirmed_count=len(self._front("full")))

    def _make_algorithm(self, sampling):
        constraints = len(self.evaluator.active_constraints) if hasattr(self.evaluator, "active_constraints") else len(self.evaluator.reference.get("constraints", []))
        self.problem = FeatureSubsetProblem(self.evaluator, self.n_features, constraints, self._record)
        importance = self.evaluator.reference["importance"]
        possible = sum(comb(self.n_features, k) for k in range(self.bounds[0], self.bounds[1]+1))
        pop_size = min(self.config.population_size, possible)
        self.algorithm = NSGA2(pop_size=pop_size, sampling=sampling,
            crossover=TwoPointCrossover(prob=0.9),
            mutation=SubsetMutation(self.bounds, self.config, importance, self.groups),
            repair=CardinalityRepair(self.bounds), eliminate_duplicates=True)
        self.algorithm.setup(self.problem, seed=self.seed, termination=NoTermination(), verbose=False)

    def _population(self, results):
        pop = Population.new("X", np.asarray([r["mask"] for r in results], dtype=bool),
                             "F", np.array([[-r["mean_gini"], r["n_features"]] for r in results]))
        if self.problem.n_ieq_constr:
            pop.set("G", np.array([r["constraints"] for r in results]))
        for individual in pop:
            individual.evaluated.update(["F", "G", "H"])
        return pop

    def _refresh_population(self, fidelity):
        results = self._evaluate(self.algorithm.pop.get("X"), fidelity)
        pop = self._population(results)
        self.algorithm.pop = self.algorithm.survival.do(self.problem, pop,
            n_survive=len(pop), random_state=self.algorithm.random_state, algorithm=self.algorithm)
        self.algorithm._set_optimum()

    def _refine(self, fidelity):
        front = self._front("full")
        before_hv = normalized_hypervolume(front, self.n_features)
        before_masks = {tuple(result["mask"]) for result in front}
        started = time.perf_counter()
        self.progress.info("Local search started", confirmed_candidates=len(front),
                           seeds=min(self.config.local_search_candidates,len(front)),
                           evaluation_budget=self.config.local_search_budget, population_fidelity=fidelity)
        neighbors = local_neighbors(front, self.bounds, self.rng, self.config.local_search_budget,
            self.config.local_search_candidates, self.evaluator.reference["importance"],
            self.config.exploration_probability, self.groups)
        if len(neighbors) == 0:
            self.progress.info("Local search completed", evaluated_neighbors=0,
                               new_frontier_candidates=0, hypervolume_improvement=0.0,
                               elapsed_seconds=time.perf_counter()-started)
            return 0
        self.progress.debug("Local search neighbors generated", unique_neighbors=len(neighbors),
                            evaluation_fidelity="full")
        full_results = self._evaluate(neighbors, "full")
        results = full_results if fidelity == "full" else self._evaluate(neighbors, fidelity)
        pop = Population.merge(self.algorithm.pop, self._population(results))
        # Duplicate elimination before survival avoids filling the population with a cached neighbor.
        pop = self.algorithm.eliminate_duplicates.do(pop)
        self.algorithm.pop = self.algorithm.survival.do(self.problem, pop,
            n_survive=self.algorithm.pop_size, algorithm=self.algorithm,
            random_state=self.algorithm.random_state)
        self.algorithm._set_optimum()
        after_front = self._front("full")
        self.progress.info("Local search completed", evaluated_neighbors=len(neighbors),
                           new_frontier_candidates=sum(tuple(result["mask"]) not in before_masks for result in after_front),
                           hypervolume_improvement=normalized_hypervolume(after_front,self.n_features)-before_hv,
                           elapsed_seconds=time.perf_counter()-started)
        return len(neighbors)

    def _serialize(self, generation, best_hv, stalled, fidelity, stop_reason=None):
        population = []
        if self.algorithm.pop is not None:
            for individual in self.algorithm.pop:
                crowding = individual.get("crowding")
                population.append({"mask": individual.X.astype(bool).tolist(),
                    "F": individual.F.tolist(), "G": individual.G.tolist(),
                    "rank": None if individual.get("rank") is None else int(individual.get("rank")),
                    "crowding": None if crowding is None else ("inf" if np.isinf(crowding) else float(crowding))})
        return {"state_version": self.STATE_VERSION, "generation": generation,
            "population": population, "rng_state": self.rng.bit_generator.state,
            "optimizer_rng_state": self.algorithm.random_state.bit_generator.state,
            "evaluations": list(self.results.values()), "confirmed": self._front("full"),
            "provisional": self._front(fidelity) if fidelity != "full" else [],
            "history": self.history, "rfe_trajectory": self.rfe_trajectory,
            "best_hypervolume": best_hv, "stalled": stalled, "fidelity": fidelity,
            "stop_reason": stop_reason, "n_features": self.n_features,
            "feature_range": list(self.bounds), "optimizer_evaluations": self.algorithm.evaluator.n_eval}

    def _restore(self, state):
        if state.get("state_version") != self.STATE_VERSION:
            raise ValueError("Unsupported optimizer checkpoint state version")
        if state.get("n_features") != self.n_features or tuple(state["feature_range"]) != self.bounds:
            raise ValueError("Optimizer checkpoint feature space is incompatible")
        self._make_algorithm(np.array([p["mask"] for p in state["population"]], dtype=bool))
        values = state["population"]
        pop = Population.new("X", np.array([p["mask"] for p in values], dtype=bool),
                             "F", np.array([p["F"] for p in values]))
        if self.problem.n_ieq_constr:
            pop.set("G", np.array([p["G"] for p in values]))
        for individual, value in zip(pop, values):
            individual.set("rank", value["rank"])
            individual.set("crowding", np.inf if value["crowding"] == "inf" else value["crowding"])
            individual.evaluated.update(["F", "G", "H"])
        self.algorithm.pop = pop
        self.algorithm.is_initialized = True
        self.algorithm.n_iter = state["generation"] + 1
        self.algorithm.start_time = time.time()
        self.algorithm.evaluator.n_eval = state.get("optimizer_evaluations", 0)
        self.algorithm.random_state.bit_generator.state = state["optimizer_rng_state"]
        self.rng.bit_generator.state = state["rng_state"]
        self.algorithm._set_optimum()
        self.history = list(state["history"])
        self.rfe_trajectory = state.get("rfe_trajectory", [])
        self._record(state.get("evaluations", []))
        self.problem.fidelity = state["fidelity"]
        self.last_state = state

    @staticmethod
    def _cancelled(cancel):
        if cancel is None:
            return False
        return bool(cancel() if callable(cancel) else cancel.is_set())

    def run(self, checkpoint_callback: Callable | None = None, state=None, cancel=None):
        """Execute until confirmed-front convergence, safety limit, or cancellation."""
        self._run_started = time.perf_counter()
        self.progress.milestone("Optimization started", max_generations=self.config.max_generations,
                           population_size=self.config.population_size, candidate_features=self.n_features,
                           min_features=self.bounds[0], max_features=self.bounds[1],
                           resuming=state is not None)
        if state is not None:
            self._restore(state)
            self.progress.milestone("Optimization checkpoint restored", completed_generation=state["generation"],
                               population_size=len(self.algorithm.pop), fidelity=state["fidelity"],
                               completed_evaluations=len(self.results),
                               confirmed_count=len(self._front("full")))
            if state.get("stop_reason") in {"pareto_convergence", "search_space_exhausted"}:
                self.progress.milestone("Optimization already completed; retaining checkpoint result",
                                   reason=state["stop_reason"], completed_generation=state["generation"])
                return self._result(state["stop_reason"])
            start = state["generation"] + 1
            best_hv, stalled = state["best_hypervolume"], state["stalled"]
            previous_fidelity = state["fidelity"]
        else:
            if self._cancelled(cancel):
                self.progress.warning("Optimization cancelled before initialization",
                                      completed_generation=0, completed_evaluations=len(self.results))
                raise InterruptedError("Optimization cancelled before initialization; completed evaluations remain cached")
            rfe = []
            if self.use_rfe:
                try:
                    rfe, self.rfe_trajectory = rfe_seeds(self.evaluator, self.n_features, self.bounds, self.config.rfe_seed_sizes)
                except (KeyboardInterrupt, InterruptedError):
                    self.progress.warning("RFE initialization interrupted; completed evaluations remain cached",
                                          elapsed_seconds=time.perf_counter()-self._run_started)
                    raise
                self._record(self.rfe_trajectory)
            sampling = initial_population(self.n_features, self.bounds, self.config.population_size,
                self.rng, self.config, self.evaluator.reference["importance"], self.user_seeds, rfe, self.groups)
            cardinalities = sampling.sum(axis=1)
            self.progress.milestone("Population initialization completed", user_seeds=len(self.user_seeds),
                               rfe_seeds=len(rfe), unique_candidates=len(sampling),
                               min_features=int(cardinalities.min()), mean_features=float(cardinalities.mean()),
                               max_features=int(cardinalities.max()))
            self._make_algorithm(sampling)
            start, best_hv, stalled = 1, 0.0, 0
            previous_fidelity = None
        stop_reason = "max_generations"
        generation = start - 1
        fidelity = previous_fidelity or "full"
        try:
            for generation in range(start, self.config.max_generations+1):
                if self._cancelled(cancel):
                    self.progress.warning("Optimization cancellation requested",
                                          completed_generation=generation-1,
                                          completed_evaluations=len(self.results))
                    if self.last_state is None:
                        raise InterruptedError("Optimization cancelled before a completed generation")
                    stop_reason = "cancelled"
                    generation -= 1
                    break
                fidelity = f"last_{self.config.reduced_fold_count}" if generation <= self.config.reduced_generations else "full"
                generation_started = time.perf_counter()
                self.progress.generation(generation)
                self.progress.info("Generation started", generation=generation,
                                   max_generations=self.config.max_generations, fidelity=fidelity,
                                   phase="initial_population" if not self.algorithm.is_initialized else "offspring")
                self.progress.milestone(f"Running generation {generation}", fidelity=fidelity)
                self.problem.fidelity = fidelity
                if previous_fidelity is not None and previous_fidelity != fidelity:
                    # Existing parents and new offspring must share one fitness basis.
                    self.progress.info("Changing fidelity; reevaluating all parents before survival",
                                       generation=generation, previous_fidelity=previous_fidelity,
                                       fidelity=fidelity, parents=len(self.algorithm.pop))
                    self._refresh_population(fidelity)
                offspring = self.algorithm.ask()
                if offspring is None or len(offspring) == 0:
                    self.progress.info("No distinct offspring remain in the search space",
                                       completed_generation=generation-1)
                    stop_reason = "search_space_exhausted"
                    generation -= 1
                    break
                self.progress.debug("Evaluating generation candidates", generation=generation,
                                    candidates=len(offspring), fidelity=fidelity)
                self.algorithm.evaluator.eval(self.problem, offspring)
                self.algorithm.tell(infills=offspring)
                self._confirm(fidelity)
                neighbors = self._refine(fidelity) if generation % self.config.local_search_interval == 0 else 0
                full_front = self._front("full")
                hypervolume = normalized_hypervolume(full_front, self.n_features)
                if hypervolume > best_hv + self.config.min_improvement:
                    best_hv, stalled = hypervolume, 0
                else:
                    stalled += 1
                counts = self.algorithm.pop.get("X").sum(axis=1)
                self.history.append({"generation": generation, "fidelity": fidelity,
                    "confirmed_hypervolume": hypervolume,
                    "best_confirmed_gini": max((r["mean_gini"] for r in full_front), default=None),
                    "unique_evaluations": len(self.results), "confirmed_count": len(full_front),
                    "provisional_count": len(self._front(fidelity)) if fidelity != "full" else 0,
                    "feature_count_min": int(counts.min()), "feature_count_max": int(counts.max()),
                    "feature_count_mean": float(counts.mean()), "local_search_evaluations": neighbors,
                    "stalled_generations": stalled, "converged": False})
                previous_fidelity = fidelity
                if generation >= self.config.min_generations and stalled >= self.config.patience:
                    stop_reason = "pareto_convergence"
                    self.history[-1]["converged"] = True
                    self.progress.milestone("Confirmed Pareto frontier converged", generation=generation,
                                       stalled_generations=stalled, patience=self.config.patience,
                                       minimum_improvement=self.config.min_improvement,
                                       confirmed_hypervolume=hypervolume)
                self.progress.info("Generation completed", generation=generation, fidelity=fidelity,
                                   best_confirmed_gini=self.history[-1]["best_confirmed_gini"],
                                   confirmed_hypervolume=hypervolume, confirmed_count=len(full_front),
                                   provisional_count=self.history[-1]["provisional_count"],
                                   unique_evaluations=len(self.results),
                                   min_features=int(counts.min()), mean_features=float(counts.mean()),
                                   max_features=int(counts.max()), stalled_generations=stalled,
                                   generation_seconds=time.perf_counter()-generation_started,
                                   elapsed_seconds=time.perf_counter()-self._run_started)
                self.progress.generation_completed(generation)
                self.progress.milestone(f"Generation {generation} completed",
                                        best_confirmed_gini=self.history[-1]["best_confirmed_gini"],
                                        confirmed_hypervolume=hypervolume)
                checkpoint_reason = "running" if stop_reason == "max_generations" and generation < self.config.max_generations else stop_reason
                self.last_state = self._serialize(generation, best_hv, stalled, fidelity, checkpoint_reason)
                if checkpoint_callback:
                    checkpoint_callback(self.last_state)
                    self.progress.debug("Generation checkpoint completed", generation=generation)
                if stop_reason == "pareto_convergence":
                    break
        except KeyboardInterrupt:
            stop_reason = "cancelled"
            self.progress.warning("Optimization interrupted; completed evaluations remain cached",
                                  completed_generation=self.last_state["generation"] if self.last_state else 0,
                                  completed_evaluations=len(self.results),
                                  elapsed_seconds=time.perf_counter()-self._run_started)
            # The preceding generation boundary remains replayable; cache contains completed jobs.
            if self.last_state is not None:
                generation = self.last_state["generation"]
                if checkpoint_callback:
                    checkpoint_callback(self.last_state)
            raise
        except InterruptedError:
            self.progress.warning("Optimization dispatch cancelled; resume from the last completed generation",
                                  completed_generation=self.last_state["generation"] if self.last_state else 0,
                                  completed_evaluations=len(self.results),
                                  elapsed_seconds=time.perf_counter()-self._run_started)
            raise
        if self.algorithm.pop is not None and len(self.algorithm.pop):
            # Always confirm every survivor, including infeasible provisional candidates.
            self.progress.milestone("Final population confirmation started", candidates=len(self.algorithm.pop),
                               fidelity="full")
            self._evaluate(self.algorithm.pop.get("X"), "full")
            self._confirm(fidelity)
            if self.history:
                front = self._front("full")
                self.history[-1].update({
                    "confirmed_hypervolume": normalized_hypervolume(front, self.n_features),
                    "best_confirmed_gini": max((r["mean_gini"] for r in front), default=None),
                    "confirmed_count": len(front), "unique_evaluations": len(self.results)})
            self.last_state = self._serialize(generation, best_hv, stalled, fidelity, stop_reason)
            if checkpoint_callback:
                checkpoint_callback(self.last_state)
        return self._result(stop_reason)

    def _result(self, stop_reason):
        front = self._front("full")
        self.progress.milestone("Optimization completed", reason=stop_reason,
                           completed_generation=self.last_state["generation"] if self.last_state else 0,
                           confirmed_count=len(front), unique_evaluations=len(self.results),
                           best_confirmed_gini=max((result["mean_gini"] for result in front),default=None),
                           elapsed_seconds=time.perf_counter()-self._run_started)
        return {"confirmed": self._front("full"),
            "provisional": [r for fidelity in {r["fidelity"] for r in self.results.values()} if fidelity != "full" for r in self._front(fidelity)],
            "history": self.history, "state": self.last_state,
            "rfe_trajectory": self.rfe_trajectory, "stop_reason": stop_reason}


# === Source: memetic_fs/selection.py ===
"""Transparent final-selection policies on a common confirmed CV basis."""

import warnings
import numpy as np


def select_candidate(frontier: list[dict], config: SelectionConfig,
                     random_state: int = 42, candidate_id: str | None = None) -> dict:
    """Choose a feasible confirmed candidate; statistical policy is a heuristic.

    Statistical comparison resamples adjacent pairs of validation folds as
    temporal blocks and bounds paired mean Gini loss by the explicit tolerance.
    This supports approximate uncertainty, not a proof of equivalence.
    """
    candidates = [r for r in frontier if r["fidelity"] == "full" and all(g <= 0 for g in r["constraints"])]
    if not candidates:
        raise ValueError("No feasible confirmed Pareto candidates; inspect segment violations or expand the search")
    candidates.sort(key=lambda r: (r["n_features"], -r["mean_gini"], r["key"]))
    best = max(candidates, key=lambda r: r["mean_gini"])
    if config.policy == "manual":
        if candidate_id is None:
            raise ValueError("Manual selection requires candidate_id from pareto_front_")
        for r in candidates:
            if r["key"] == candidate_id:
                return r
        raise ValueError("candidate_id is not a feasible confirmed Pareto candidate")
    if config.policy == "tolerance":
        return next(r for r in candidates if best["mean_gini"] - r["mean_gini"] <= config.gini_tolerance + 1e-12)
    if config.policy == "knee":
        if len(candidates) < 3:
            return select_candidate(candidates, SelectionConfig(gini_tolerance=config.gini_tolerance), random_state)
        xy = np.array([[r["n_features"], r["mean_gini"]] for r in candidates], dtype=float)
        spans = np.ptp(xy, axis=0)
        z = (xy - xy.min(axis=0)) / np.where(spans > 0, spans, 1)
        # Largest improvement over the endpoint chord after normalization.
        distances = z[:, 1] - z[:, 0]
        return candidates[int(np.argmax(distances))]
    if len(best["folds"]) < 6:
        warnings.warn("Statistical selection has fewer than six validation folds; temporal uncertainty estimates are weak and do not establish equivalence", UserWarning, stacklevel=2)
    rng = np.random.default_rng(random_state)
    n = len(best["folds"])
    starts = rng.integers(0, n, size=(config.bootstrap_repeats, (n + 1) // 2))
    indices = np.stack((starts, (starts + 1) % n), axis=-1).reshape(config.bootstrap_repeats, -1)[:, :n]
    for r in candidates:
        if [f["fold"] for f in r["folds"]] != [f["fold"] for f in best["folds"]]:
            raise ValueError("Paired comparison requires identical validation folds")
        losses = np.array([a["gini"] - b["gini"] for a, b in zip(best["folds"], r["folds"])])
        upper_loss = float(np.quantile(losses[indices].mean(axis=1), config.confidence))
        if upper_loss <= config.gini_tolerance:
            return r
    return best


# === Source: memetic_fs/api.py ===
"""Public estimator-style API and structured checkpoint recovery."""


from dataclasses import asdict
from pathlib import Path
import hashlib
import inspect
from importlib.metadata import version
import logging
import shutil
import threading
import time
from typing import Callable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted











log = logging.getLogger(__name__)

def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

class MemeticNSGA2Selector(TransformerMixin, BaseEstimator):
    """Select development predictors using temporal, full-row memetic NSGA-II.

    ``time_cv`` is a DataFrame time column or a callable receiving normalized
    monthly ordinals and returning integer ``(train, valid)`` positions. For a
    callable, ``fit(time_values=...)`` is mandatory. No external OOT input exists.
    ``feature_range`` is inclusive; the upper bound is clipped to predictor count.
    """
    def __init__(self, model_params: dict | None = None, time_cv: str | Callable = "reference_month",
                 feature_range: tuple[int, int] = (10, 200), objectives: tuple[str, str] = ("gini", "n_features"),
                 segment_constraints: list[SegmentConstraint | dict] | None = None,
                 use_rfe_initialization: bool = True, resources: ResourceConfig | dict | str = "auto",
                 random_state: int = 42, checkpoint_dir: str | Path = "./runs/run_001",
                 cv_config: CVConfig | dict | None = None, optimization: OptimizationConfig | dict | None = None,
                 evaluation: EvaluationConfig | dict | None = None, selection: SelectionConfig | dict | None = None,
                 num_boost_round: int = 100, allow_repeated_ids: bool = False,
                 initial_subsets: list | None = None, feature_groups: dict | None = None,
                 interactive_plots: bool = True, verbose: int = 1,
                 output_format: str = "parquet") -> None:
        self.model_params = model_params
        self.time_cv = time_cv
        self.feature_range = feature_range
        self.objectives = objectives
        self.segment_constraints = segment_constraints
        self.use_rfe_initialization = use_rfe_initialization
        self.resources = resources
        self.random_state = random_state
        self.checkpoint_dir = checkpoint_dir
        self.cv_config = cv_config
        self.optimization = optimization
        self.evaluation = evaluation
        self.selection = selection
        self.num_boost_round = num_boost_round
        self.allow_repeated_ids = allow_repeated_ids
        self.initial_subsets = initial_subsets
        self.feature_groups = feature_groups
        self.interactive_plots = interactive_plots
        self.verbose = verbose
        self.output_format = output_format

    def _settings(self):
        self._validate_options()
        self._cancel = threading.Event()
        self.directory_ = Path(self.checkpoint_dir).resolve()
        self.directory_.mkdir(parents=True, exist_ok=True)
        self.progress_ = ProgressReporter(self.directory_, verbose=self.verbose)
        self.log_path_ = self.progress_.path

    def _validate_options(self):
        for name in ("use_rfe_initialization", "allow_repeated_ids", "interactive_plots"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.output_format, str) or self.output_format not in {"csv", "json", "parquet"}:
            raise ValueError("output_format must be 'csv', 'json', or 'parquet'")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        if tuple(self.objectives) != ("gini", "n_features"):
            raise ValueError("Supported objectives are ('gini', 'n_features')")
        if isinstance(self.resources, str) and self.resources != "auto":
            raise ValueError("resources must be 'auto', ResourceConfig, or a mapping")
        self.cv_config_ = from_dict(CVConfig, self.cv_config)
        self.optimization_ = from_dict(OptimizationConfig, self.optimization)
        self.resources_ = from_dict(ResourceConfig, None if self.resources == "auto" else self.resources)
        self.evaluation_ = from_dict(EvaluationConfig, self.evaluation)
        self.selection_ = from_dict(SelectionConfig, self.selection)
        self.constraints_ = [from_dict(SegmentConstraint, c) for c in self.segment_constraints or []]
        if not isinstance(self.num_boost_round, int) or isinstance(self.num_boost_round, bool) or self.num_boost_round < 1:
            raise ValueError("num_boost_round must be a positive integer")
        if len(self.feature_range) != 2 or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in self.feature_range):
            raise ValueError("feature_range must contain two integer bounds")
        if self.feature_range[0] < 1 or self.feature_range[1] < self.feature_range[0]:
            raise ValueError("feature_range must have positive ascending bounds")
        if not isinstance(self.time_cv, str) and not callable(self.time_cv):
            raise ValueError("time_cv must be a column name or callable")
        if not isinstance(self.verbose, int) or isinstance(self.verbose, bool) or self.verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1, or 2")
        strict_params(self.model_params, seed=self.evaluation_.estimator_seed, threads=1, n_rows=1)

    @classmethod
    def from_config(cls, path: str | Path, **overrides) -> MemeticNSGA2Selector:
        """Load selector or experiment YAML, then apply explicit keyword overrides."""
        pass
        settings = load_selector_settings(path)
        allowed = set(inspect.signature(cls).parameters)
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"Unknown selector overrides: {sorted(unknown)}")
        for key, value in overrides.items():
            if key in {"model_params", "cv_config", "optimization", "evaluation", "resources", "selection"} and isinstance(value, dict):
                previous = settings.get(key)
                if hasattr(previous, "__dataclass_fields__"):
                    previous = asdict(previous)
                settings[key] = {**(previous if isinstance(previous, dict) else {}), **value}
            else:
                settings[key] = value
        selector = cls(**settings)
        selector._validate_options()
        return selector

    @classmethod
    def warmup_settings_params(cls, X, y, *, depth: str = "standard", output_path: str | Path,
                               base_config: str | Path | None = None, overwrite: bool = False,
                               id_col: str | None = None, segment_cols: Sequence[str] | None = None,
                               time_values=None, ids=None, feature_names: Sequence[str] | None = None,
                               time_cv: str | Callable | None = None):
        """Inspect development data and save separate suggested selector YAML."""
        pass
        return warmup_settings_params(cls, X, y, depth=depth, output_path=output_path,
                                      base_config=base_config, overwrite=overwrite,
                                      id_col=id_col, segment_cols=segment_cols,
                                      time_values=time_values, ids=ids, feature_names=feature_names,
                                      time_cv=time_cv)

    def fit(self, X: pd.DataFrame | np.ndarray, y: str | Sequence[int] | np.ndarray, *,
            id_col: str | None = None, segment_cols: Sequence[str] | None = None,
            time_values=None, ids=None, feature_names: Sequence[str] | None = None) -> MemeticNSGA2Selector:
        """Validate development data, optimize subsets, and export an experiment."""
        self._settings()
        shape = getattr(X, "shape", ())
        self.progress_.info("Validating development data", phase="data", rows=shape[0] if len(shape) else None,
                            columns=shape[1] if len(shape) > 1 else None)
        if (self.directory_ / "manifest.json").exists():
            raise ValueError("Experiment directory already contains a run; use MemeticNSGA2Selector.resume(directory) or a new directory")
        data = prepare_data(X, y, self.time_cv, id_col, segment_cols, self.allow_repeated_ids,
                            time_values, ids, feature_names)
        self._set_features(data)
        self.progress_.milestone("Development data validated", phase="data", rows=data.n_rows,
                            predictors=data.n_features, categorical_features=len(data.categorical_indices),
                            observed_months=len(np.unique(data.months)))
        self.folds_ = build_folds(data.months, data.y, self.cv_config_, custom=self.time_cv if callable(self.time_cv) else None)
        if self.optimization_.reduced_generations and self.optimization_.reduced_fold_count > len(self.folds_):
            raise ValueError("reduced_fold_count exceeds the available canonical fold count")
        pass
        normalize_subsets(self.initial_subsets, data.feature_names, self.feature_range_)
        self.progress_.info("Temporal folds validated", phase="cv", folds=len(self.folds_),
                            training_rows=[len(f.train) for f in self.folds_],
                            validation_rows=[len(f.valid) for f in self.folds_])
        self.progress_.milestone("Temporal folds validated", folds=len(self.folds_))
        required_bytes = data.X.nbytes + data.y.nbytes + data.months.nbytes
        required_bytes += sum(f.train.nbytes + f.valid.nbytes for f in self.folds_)
        if shutil.disk_usage(self.directory_).free < required_bytes * 1.1 + 64 * 1024**2:
            raise OSError("Insufficient disk space for full shared data, materialized folds and checkpoint reserve")
        save_dataset(data, self.directory_ / "data")
        self.progress_.info("Read-only experiment data persisted", phase="storage", data_bytes=required_bytes)
        data_fingerprint = data.fingerprint
        # Workers reopen the persisted arrays; release the one-time conversion
        # before calibration instead of retaining a second complete matrix.
        del data
        self._build_evaluator()
        config = self._config_record()
        artifacts = [p for p in (self.directory_ / "data").iterdir() if p.is_file()]
        artifacts.extend((self.directory_ / "folds").glob("*.npy"))
        self.manifest_ = {"schema": 1, "config": config, "evaluation_identity": self.evaluator_.identity,
                          "data_fingerprint": data_fingerprint, "versions": self.evaluator_.versions,
                          "training_settings": strict_params(self.model_params, seed=self.evaluation_.estimator_seed,
                              threads=self.evaluator_.training_threads, n_rows=max(len(f.train) for f in self.folds_)),
                          "bin_construction": "bin_construct_sample_cnt equals each complete training fold length",
                          "folds": [{"label": f.label, "train": f"folds/{i}_train.npy", "valid": f"folds/{i}_valid.npy"} for i, f in enumerate(self.folds_)],
                          "files": {str(p.relative_to(self.directory_)): _file_digest(p) for p in artifacts},
                          "excluded_constraints": self.evaluator_.excluded_constraints}
        atomic_json(self.directory_ / "manifest.json", self.manifest_)
        atomic_json(self.directory_ / "config.json", config)
        return self._run()

    def _set_features(self, data):
        self.feature_names_in_ = np.asarray(data.feature_names, dtype=object)
        self.n_features_in_ = data.n_features
        if len(self.feature_range) != 2 or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) for x in self.feature_range):
            raise ValueError("feature_range must contain two integer bounds")
        low, high = self.feature_range
        high = min(high, data.n_features)
        if low < 1 or high < low:
            raise ValueError(f"Invalid feature_range {self.feature_range} for {data.n_features} predictors; lower bound must be >=1 and <= upper")
        self.feature_range_ = (low, high)

    def _build_evaluator(self, expected_identity=None):
        self.evaluator_ = Evaluator(self.directory_ / "data", self.folds_, self.directory_, self.model_params,
                                    self.num_boost_round, self.evaluation_, self.resources_, self.constraints_,
                                    cancelled=self._cancel.is_set, expected_identity=expected_identity,
                                    progress=self.progress_)

    def _config_record(self):
        return {"model_params": self.model_params, "time_cv": self.time_cv if isinstance(self.time_cv, str) else "__persisted_custom_folds__",
                "feature_range": self.feature_range_, "objectives": self.objectives,
                "segment_constraints": [asdict(c) for c in self.constraints_],
                "use_rfe_initialization": self.use_rfe_initialization, "resources": asdict(self.resources_),
                "random_state": self.random_state, "cv_config": asdict(self.cv_config_),
                "optimization": asdict(self.optimization_), "evaluation": asdict(self.evaluation_),
                "selection": asdict(self.selection_), "num_boost_round": self.num_boost_round,
                "allow_repeated_ids": self.allow_repeated_ids, "initial_subsets": self.initial_subsets,
                "feature_groups": self.feature_groups, "interactive_plots": self.interactive_plots,
                "output_format": self.output_format,
                "verbose": self.verbose}

    def _run(self, state=None):
        started = time.monotonic()
        self.progress_.milestone("Starting feature subset search" if state is None else "Resuming feature subset search",
                            phase="optimization", generation=state["generation"] if state else 0,
                            population=self.optimization_.population_size,
                            max_generations=self.optimization_.max_generations,
                            feature_range=self.feature_range_)
        self.engine_ = SearchEngine(self.evaluator_, self.n_features_in_, self.feature_range_, self.optimization_,
                                   random_state=self.random_state, use_rfe_initialization=self.use_rfe_initialization,
                                   initial_subsets=self.initial_subsets, feature_names=list(self.feature_names_in_),
                                   feature_groups=self.feature_groups)
        self.progress_.start_generations(self.optimization_.max_generations,
                                         completed=state["generation"] if state else 0)
        try:
            result = self.engine_.run(checkpoint_callback=self._checkpoint, state=state, cancel=self._cancel.is_set)
        except (KeyboardInterrupt, InterruptedError):
            self.progress_.warning("Search interrupted; completed evaluations and the last checkpoint are reusable", phase="optimization")
            log.warning("Search interrupted; completed evaluations are durable. Resume the experiment to continue.")
            raise
        except Exception as error:
            self.progress_.error("Feature subset search failed; completed evaluations remain cached", phase="optimization", error_type=type(error).__name__, reason=str(error))
            raise
        finally:
            self.progress_.finish_generations()
        self._state = result["state"]
        self.confirmed_ = result["confirmed"]
        self.provisional_ = result["provisional"]
        self.history_ = pd.DataFrame(result["history"])
        self.history_ = self.history_.reindex(sorted(self.history_.columns), axis=1)
        self.stop_reason_ = result["stop_reason"]
        self.rfe_trajectory_ = result.get("rfe_trajectory", [])
        self.pareto_front_ = pd.DataFrame(self.confirmed_)
        self.reference_ = self.evaluator_.reference
        if self.selection_.policy == "manual":
            self.best_features_ = None
            self.selected_candidate_ = None
            selected_id = self.manifest_.get("selected_candidate_id")
            if selected_id is not None and any(r["key"] == selected_id for r in self.confirmed_):
                self._select(selected_id)
            else:
                self.progress_.milestone("Manual selection awaits a confirmed candidate ID", phase="selection",
                                    candidates=len(self.confirmed_))
        else:
            self._select()
        self._export()
        self.save_checkpoint()
        self.progress_.milestone("Feature selection completed", phase="complete", stop_reason=self.stop_reason_,
                            confirmed_pareto_candidates=len(self.confirmed_), selected_features=len(self.best_features_ or []),
                            elapsed_seconds=round(time.monotonic() - started, 3), log=str(self.log_path_))
        self.progress_.close()
        return self

    def _select(self, candidate_id=None):
        self.selected_candidate_ = select_candidate(self.confirmed_, self.selection_, self.random_state, candidate_id)
        self.best_features_ = self.feature_names_in_[np.asarray(self.selected_candidate_["mask"], bool)].tolist()
        self.progress_.milestone("Final candidate selected", phase="selection", policy=self.selection_.policy,
                            features=len(self.best_features_), mean_gini=self.selected_candidate_["mean_gini"],
                            candidate=self.selected_candidate_["key"][:12])

    def select(self, policy: str = "tolerance", *, candidate_id: str | None = None, gini_tolerance: float | None = None) -> list[str]:
        """Change final selection among feasible confirmed Pareto candidates."""
        check_is_fitted(self, "confirmed_")
        settings = asdict(self.selection_)
        settings["policy"] = policy
        if gini_tolerance is not None:
            settings["gini_tolerance"] = gini_tolerance
        self.selection_ = SelectionConfig(**settings)
        self._select(candidate_id)
        self.manifest_["config"]["selection"] = asdict(self.selection_)
        self.manifest_["selected_candidate_id"] = self.selected_candidate_["key"]
        atomic_json(self.directory_ / "manifest.json", self.manifest_)
        atomic_json(self.directory_ / "config.json", self.manifest_["config"])
        self._export()
        self.save_checkpoint()
        self.progress_.close()
        return self.best_features_

    def _checkpoint(self, state):
        self._state = state
        self.save_checkpoint()

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Atomically save portable optimizer state; completed fits live in SQLite."""
        if not hasattr(self, "manifest_"):
            raise ValueError("No initialized experiment is available to checkpoint")
        destination = Path(path) if path is not None else self.directory_ / "checkpoint.json"
        atomic_json(destination, {"schema": 1, "evaluation_identity": self.evaluator_.identity,
                                  "config_hash": stable_hash(self.manifest_["config"]),
                                  "state": getattr(self, "_state", None),
                                  "pending": self.evaluator_.cache.incomplete()})
        self.progress_.info("Checkpoint saved", phase="checkpoint",
                            generation=self._state.get("generation", 0) if getattr(self, "_state", None) else 0)
        return destination

    @classmethod
    def resume(cls, experiment_dir: str | Path, *, verbose: int | None = None) -> MemeticNSGA2Selector:
        """Verify persisted data, versions, settings and resume without source access."""
        directory = Path(experiment_dir).resolve()
        manifest = read_json(directory / "manifest.json")
        if manifest.get("schema") != 1:
            raise ValueError("Unsupported experiment schema")
        if any(version(name) != recorded for name, recorded in manifest["versions"].items()):
            raise ValueError("Checkpoint incompatible with current dependency versions; use the original environment")
        for relative, digest in manifest["files"].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or _file_digest(path) != digest:
                raise ValueError(f"Persisted input changed or invalid path: {relative}")
        settings = dict(manifest["config"])
        if verbose is not None:
            settings["verbose"] = verbose
        selector = cls(**settings, checkpoint_dir=directory)
        selector._settings()
        selector.progress_.milestone("Checkpoint inputs and dependency versions verified", phase="resume", directory=str(directory))
        data = load_dataset(directory / "data")
        selector._set_features(data)
        selector.folds_ = [Fold(np.load(directory / f["train"], mmap_mode="r"),
                               np.load(directory / f["valid"], mmap_mode="r"), f["label"]) for f in manifest["folds"]]
        selector._build_evaluator(expected_identity=manifest["evaluation_identity"])
        if selector.evaluator_.identity != manifest["evaluation_identity"]:
            raise ValueError("Checkpoint incompatible with current evaluator, dependency versions, data or settings")
        selector.manifest_ = manifest
        checkpoint_path = directory / "checkpoint.json"
        state = None
        if checkpoint_path.exists():
            checkpoint = read_json(checkpoint_path)
            if checkpoint.get("schema") != 1 or checkpoint["evaluation_identity"] != selector.evaluator_.identity or checkpoint["config_hash"] != stable_hash(manifest["config"]):
                raise ValueError("Checkpoint configuration or evaluator identity differs from manifest")
            state = checkpoint["state"]
        selector.evaluator_.all_results = {r["key"]: r for r in selector.evaluator_.cache.completed()}
        return selector._run(state)

    def cancel(self) -> None:
        """Request cancellation at the next safe admission/generation boundary."""
        if hasattr(self, "_cancel"):
            self._cancel.set()
            self.progress_.warning("Cancellation requested; waiting for the next safe boundary", phase="cancellation")

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Return selected predictor columns in their original order."""
        check_is_fitted(self, "best_features_")
        if self.best_features_ is None:
            raise ValueError("Use select('manual', candidate_id=...) before transform")
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.best_features_]
        array = np.asanyarray(X)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("Array predictor count differs from fitted data")
        return array[:, self.get_support()]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return selected predictor mask or positions."""
        check_is_fitted(self, "best_features_")
        if self.best_features_ is None:
            raise ValueError("Select a candidate first")
        mask = np.isin(self.feature_names_in_, self.best_features_)
        return np.flatnonzero(mask) if indices else mask

    def fit_final_model(self) -> lgb.Booster:
        """Fit selected predictors on all development rows and persist the Booster."""
        check_is_fitted(self, "best_features_")
        if self.best_features_ is None:
            raise ValueError("Select a candidate first")
        data = self.evaluator_.data
        started = time.monotonic()
        self.progress_.milestone("Fitting final model on all development rows", phase="final_model", rows=len(data.y),
                            features=len(self.best_features_), rounds=self.num_boost_round)
        cols = np.flatnonzero(self.get_support())
        cats = [i for i, col in enumerate(cols) if col in data.categorical_indices]
        plan = self.evaluator_.manager.plan()
        if plan.workers < 1:
            raise MemoryError("Insufficient memory headroom for final full-development fit")
        params = strict_params(self.model_params, seed=self.evaluation_.estimator_seed,
                               threads=plan.threads_per_worker, n_rows=len(data.y))
        matrix = selected_matrix(data.X, np.arange(len(data.y)), cols)
        with progress_activity(self.progress_, "Final development fit still running", phase="final_model",
                               rows=len(data.y), features=len(cols)):
            self.final_model_ = lgb.train(params, lgb.Dataset(matrix, label=data.y, categorical_feature=cats),
                                          num_boost_round=self.num_boost_round)
        # LightGBM uses this mapping to align DataFrame categories on predict
        # and includes it when saving/reloading the native model text file.
        self.final_model_.pandas_categorical = [data.categories[self.feature_names_in_[cols[i]]] for i in cats]
        self.final_model_.save_model(str(self.directory_ / "final_model.txt"))
        atomic_json(self.directory_ / "final_model_metadata.json", {"features": self.best_features_, "categories": data.categories, "parameters": params})
        self.progress_.milestone("Final development model saved", phase="final_model", elapsed_seconds=round(time.monotonic() - started, 3))
        self.progress_.close()
        return self.final_model_

    def close(self) -> None:
        """Flush and close this selector's log files; later operations reopen them."""
        if hasattr(self, "progress_"):
            self.progress_.close()

    def _export(self):
        self.progress_.info("Exporting experiment results", phase="reporting", interactive_plots=self.interactive_plots)
        pass
        export_results(self.directory_, results=list(self.evaluator_.all_results.values()),
                       pareto=self.confirmed_, feature_names=list(self.feature_names_in_),
                       best_features=self.best_features_ or [], history=self.history_.to_dict("records"),
                       reference=self.reference_, rfe_trajectory=self.rfe_trajectory_,
                       feature_groups=self.feature_groups,
                       segment_constraints=self.evaluator_.active_constraints,
                       gini_tolerance=self.selection_.gini_tolerance,
                       selected_candidate_id=self.selected_candidate_["key"] if self.selected_candidate_ else None,
                       interactive_plots=self.interactive_plots,
                       output_format=self.output_format)
        pass
        self.selection_frequency_ = selection_frequency(self.confirmed_, list(self.feature_names_in_), self.feature_groups)
        chosen = self.selected_candidate_ or self.reference_
        self.temporal_performance_ = pd.DataFrame(chosen["folds"])
        self.segment_performance_ = pd.DataFrame(chosen["segments"])
        self.progress_.milestone("Result artifacts saved", phase="reporting", directory=str(self.directory_))


# === Source: memetic_fs/cli.py ===
"""YAML-driven experiment execution; no executable Python is loaded from YAML."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import logging
import pandas as pd
import yaml


def _check(values: dict, allowed: set[str], label: str):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")

def load_experiment(path: str | Path, *, _validate: bool = True) -> tuple[dict, dict]:
    """Read strict YAML and map it onto the public Python constructor."""
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    _check(config, {"data", "model", "cross_validation", "optimization", "evaluation", "resources", "output", "selection", "segment_constraints", "initial_subsets", "feature_groups"}, "root")
    for key in ("data", "output"):
        if key not in config:
            raise ValueError(f"Missing YAML section {key}")
    data = dict(config["data"])
    _check(data, {"path", "target", "time_column", "id_column", "allow_repeated_ids", "segment_columns"}, "data")
    for key in ("path", "target", "time_column"):
        if key not in data:
            raise ValueError(f"Missing data.{key}")
    data["path"] = str((source.parent / data["path"]).resolve())
    model = dict(config.get("model", {}))
    _check(model, {"estimator", "params", "num_boost_round"}, "model")
    if model.get("estimator", "lightgbm") != "lightgbm":
        raise ValueError("model.estimator must be lightgbm")
    cv = dict(config.get("cross_validation", {}))
    if cv.pop("strategy", "expanding") != "expanding":
        raise ValueError("Only expanding auto CV is supported; use Python callable API for custom windows")
    optimization = dict(config.get("optimization", {}))
    if optimization.pop("algorithm", "memetic_nsga2") != "memetic_nsga2":
        raise ValueError("optimization.algorithm must be memetic_nsga2")
    special = {key: optimization.pop(key) for key in ("feature_range", "random_state", "use_rfe_initialization") if key in optimization}
    evaluation = dict(config.get("evaluation", {}))
    if evaluation.pop("metric", "gini") != "gini":
        raise ValueError("evaluation.metric must be gini")
    for name in ("row_sampling", "feature_sampling", "use_sample_weights"):
        if evaluation.pop(name, False) is not False:
            raise ValueError(f"evaluation.{name} must be false")
    reduced = evaluation.pop("allow_reduced_fold_fidelity", True)
    if not isinstance(reduced, bool):
        raise ValueError("allow_reduced_fold_fidelity must be boolean")
    if not reduced:
        optimization["reduced_generations"] = 0
    resources = dict(config.get("resources", {}))
    if resources.pop("mode", "auto") != "auto":
        raise ValueError("resources.mode must be auto; apply limits with max_workers/max_ram_gb/threads_per_worker")
    output = dict(config["output"])
    _check(output, {"experiment_dir", "checkpoints", "interactive_plots", "verbose", "format"}, "output")
    if "experiment_dir" not in output:
        raise ValueError("Missing output.experiment_dir")
    if output.get("checkpoints", True) is not True:
        raise ValueError("Durable checkpoints are required and cannot be disabled")
    kwargs = {"model_params": model.get("params"), "num_boost_round": model.get("num_boost_round", 100),
              "time_cv": data["time_column"], "allow_repeated_ids": data.get("allow_repeated_ids", False),
              "cv_config": cv, "optimization": optimization, "evaluation": evaluation,
              "resources": resources, "selection": config.get("selection"),
              "segment_constraints": config.get("segment_constraints"),
              "initial_subsets": config.get("initial_subsets"), "feature_groups": config.get("feature_groups"),
              "checkpoint_dir": str((source.parent / output["experiment_dir"]).resolve()),
              "interactive_plots": output.get("interactive_plots", True),
              "verbose": output.get("verbose", 1),
              "output_format": output.get("format", "parquet"), **special}
    # Validate nested settings before loading a potentially huge dataset.
    if _validate:
        MemeticNSGA2Selector(**kwargs)._validate_options()
    return data, kwargs

def main(argv: list[str] | None = None) -> int:
    """Run, resume or rebuild an experiment report."""
    parser = argparse.ArgumentParser(prog="memetic-fs")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", required=True)
    commands = [run_parser]
    for command in ("resume", "report"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--experiment-dir", required=True)
        commands.append(command_parser)
    for command_parser in commands:
        verbosity = command_parser.add_mutually_exclusive_group()
        verbosity.add_argument("-v", "--verbose", action="store_true", help="Detailed candidate, fold and cache progress")
        verbosity.add_argument("-q", "--quiet", action="store_true", help="Warnings/errors only; retain detailed files")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            data, kwargs = load_experiment(args.config)
            if args.verbose or args.quiet:
                kwargs["verbose"] = 2 if args.verbose else 0
            path = Path(data["path"])
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
            elif path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
            else:
                raise ValueError("CLI data.path must point to .parquet or .csv; use Python API for arrays")
            selector = MemeticNSGA2Selector(**kwargs).fit(frame, data["target"], id_col=data.get("id_column"), segment_cols=data.get("segment_columns"))
            print(f"Completed: {selector.directory_}; selected {len(selector.best_features_ or [])} predictors; {selector.stop_reason_}")
        elif args.command == "resume":
            selector = MemeticNSGA2Selector.resume(args.experiment_dir,
                verbose=2 if args.verbose else (0 if args.quiet else None))
            print(f"Resumed: {selector.directory_}; {selector.stop_reason_}")
        else:
            pass
            pass
            progress = ProgressReporter(args.experiment_dir, verbose=2 if args.verbose else (0 if args.quiet else 1))
            progress.info("Rebuilding interactive report from saved metrics", phase="reporting")
            rebuild_report(args.experiment_dir)
            progress.info("Interactive report saved", phase="reporting")
            print(f"Report: {Path(args.experiment_dir).resolve() / 'report.html'}")
        return 0
    except (ValueError, FileNotFoundError, MemoryError) as error:
        parser.exit(2, f"memetic-fs: {error}\n")
    except (KeyboardInterrupt, InterruptedError):
        print("Cancelled; completed evaluations are cached. Use memetic-fs resume.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())


# === Source: memetic_fs/config_io.py ===
"""Read selector settings without initializing an experiment."""


import inspect
from pathlib import Path

import yaml


def load_selector_settings(path: str | Path) -> dict:
    """Accept flat selector YAML or the existing CLI experiment YAML."""
    source = Path(path).resolve()
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Selector YAML must be a mapping")
    if "data" in document or "model" in document or "output" in document:
        pass
        _, settings = load_experiment(source, _validate=False)
        return settings
    pass
    allowed = set(inspect.signature(MemeticNSGA2Selector).parameters)
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"Unknown selector YAML keys: {sorted(unknown)}")
    if "checkpoint_dir" in document:
        directory = Path(document["checkpoint_dir"])
        if not directory.is_absolute():
            document["checkpoint_dir"] = str((source.parent / directory).resolve())
    return document


# === Source: memetic_fs/synthetic.py ===
"""Reproducible development-only data for examples, tests and scale benchmarks."""

import numpy as np
import pandas as pd


def make_synthetic_data(n_rows: int = 2400, n_features: int = 20,
                        n_months: int = 12, random_state: int = 42, *,
                        n_signal_features: int = 3, n_redundant_features: int = 1,
                        n_interaction_features: int = 2,
                        n_segmentation_specific_features: int = 1) -> pd.DataFrame:
    """Construct temporal binary risk data with configurable predictor types.

    Signal features affect the target directly (the third has temporal drift).
    Redundant features are correlated copies of signals; when there are no
    signals, they share an informative latent source. Interaction features act
    in pairs, with an unpaired final feature interacting with time. Segmentation
    features act within one customer type. All remaining predictors are noise.
    Metadata consists of ``target``, ``reference_month`` (YYYYMM), unique
    ``applicant_id``, ``customer_type`` and ``age_group``. No independent OOT
    data is generated. At least eight rows per observed month are required.
    The generator does not promise an exactly recoverable feature subset.
    """
    counts = (n_signal_features, n_redundant_features, n_interaction_features,
              n_segmentation_specific_features)
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (n_rows, n_features, n_months, *counts)):
        raise ValueError("Synthetic data dimensions and feature counts must be integers")
    if any(value < 0 for value in counts):
        raise ValueError("Feature-type counts must be nonnegative")
    if n_months < 6 or n_rows < 8*n_months:
        raise ValueError("Require >=6 months and >=8 rows per month")
    if sum(counts) >= n_features:
        raise ValueError("Signal + redundant + interaction + segmentation-specific feature counts must be less than n_features, leaving at least one noise feature")
    rng = np.random.default_rng(random_state)
    months = np.minimum(np.arange(n_rows)*n_months//n_rows, n_months-1)
    matrix = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    n_signal, n_redundant, n_interaction, n_segment = counts
    redundant_start = n_signal
    interaction_start = redundant_start + n_redundant
    segment_start = interaction_start + n_interaction
    names = ([f"signal_{i}" for i in range(n_signal)]
             + [f"redundant_{i}" for i in range(n_redundant)]
             + [f"interaction_{i}" for i in range(n_interaction)]
             + [f"segment_signal_{i}" for i in range(n_segment)]
             + [f"noise_{i}" for i in range(n_features-segment_start-n_segment)])
    for i in range(n_redundant):
        source = matrix[:, i % n_signal] if n_signal else matrix[:, redundant_start]
        if n_signal or i:
            matrix[:, redundant_start + i] = source + rng.normal(scale=.06, size=n_rows).astype(np.float32)
    prospect = np.zeros(n_rows, dtype=bool)
    for month in range(n_months):
        positions = np.flatnonzero(months == month)
        prospect[rng.choice(positions, max(2, len(positions)//8), replace=False)] = True
    trend = 2*months/max(1, n_months-1) - 1
    logits = np.full(n_rows, -.45)
    for i in range(n_signal):
        strength = 2.8 if i == 0 else (1.2 if i == 1 else .7 / (i-1))
        logits += strength * matrix[:, i] * (trend if i == 2 else 1)
    if not n_signal and n_redundant:
        logits += 1.2 * matrix[:, redundant_start]
    for i in range(0, n_interaction, 2):
        left = matrix[:, interaction_start + i]
        right = matrix[:, interaction_start + i + 1] if i + 1 < n_interaction else trend
        logits += 1.2 * left * right
    for i in range(n_segment):
        group = prospect if i % 2 == 0 else ~prospect
        logits += (2.0 / (i + 1)) * matrix[:, segment_start + i] * group
    probability = 1/(1+np.exp(-np.clip(logits, -30, 30)))
    target = rng.binomial(1, probability).astype(np.int8)
    # Avoid a classless demonstration fold on tiny examples while preserving
    # the Bernoulli mechanism; this is synthetic generation, never fit sampling.
    for month in range(n_months):
        positions = np.flatnonzero(months == month)
        while len(np.unique(target[positions])) != 2:
            target[positions] = rng.binomial(1, probability[positions])
    frame = pd.DataFrame(matrix, columns=names)
    periods = pd.period_range("2023-01", periods=n_months, freq="M")
    calendar = np.array([period.year*100+period.month for period in periods])
    frame["target"] = target
    frame["reference_month"] = calendar[months]
    frame["applicant_id"] = np.arange(n_rows, dtype=np.int64)
    frame["customer_type"] = np.where(prospect, "prospect", "client")
    frame["age_group"] = rng.choice(["18-30", "31-45", "46-60", "60+"], n_rows)
    return frame


# === Source: memetic_fs/execution/workers.py ===
"""Worker-side memory measurements, excluding shared file-backed pages."""


import threading




class MemoryMonitor:
    """Sample private memory during an actual full-fold evaluation.

    The peak is sampled rather than an OS-enforced allocation limit. Very short
    allocation spikes can be missed; admission adds a conservative margin.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_private_bytes = 0
        self.baseline_private_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_private_bytes = max(self.peak_private_bytes, private_process_bytes())

    def __enter__(self) -> "MemoryMonitor":
        self.baseline_private_bytes = private_process_bytes()
        self.peak_private_bytes = self.baseline_private_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.peak_private_bytes = max(self.peak_private_bytes, private_process_bytes())
        self._stop.set()
        if self._thread:
            self._thread.join()

    @property
    def peak_increment_bytes(self) -> int:
        """Peak additional private allocation above the pre-evaluation baseline."""
        return max(0, self.peak_private_bytes - self.baseline_private_bytes)


# === Source: memetic_fs/warmup.py ===
"""Optional, reproducible diagnostics and LightGBM pilot recommendations."""


from dataclasses import asdict, dataclass
from pathlib import Path
from copy import deepcopy
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml









@dataclass
class WarmupResult:
    settings_path: Path
    suggestions: dict
    diagnostics: dict
    pilots: dict
    reasons: dict
    changes: dict
    warnings: list[str]


def _plain(value):
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sizes(n: int, depth: str) -> list[int]:
    fractions = (.1, .25, .5, 1.0) if depth == "standard" else (.02, .05, .1, .2, .4, .6, .8, 1.0)
    return sorted({max(1, min(n, round(n * fraction))) for fraction in fractions})


def _candidates(depth: str, min_train: int, baseline: dict, rounds: int) -> list[dict]:
    leaf = max(5, min(300, round(min_train / 500), max(5, min_train // 4)))
    extras = {key: value for key, value in (baseline or {}).items()
              if key not in {"learning_rate", "num_leaves", "min_data_in_leaf"}}
    if depth == "standard":
        tuples = [(.1, 100, 31, leaf), (.05, 200, 31, leaf),
                  (.05, 400, 31, 2 * leaf), (.1, 200, 15, leaf),
                  (.05, 400, 15, leaf), (.05, 400, 63, 2 * leaf)]
    else:
        tuples = [(rate, {0.03: 600, 0.05: 400, 0.1: 200}[rate], leaves, size)
                  for rate in (.03, .05, .1) for leaves in (15, 31, 63)
                  for size in (leaf, 2 * leaf)]
    candidates = [{"params": {**extras, "learning_rate": rate, "num_leaves": leaves,
                               "min_data_in_leaf": size}, "num_boost_round": nrounds}
                  for rate, nrounds, leaves, size in tuples]
    original = {"params": dict(baseline or {}), "num_boost_round": rounds}
    if original not in candidates:
        candidates.insert(0, original)
    return candidates


def _fit(data, fold, columns, candidate, threads, seed):
    train, valid = fold.train, fold.valid
    cats = [i for i, col in enumerate(columns) if col in data.categorical_indices]
    params = strict_params(candidate["params"], seed=seed, threads=threads, n_rows=len(train))
    started = time.perf_counter()
    with MemoryMonitor() as monitor, threadpool_limits(limits=1):
        matrix = selected_matrix(data.X, train, columns)
        model = lgb.train(params, lgb.Dataset(matrix, label=data.y[train],
                          categorical_feature=cats), num_boost_round=candidate["num_boost_round"])
        predictions = model.predict(selected_matrix(data.X, valid, columns), num_threads=threads)
    result = {"fold": fold.label, "gini": score(data.y[valid], predictions)["gini"],
              "elapsed_seconds": time.perf_counter() - started,
              "peak_increment_bytes": monitor.peak_increment_bytes,
              "training_matrix_bytes": matrix.nbytes}
    return result, model.feature_importance(importance_type="gain")


def _diagnostics(data, folds, snapshot):
    months = []
    for month in np.unique(data.months):
        mask = data.months == month
        label = str(pd.Period(ordinal=int(month), freq="M"))
        months.append({"month": label, "rows": int(mask.sum()),
                       "events": int(data.y[mask].sum()),
                       "nonevents": int(mask.sum() - data.y[mask].sum())})
    missing = {}
    for i, name in enumerate(data.feature_names):
        column = data.X[:, i]
        missing[name] = int((column == -1).sum() if i in data.categorical_indices
                            else pd.isna(column).sum())
    return {"rows": data.n_rows, "predictors": data.n_features,
            "data_fingerprint": data.fingerprint,
            "events": int(data.y.sum()), "nonevents": int(len(data.y) - data.y.sum()),
            "categorical_predictors": len(data.categorical_indices),
            "missing_by_feature": missing,
            "months": months,
            "folds": [{"label": fold.label, "train_rows": len(fold.train),
                       "validation_rows": len(fold.valid),
                       "train_events": int(data.y[fold.train].sum()),
                       "validation_events": int(data.y[fold.valid].sum())} for fold in folds],
            "resources": asdict(snapshot)}


def warmup_settings_params(cls, X, y, *, depth, output_path, base_config, overwrite,
                           id_col, segment_cols, time_values, ids, feature_names, time_cv):
    if depth not in {"quick", "standard", "deep"}:
        raise ValueError("depth must be 'quick', 'standard', or 'deep'")
    destination = Path(output_path).resolve()
    if base_config is not None and destination == Path(base_config).resolve():
        raise ValueError("output_path must differ from base_config so the source YAML stays unchanged")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Warm-up YAML already exists: {destination}; use overwrite=True to replace it")
    baseline = cls.from_config(base_config) if base_config is not None else cls()
    if time_cv is not None:
        baseline.time_cv = time_cv
    baseline._validate_options()
    if not isinstance(baseline.time_cv, str):
        raise ValueError("YAML warm-up requires a named time_cv column; callable CV cannot be saved in YAML")
    settings = dict(baseline.get_params(deep=False))
    for key in ("cv_config", "optimization", "evaluation", "resources", "selection"):
        settings[key] = asdict(getattr(baseline, key + "_"))
    if settings["segment_constraints"]:
        settings["segment_constraints"] = [asdict(c) for c in baseline.constraints_]
    original_settings = deepcopy(settings)
    data = prepare_data(X, y, baseline.time_cv, id_col, segment_cols,
                        baseline.allow_repeated_ids, time_values, ids, feature_names)
    for constraint in baseline.constraints_:
        if constraint.column not in data.segments:
            raise ValueError(f"Constraint column {constraint.column!r} must be in segment_cols")
    normalize_subsets(baseline.initial_subsets, data.feature_names, (1, data.n_features))
    folds = build_folds(data.months, data.y, baseline.cv_config_,
                        custom=baseline.time_cv if callable(baseline.time_cv) else None)
    selected_folds = [] if depth == "quick" else (folds[-3:] if depth == "standard" else folds)
    sizes = [] if depth == "quick" else _sizes(data.n_features, depth)
    candidates = [] if depth == "quick" else _candidates(depth, min(len(f.train) for f in selected_folds),
                                                           baseline.model_params, baseline.num_boost_round)
    fit_count = len(candidates) * len(selected_folds) + (len(sizes) - 1) * len(selected_folds)
    notice = (f"Warm-up depth={depth}: {fit_count} planned LightGBM fits on {data.n_rows} rows and "
              f"{data.n_features} predictors. Duration depends on data size and the selected depth; "
              "there is no time or fit-count cutoff.")
    warnings.warn(notice, UserWarning, stacklevel=3)
    messages = [notice]
    snapshot_estimate = max(64 * 1024**2, 3 * max(len(f.train) for f in folds) * data.n_features * data.X.dtype.itemsize)
    manager = ResourceManager(baseline.resources_, snapshot_estimate)
    plan = manager.plan()
    if plan.workers < 1:
        raise MemoryError("Insufficient memory headroom for warm-up pilot or later full-feature fit")
    diagnostics = _diagnostics(data, folds, plan.snapshot)
    diagnostics["planned_fits"] = fit_count
    diagnostics["pilot_folds"] = [f.label for f in selected_folds]
    messages.append("Worker and thread suggestions reflect this resource snapshot and should be reviewed on another machine.")
    pilots = {"model_candidates": [], "subset_scores": [], "subset_score_gaps": []}
    reasons = {"model_params": "Heuristic defaults without model fits" if depth == "quick" else "Chronological full-row pilot",
               "num_boost_round": "Fixed-round LightGBM setting from heuristic" if depth == "quick" else "Fixed-round setting from chronological pilot",
               "resources": "CPU and memory snapshot; caps may become stale if the environment changes",
               "selection": "Conservative 0.001 absolute Gini tolerance; review final Pareto score gaps",
               "feature_range": "Explore all valid subset sizes rather than forcing a minimum signal count"}
    settings["feature_range"] = (1, data.n_features)
    if depth == "quick":
        n_train = min(len(f.train) for f in folds)
        settings["model_params"] = {**(baseline.model_params or {}), "learning_rate": .05,
                                    "num_leaves": 31, "min_data_in_leaf": max(5, min(300, round(n_train / 500)))}
        settings["num_boost_round"] = 200
        messages.append("Quick model suggestions are heuristics and were not validated by pilot fits.")
    else:
        full_cols = np.arange(data.n_features)
        full_records = []
        for candidate in candidates:
            fold_records, importances = [], []
            for fold in selected_folds:
                record, importance = _fit(data, fold, full_cols, candidate, plan.threads_per_worker,
                                          baseline.evaluation_.estimator_seed)
                fold_records.append(record)
                importances.append(importance.tolist())
            summary = {"params": candidate["params"], "num_boost_round": candidate["num_boost_round"],
                       "folds": fold_records, "mean_gini": float(np.mean([r["gini"] for r in fold_records])),
                       "elapsed_seconds": float(sum(r["elapsed_seconds"] for r in fold_records))}
            pilots["model_candidates"].append(summary)
            full_records.append((summary, importances))
        best_mean = max(item[0]["mean_gini"] for item in full_records)
        close = [item for item in full_records if item[0]["mean_gini"] >= best_mean - .001]
        winner, rankings = min(close, key=lambda item: (
            item[0]["params"].get("num_leaves", 31), item[0]["num_boost_round"],
            item[0]["elapsed_seconds"]))
        settings["model_params"] = winner["params"]
        settings["num_boost_round"] = winner["num_boost_round"]
        pilots["selected_model"] = {"params": winner["params"],
                                    "num_boost_round": winner["num_boost_round"]}
        for size in sizes:
            records = []
            for fold, gains, full_record in zip(selected_folds, rankings, winner["folds"]):
                if size == data.n_features:
                    record = full_record
                else:
                    # Every ranking comes from this fold's training-only full model.
                    columns = np.sort(np.argsort(-np.asarray(gains), kind="stable")[:size])
                    record, _ = _fit(data, fold, columns, pilots["selected_model"], plan.threads_per_worker,
                                     baseline.evaluation_.estimator_seed)
                records.append(record)
            pilots["subset_scores"].append({"n_features": size,
                "mean_gini": float(np.mean([r["gini"] for r in records])), "folds": records})
        for smaller, larger in zip(pilots["subset_scores"], pilots["subset_scores"][1:]):
            pilots["subset_score_gaps"].append({"from": smaller["n_features"], "to": larger["n_features"],
                "mean_gini_gain": larger["mean_gini"] - smaller["mean_gini"]})
        observed_peak = max(r["peak_increment_bytes"] + r["training_matrix_bytes"]
                            for item in pilots["model_candidates"] for r in item["folds"])
        manager.calibrate(observed_peak)
        plan = manager.plan()
        if plan.workers < 1:
            raise MemoryError("Pilot memory measurement leaves insufficient headroom for a full-feature fit")
        if pilots["subset_scores"][0]["mean_gini"] >= max(r["mean_gini"] for r in pilots["subset_scores"]) - .001:
            messages.append("Pilot scores are near the smallest probed subset; review the final frontier before accepting a minimum-boundary selection.")
    resources = settings["resources"]
    resources["max_workers"] = plan.workers
    resources["threads_per_worker"] = plan.threads_per_worker
    reasons["resources"] += "; measured pilot memory included" if depth != "quick" else "; conservative matrix estimate used"
    selection = settings["selection"]
    if selection["policy"] in {"tolerance", "statistical"}:
        selection["gini_tolerance"] = min(float(selection["gini_tolerance"]), .001)
    if depth in {"standard", "deep"}:
        settings["optimization"]["rfe_seed_sizes"] = sizes
        reasons["optimization"] = "RFE checkpoints span the pilot subset sizes"
    if depth == "deep":
        optimization = settings["optimization"]
        optimization["population_size"] = min(80, max(20, 2 * data.n_features))
        optimization["max_generations"] = min(100, max(30, 4 * data.n_features))
        optimization["min_generations"] = min(10, optimization["max_generations"])
        optimization["patience"] = 12
        optimization["local_search_budget"] = min(40, max(8, data.n_features // 2))
        optimization["reduced_fold_count"] = min(2, len(folds))
        reasons["optimization"] = "Complete search configuration from predictor count, fold count, and pilot cardinalities; unchanged fields retain validated defaults"
    plain = _plain(settings)
    # A selector YAML cannot contain an executable callable or a non-YAML object.
    try:
        serialized = yaml.safe_dump(plain, sort_keys=False, allow_unicode=True)
    except yaml.YAMLError as error:
        raise ValueError("Selector settings cannot be represented safely in YAML") from error
    cls(**plain)._validate_options()
    previous = _plain(original_settings)
    changes = {key: {"previous": previous.get(key), "proposed": value,
                     "reason": reasons.get(key, "Normalized and validated for the suggested selector YAML")}
               for key, value in plain.items() if previous.get(key) != value}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    else:
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return WarmupResult(destination, plain, diagnostics, pilots, reasons, changes, messages)


# === Source: memetic_fs/reporting/exports.py ===
"""Standardized metrics exports; no estimator objects or predictions are persisted."""


from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _clean(value: Any) -> Any:
    if is_dataclass(value):
        return _clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if k not in {"predictions", "oof_predictions", "models"}}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp, Path)):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _unique(results: Sequence[dict]) -> list[dict]:
    return list({tuple(result["mask"]): result for result in results}.values())


def _families(feature_names: Sequence[str], feature_groups: dict | None) -> dict[str, str]:
    if not feature_groups:
        return {}
    if all(name in feature_names for name in feature_groups):
        return {name: str(family) for name, family in feature_groups.items()}
    result = {}
    for family, features in feature_groups.items():
        for feature in features:
            name = feature_names[feature] if isinstance(feature, int) else feature
            if name not in feature_names:
                raise ValueError(f"Unknown grouped feature {name!r}")
            result[name] = str(family)
    return result


def selection_frequency(
    results: Sequence[dict], feature_names: Sequence[str], feature_groups: dict | None = None,
) -> pd.DataFrame:
    """Count inclusion across distinct supplied subsets, without score weighting.

    Supply a confirmed frontier or a chosen high-performing solution family.
    Frequencies describe the searched solutions and are not feature significance.
    """
    results = _unique(results)
    masks = np.asarray([result["mask"] for result in results], dtype=bool)
    if results and (masks.ndim != 2 or masks.shape[1] != len(feature_names)):
        raise ValueError("Frequency masks must match feature_names")
    counts = masks.sum(axis=0) if len(results) else np.zeros(len(feature_names), dtype=int)
    table = pd.DataFrame({"feature": list(feature_names), "count": counts,
                          "frequency": counts / len(results) if results else np.zeros(len(feature_names)),
                          "n_solutions": len(results)})
    families = _families(feature_names, feature_groups)
    if families:
        table["family"] = table.feature.map(families).fillna("ungrouped")
    return table.sort_values(["frequency", "feature"], ascending=[False, True]).reset_index(drop=True)


def aggregate_run_frequencies(directories: Sequence[str | Path]) -> pd.DataFrame:
    """Aggregate confirmed Pareto frequencies across independent saved runs.

    ``run_frequency`` is the fraction of runs in which a feature appears in at
    least one Pareto subset. ``frequency`` pools all subsets without collapsing
    identical masks from different runs. Runs must use the same predictor names.
    """
    if not directories:
        raise ValueError("At least one experiment directory is required")
    names = None
    counts, run_counts, total = None, None, 0
    for directory in directories:
        payload = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
        if names is None:
            names = payload["feature_names"]
            counts = np.zeros(len(names), dtype=int)
            run_counts = np.zeros(len(names), dtype=int)
        elif names != payload["feature_names"]:
            raise ValueError("Independent runs must use identical predictor names and order")
        front = _unique(payload["pareto"])
        if front:
            masks = np.array([r["mask"] for r in front], dtype=bool)
            counts += masks.sum(axis=0)
            run_counts += masks.any(axis=0)
            total += len(front)
    return pd.DataFrame({"feature": names, "count": counts,
                         "frequency": counts / total if total else np.zeros(len(names)),
                         "run_count": run_counts, "run_frequency": run_counts / len(directories),
                         "n_solutions": total, "n_runs": len(directories)}).sort_values(
                             ["frequency", "feature"], ascending=[False, True]).reset_index(drop=True)


def _table(directory: Path, name: str, rows: list | pd.DataFrame, output_format: str) -> pd.DataFrame:
    empty_columns = {
        "evaluations": ["key", "fidelity", "n_features", "mean_gini", "std_gini", "min_gini", "pooled_gini", "pareto", "feasible", "max_constraint_violation"],
        "confirmed_masks": ["key"],
        "fold_performance": ["key", "fidelity", "fold", "auc", "gini", "n", "events", "nonevents"],
        "segment_performance": ["key", "fidelity", "column", "value", "fold", "auc", "gini", "n", "events", "nonevents"],
        "history": ["generation", "fidelity", "confirmed_hypervolume", "best_confirmed_gini", "unique_evaluations", "converged"],
        "rfe_trajectory": ["key", "n_features", "mean_gini", "std_gini"],
        "segment_constraints": ["key", "fidelity", "column", "value", "fold", "candidate_gini", "reference_gini", "minimum_gini", "violation", "violated", "n", "events", "nonevents"],
    }
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows if rows else None, columns=None if rows else empty_columns.get(name))
    path = directory / f"{name}.{output_format}"
    if output_format == "csv":
        frame.to_csv(path, index=False)
    elif output_format == "json":
        _json(path, frame.to_dict(orient="records"))
    else:
        frame.to_parquet(path, index=False)
    for other in {"csv", "json", "parquet"} - {output_format}:
        (directory / f"{name}.{other}").unlink(missing_ok=True)
    return frame


def export_results(
    directory: str | Path,
    results: Sequence[dict],
    feature_names: Sequence[str],
    best_features: Sequence[str],
    history: Sequence[dict],
    reference: dict,
    rfe_trajectory: Sequence[dict] | None = None,
    feature_groups: dict | None = None,
    pareto: Sequence[dict] | None = None,
    segment_constraints: Sequence[dict] | None = None,
    interactive_plots: bool = True,
    gini_tolerance: float = 0.005,
    selected_candidate_id: str | None = None,
    output_format: str = "parquet",
) -> dict:
    """Export reproducible metrics tables and an interactive offline HTML report.

    All metrics are development validation metrics. The caller supplies the
    canonical confirmed frontier; otherwise pymoo constructs one from full-CV
    feasible results. The full-feature reference is shown separately. The
    function returns exported paths and the Pareto selection-frequency table.
    """
    pass
    pass

    if not isinstance(gini_tolerance, (int, float)) or not math.isfinite(gini_tolerance) or gini_tolerance < 0:
        raise ValueError("Reporting gini_tolerance must be finite and nonnegative on the Gini coefficient scale")
    if not isinstance(output_format, str) or output_format not in {"csv", "json", "parquet"}:
        raise ValueError("output_format must be 'csv', 'json', or 'parquet'")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = _clean(list(results))
    front = _clean(list(pareto)) if pareto is not None else pareto_front(results)
    if any(r.get("fidelity") != "full" for r in front):
        raise ValueError("Exported confirmed Pareto frontier must contain full-fidelity results only")
    if any(any(c > 0 for c in r.get("constraints", [])) or r.get("mean_gini") is None for r in front):
        raise ValueError("Exported confirmed Pareto frontier must contain feasible, defined metrics only")
    payload = {"schema": 1, "feature_names": list(feature_names), "best_features": list(best_features),
               "results": results, "pareto": front, "history": _clean(list(history)),
               "reference": _clean(reference), "rfe_trajectory": _clean(list(rfe_trajectory or [])),
               "feature_groups": _clean(feature_groups), "segment_constraints": _clean(list(segment_constraints or [])),
               "gini_tolerance": gini_tolerance, "selected_candidate_id": selected_candidate_id,
               "output_format": output_format}
    _json(directory / "results.json", payload)
    _json(directory / "selected_features.json", list(best_features))
    _json(directory / "pareto.json", [{**r, "features": [f for f, flag in zip(feature_names, r["mask"]) if flag]} for r in front])
    _json(directory / "feature_masks.json", [{"key": r["key"], "fidelity": r["fidelity"], "mask": r["mask"]} for r in results])
    confirmed = _unique([r for r in results if r["fidelity"] == "full"])
    keys = {r["key"] for r in front}
    summary = [{k: r.get(k) for k in ["key", "fidelity", "n_features", "mean_gini", "std_gini", "min_gini", "pooled_gini"]} | {
                "pareto": r["key"] in keys, "feasible": all(c <= 0 for c in r.get("constraints", [])),
                "max_constraint_violation": max([0.] + r.get("constraints", []))} for r in results]
    _table(directory, "evaluations", summary, output_format)
    _table(directory, "confirmed_masks", [{"key": r["key"], **{f"mask::{f}": flag for f, flag in zip(feature_names, r["mask"])}} for r in confirmed], output_format)
    _table(directory, "fold_performance", [{"key": r["key"], "fidelity": r["fidelity"], **f} for r in results for f in r["folds"]], output_format)
    _table(directory, "segment_performance", [{"key": r["key"], "fidelity": r["fidelity"], **s} for r in results for s in r["segments"]], output_format)
    _table(directory, "segment_constraints", segment_constraint_comparisons(payload), output_format)
    frequency = selection_frequency(front, feature_names, feature_groups)
    _table(directory, "selection_frequency", frequency, output_format)
    high = [r for r in front if r["mean_gini"] >= max((v["mean_gini"] for v in front), default=0) - gini_tolerance]
    _table(directory, "high_performing_frequency", selection_frequency(high, feature_names, feature_groups), output_format)
    if feature_groups:
        family_map = _families(feature_names, feature_groups)
        families = sorted(set(family_map.values()))
        rows = []
        for family in families:
            indices = [i for i, name in enumerate(feature_names) if family_map.get(name) == family]
            count = sum(any(r["mask"][i] for i in indices) for r in front)
            rows.append({"family": family, "count": count, "frequency": count / len(front) if front else 0., "n_solutions": len(front)})
        _table(directory, "family_frequency", rows, output_format)
    _table(directory, "history", list(history), output_format)
    _table(directory, "rfe_trajectory", [{k: r.get(k) for k in ["key", "n_features", "mean_gini", "std_gini"]} for r in (rfe_trajectory or [])], output_format)
    if interactive_plots:
        write_report(directory / "report.html", payload)
    return {"directory": directory.resolve(), "report": (directory / "report.html").resolve(),
            "frequency": frequency, "results": (directory / "results.json").resolve()}


def rebuild_report(directory: str | Path) -> Path:
    """Rebuild offline plots from saved metrics without loading data or models."""
    pass
    directory = Path(directory)
    payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise ValueError("Unsupported results export schema")
    path = directory / "report.html"
    write_report(path, payload)
    return path.resolve()


# === Source: memetic_fs/reporting/visualizations.py ===
"""Plotly diagnostics built exclusively from standardized compact metrics."""


from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


def segment_constraint_comparisons(payload: dict) -> list[dict]:
    """Compute explicit per-fold segment floors and violations for saved results.

    Reference and candidate results share canonical folds; provisional results
    compare only their selected folds. Undefined descriptive metrics remain
    undefined and are never labelled feasible by this diagnostic helper.
    """
    reference = payload["reference"]
    rows = []
    for result in payload.get("results", []):
        for constraint in payload.get("segment_constraints", []):
            for metric in result.get("segments", []):
                if metric["column"] != constraint["column"] or metric["value"] != constraint["value"]:
                    continue
                baseline = next((r for r in reference.get("segments", []) if r["column"] == metric["column"] and
                                 r["value"] == metric["value"] and r["fold"] == metric["fold"]), None)
                reference_gini = baseline["gini"] if baseline else None
                floor = reference_gini - constraint["max_gini_drop"] if reference_gini is not None else None
                violation = floor - metric["gini"] if floor is not None and metric["gini"] is not None else None
                rows.append({"key": result["key"], "fidelity": result["fidelity"], "column": metric["column"],
                             "value": metric["value"], "fold": metric["fold"], "candidate_gini": metric["gini"],
                             "reference_gini": reference_gini, "minimum_gini": floor,
                             "max_gini_drop": constraint["max_gini_drop"], "violation": violation,
                             "violated": violation > 0 if violation is not None else None,
                             "n": metric["n"], "events": metric["events"], "nonevents": metric["nonevents"]})
    return rows


def _base(figure: go.Figure, title: str, x: str = "", y: str = "") -> go.Figure:
    figure.update_layout(title=title, xaxis_title=x, yaxis_title=y, template="plotly_white",
                         font={"family": "Segoe UI, sans-serif", "size": 13, "color": "#223943"},
                         colorway=["#006F83", "#395DA8", "#9A6A28", "#755CA4", "#477B68"],
                         paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=490,
                         margin={"l": 70, "r": 30, "t": 75, "b": 70})
    figure.update_xaxes(gridcolor="#E3EAED", linecolor="#B7C8CE", zerolinecolor="#B7C8CE")
    figure.update_yaxes(gridcolor="#E3EAED", linecolor="#B7C8CE", zerolinecolor="#B7C8CE")
    return figure


def build_figures(payload: dict, max_cofeatures: int = 30) -> dict[str, go.Figure]:
    """Create aligned full/provisional charts and a top-feature co-selection map."""
    pass

    results, front = payload["results"], payload["pareto"]
    reference, names = payload["reference"], payload["feature_names"]
    confirmed = list({tuple(r["mask"]): r for r in results if r["fidelity"] == "full" and r["key"] != reference["key"]}.values())
    provisional = [r for r in results if r["fidelity"] != "full"]
    frontier = go.Figure()
    for label, values, marker in [("Confirmed evaluated subsets", confirmed, {"color": "#9CA3AF", "size": 7}),
                                  ("Provisional (reduced folds)", provisional, {"color": "#E3A03A", "symbol": "x", "size": 7}),
                                  ("Confirmed feasible Pareto subsets", front, {"color": "#007C91", "size": 11})]:
        frontier.add_trace(go.Scatter(x=[r["n_features"] for r in values], y=[r["mean_gini"] for r in values],
            mode="markers+lines" if "Pareto" in label else "markers", name=label, marker=marker,
            customdata=[[r["key"], r.get("std_gini"), r["fidelity"], max([0.] + r.get("constraints", [])),
                         r.get("min_gini"), len(r.get("folds", [])), sum(f.get("n", 0) for f in r.get("folds", []))] for r in values],
            hovertemplate="%{customdata[0]}<br>Features: %{x}<br>Mean Gini: %{y:.5f}<br>Fold SD: %{customdata[1]:.5f}<br>Minimum fold Gini: %{customdata[4]:.5f}<br>Validation folds: %{customdata[5]}<br>Validation observations: %{customdata[6]}<br>Fidelity: %{customdata[2]}<br>Constraint violation: %{customdata[3]:.5f}<extra>%{fullData.name}</extra>"))
    frontier.add_trace(go.Scatter(x=[reference["n_features"]], y=[reference["mean_gini"]], name="Full-feature reference",
                                  mode="markers", marker={"symbol": "star", "size": 16, "color": "#7B3294"}))
    figures = {"frontier": _base(frontier, "Observed development Pareto frontier", "Selected features", "Mean validation-fold Gini")}

    best_by_count = {}
    for result in confirmed:
        k = result["n_features"]
        best_by_count[k] = max(result["mean_gini"], best_by_count.get(k, -1.))
    counts = sorted(best_by_count)
    count_chart = go.Figure(go.Scatter(x=counts, y=[best_by_count[k] for k in counts], mode="lines+markers", name="Best observed full-CV Gini"))
    rfe = sorted(payload.get("rfe_trajectory", []), key=lambda r: r["n_features"])
    if rfe:
        count_chart.add_trace(go.Scatter(x=[r["n_features"] for r in rfe], y=[r["mean_gini"] for r in rfe], mode="lines+markers", name="RFE initialization"))
    figures["cardinality"] = _base(count_chart, "Best discovered Gini by evaluated cardinality (search evidence)", "Selected features", "Mean validation-fold Gini")

    tolerance = payload.get("gini_tolerance", 0.005)
    high = [r for r in front if r["mean_gini"] >= max((v["mean_gini"] for v in front), default=0) - tolerance]
    frequency = selection_frequency(high, names, payload.get("feature_groups")).head(40).iloc[::-1]
    figures["frequency"] = _base(go.Figure(go.Bar(x=frequency.frequency, y=frequency.feature, orientation="h", name="Inclusion frequency")),
                                f"Feature frequency within {tolerance:g} Gini of the best Pareto subset", "Fraction of distinct subsets", "")
    figures["frequency"].update_layout(height=max(490, len(frequency) * 21))
    ranked = selection_frequency(front, names).head(max_cofeatures).feature.tolist()
    indices = [names.index(name) for name in ranked]
    masks = np.asarray([r["mask"] for r in front], dtype=float)
    co = masks[:, indices].T @ masks[:, indices] / len(front) if len(front) else np.zeros((len(indices), len(indices)))
    figures["coselection"] = _base(go.Figure(go.Heatmap(x=ranked, y=ranked, z=co, zmin=0, zmax=1,
            colorscale="Blues", colorbar={"title": "Fraction"}, hovertemplate="%{x} + %{y}<br>Co-selection: %{z:.3f}<extra></extra>")),
            f"Co-selection across confirmed Pareto subsets (top {max_cofeatures})")
    figures["coselection"].update_layout(height=740, xaxis_tickangle=-50, margin={"b": 160, "l": 160})

    selected_mask = [name in payload["best_features"] for name in names]
    selected_id = payload.get("selected_candidate_id")
    selected = next((r for r in confirmed + front if r["key"] == selected_id), None) if selected_id else next((r for r in confirmed + front if r["mask"] == selected_mask), None)
    best = max(front, key=lambda r: r["mean_gini"], default=None)
    designated = [("Selected model", selected), ("Best confirmed Gini", best), ("Full-feature reference", reference)]
    for result in front:
        if len(designated) >= 7:
            break
        if result["key"] not in {r["key"] for _, r in designated if r}:
            designated.append((f"Pareto: {result['n_features']} features", result))
    temporal = go.Figure()
    for label, result in designated:
        if result:
            temporal.add_trace(go.Scatter(x=[f["fold"] for f in result["folds"]], y=[f["gini"] for f in result["folds"]],
                mode="lines+markers", name=label,
                hovertemplate="%{x}<br>Validation Gini: %{y:.5f}<extra>%{fullData.name}</extra>"))
    figures["temporal"] = _base(temporal, "Temporal development-validation stability", "Validation interval", "Gini")

    segment = go.Figure()
    constraints = payload.get("segment_constraints", [])
    for label, result in designated:
        if not result:
            continue
        groups = {}
        for value in result.get("segments", []):
            name = f"{value['column']}={value['value']}"
            if value["gini"] is not None:
                groups.setdefault(name, []).append(value["gini"])
        segment.add_trace(go.Bar(x=list(groups), y=[np.mean(values) for values in groups.values()], name=label))
    if constraints:
        reference_groups = {}
        for value in reference.get("segments", []):
            if value["gini"] is not None:
                reference_groups.setdefault(f"{value['column']}={value['value']}", []).append(value["gini"])
        for constraint in constraints:
            name = f"{constraint['column']}={constraint['value']}"
            if name in reference_groups:
                threshold = np.mean(reference_groups[name]) - constraint["max_gini_drop"]
                segment.add_trace(go.Scatter(x=[name], y=[threshold], mode="markers", name=f"Mean constraint floor: {name}",
                                            marker={"symbol": "line-ew", "size": 25, "color": "#A63232"}))
    segment.update_layout(barmode="group")
    if not any(r and r.get("segments") for _, r in designated):
        segment.add_annotation(text="No business-segment metadata was configured.", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False)
    figures["segments"] = _base(segment, "Segment validation Gini (equal fold means; constraints apply per fold)", "Segment", "Mean defined fold Gini")

    history = payload.get("history", [])
    evolution = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                              subplot_titles=["Confirmed frontier progress", "Evaluation and archive counts", "Population cardinality"])
    generations = [h["generation"] for h in history]
    for fidelity in sorted({h["fidelity"] for h in history}):
        rows = [h for h in history if h["fidelity"] == fidelity]
        evolution.add_trace(go.Scatter(x=[h["generation"] for h in rows], y=[h.get("best_confirmed_gini") for h in rows],
            name=f"Confirmed Gini; exploration={fidelity}", mode="lines+markers"), row=1, col=1)
    evolution.add_trace(go.Scatter(x=generations, y=[h.get("confirmed_hypervolume") for h in history], name="Confirmed normalized hypervolume"), row=1, col=1)
    for key, label in [("unique_evaluations", "Unique evaluation configurations"), ("confirmed_count", "Confirmed Pareto count"), ("provisional_count", "Provisional Pareto count")]:
        evolution.add_trace(go.Scatter(x=generations, y=[h.get(key) for h in history], name=label), row=2, col=1)
    for key in ["feature_count_min", "feature_count_mean", "feature_count_max"]:
        evolution.add_trace(go.Scatter(x=generations, y=[h.get(key) for h in history], name=key.replace("_", " ")), row=3, col=1)
    _base(evolution, "Evolution history (convergence uses confirmed full-CV frontier)")
    evolution.update_layout(height=920)
    evolution.update_xaxes(title_text="Generation", row=3, col=1)
    figures["history"] = evolution
    return figures


_REPORT_CSS = """
:root{color-scheme:light;--ink:#17313d;--muted:#516873;--paper:#eef3f4;--white:#fff;
  --line:#d4e0e4;--signal:#006f83;--deep:#102c39;--alert:#9b3e2c}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);
  color:var(--ink);font:16px/1.55 'Segoe UI',Arial,sans-serif}
a{color:var(--signal)}a:focus-visible{outline:3px solid #e4ad4b;outline-offset:3px}
.skip-link{position:absolute;top:-4rem;left:1rem;z-index:10;background:#fff;padding:.5rem 1rem}
.skip-link:focus{top:1rem}.masthead{background:var(--deep);color:#f4f8f9;
  border-bottom:6px solid var(--signal)}.wrap{max-width:1296px;margin:auto;padding-left:clamp(20px,4vw,56px);
  padding-right:clamp(20px,4vw,56px)}.masthead .wrap{padding-top:42px;padding-bottom:42px}
.identity{font:600 .85rem/1.2 Bahnschrift,'Arial Narrow',Arial,sans-serif;color:#a9c9d1;
  letter-spacing:.02em;margin:0 0 24px}.masthead h1{font:600 clamp(2.3rem,4.4vw,4.3rem)/1.03
  Bahnschrift,'Arial Narrow',Arial,sans-serif;letter-spacing:-.035em;max-width:24ch;margin:0 0 18px}
.masthead p{max-width:70ch;color:#c6d8de;margin:0}.overview{background:var(--white);
  border-bottom:1px solid var(--line)}.overview .wrap{display:grid;grid-template-columns:minmax(0,1.3fr)
  minmax(300px,.8fr);gap:clamp(24px,4vw,60px);padding-top:34px;padding-bottom:34px}
.overview h2{font:600 1.45rem/1.2 Bahnschrift,'Arial Narrow',Arial,sans-serif;margin:0 0 12px}
.overview p{max-width:70ch;margin:0 0 12px}.overview .caution{font-size:.91rem;color:var(--muted)}
.comparison{border-collapse:collapse;width:100%;font-size:.89rem;align-self:start}
.comparison caption{font-weight:650;text-align:left;padding:0 0 10px;color:var(--ink)}
.comparison th,.comparison td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}
.comparison th:first-child,.comparison td:first-child{text-align:left;padding-left:0}
.comparison thead th{font-weight:600;color:var(--muted);white-space:nowrap}
.comparison tbody th{font-weight:500;color:var(--muted)}.comparison td{font-variant-numeric:tabular-nums}
.contents{background:#e3ebee;border-bottom:1px solid #c8d7dc}.contents .wrap{display:flex;
  flex-wrap:wrap;gap:2px 22px;align-items:center;padding-top:11px;padding-bottom:11px}
.contents strong{font:600 .82rem Bahnschrift,'Arial Narrow',Arial,sans-serif;color:#526873;
  margin-right:3px}.contents a{text-decoration:none;font-size:.86rem;font-weight:600;padding:5px 0}
.contents a:hover{text-decoration:underline}main.wrap{padding-top:28px;padding-bottom:70px}
.report-section{scroll-margin-top:20px;background:var(--white);border:1px solid var(--line);
  margin:0 0 20px;padding:clamp(18px,2.3vw,32px)}.report-section.primary{border-top:5px solid var(--signal)}
.section-heading{display:flex;align-items:baseline;justify-content:space-between;gap:14px;
  border-bottom:1px solid var(--line);padding-bottom:13px;margin-bottom:12px}
.section-heading h2{font:600 clamp(1.35rem,2vw,1.8rem)/1.2 Bahnschrift,'Arial Narrow',Arial,sans-serif;
  letter-spacing:-.015em;margin:0}.section-heading a{font-size:.8rem;white-space:nowrap}
.section-intro{color:var(--muted);max-width:75ch;font-size:.92rem;margin:0 0 8px}
.chart{min-width:0}.chart .js-plotly-plot{width:100%}.feature-list{list-style:none;margin:18px 0 0;
  padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:0 18px}
.feature-list li{border-bottom:1px solid var(--line);padding:8px 0;overflow-wrap:anywhere}
.feature-list code,code{font: .9em/1.5 Consolas,'Courier New',monospace}
.table-scroll{overflow:auto;max-height:540px;border:1px solid var(--line);margin-top:16px}
.constraints{border-collapse:collapse;width:100%;font-size:.85rem;min-width:720px}
.constraints th,.constraints td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left}
.constraints thead th{position:sticky;top:0;background:#e8eff1;color:var(--ink);z-index:1}
.constraints tbody tr:nth-child(even){background:#f7f9fa}.constraints tr.violation{background:#fcece9;color:var(--alert)}
.constraints td:nth-child(n+4){font-variant-numeric:tabular-nums}
.footer{color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);padding-top:20px}
@media(max-width:760px){.masthead .wrap{padding-top:29px;padding-bottom:29px}
  .overview .wrap{grid-template-columns:1fr;gap:18px}.report-section{padding:16px}
  .section-heading{align-items:flex-start}.feature-list{grid-template-columns:1fr 1fr}}
@media(max-width:520px){.feature-list{grid-template-columns:1fr}.contents .wrap{gap:0 14px}
  .comparison{font-size:.78rem}.comparison th,.comparison td{padding:7px 4px}
  .section-heading h2{font-size:1.3rem}.section-heading a{display:none}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:#fff}.masthead{background:#fff;color:#000;border-color:#000}
  .masthead p,.identity{color:#333}.contents,.section-heading a{display:none}
  .report-section{break-inside:avoid;border-color:#bbb}.table-scroll{max-height:none;overflow:visible}}
"""


def write_report(path: str | Path, payload: dict) -> Path:
    """Write one fully self-contained Plotly HTML artifact, usable without internet."""
    path = Path(path)
    figures = build_figures(payload)
    descriptions = {
        "frontier": ("Observed Pareto frontier", "Each point is an evaluated subset. The teal line joins confirmed feasible solutions; the star is the full-feature reference."),
        "cardinality": ("Performance by subset size", "Best observed full-validation Gini at each evaluated feature count. This is search evidence, not a complete sweep."),
        "frequency": ("Features in strong subsets", f"Inclusion frequency across distinct confirmed Pareto subsets within {payload.get('gini_tolerance', 0.005):g} Gini of the best observed result."),
        "coselection": ("Feature co-selection", "How often two features appear together in the confirmed Pareto set."),
        "temporal": ("Validation over time", "Fold-level Gini for the selected model, best confirmed result, reference, and other frontier subsets."),
        "segments": ("Segment performance", "Equal-fold mean Gini by segment. Any configured hard floors are checked per fold in the table below."),
        "history": ("Search history", "Confirmed frontier progress, evaluation counts, and population size across generations."),
    }
    sections = []
    for name, figure in figures.items():
        heading, description = descriptions[name]
        figure.update_layout(title=None, margin={"t": 35})
        chart = figure.to_html(full_html=False, include_plotlyjs=False, div_id="chart-" + name,
                               config={"responsive": True, "displaylogo": False})
        sections.append(f'<section class="report-section{" primary" if name == "frontier" else ""}" id="{name}">'
                        f'<div class="section-heading"><h2>{heading}</h2><a href="#top">Back to top</a></div>'
                        f'<p class="section-intro">{description}</p><div class="chart">{chart}</div></section>')
        if name == "frontier":
            feature_items = ''.join(f'<li><code>{escape(str(feature))}</code></li>' for feature in payload['best_features'])
            feature_body = (f'<ul class="feature-list">{feature_items}</ul>' if feature_items else
                            '<p>No features were selected in this saved result.</p>')
            sections.append('<section class="report-section" id="selected-features">'
                            '<div class="section-heading"><h2>Selected features</h2><a href="#top">Back to top</a></div>'
                            '<p class="section-intro">The feature set designated by this run for external validation.</p>'
                            f'{feature_body}</section>')
    constraints = segment_constraint_comparisons(payload)
    if constraints:
        selected_mask = [name in payload["best_features"] for name in payload["feature_names"]]
        designated = {r["key"] for r in payload.get("pareto", []) if r["mask"] == selected_mask}
        best = max(payload.get("pareto", []), key=lambda r: r["mean_gini"], default=None)
        if best:
            designated.add(best["key"])
        designated.add(payload["reference"]["key"])
        display = [row for row in constraints if row["fidelity"] == "full" and row["key"] in designated]
        violations = [row for row in constraints if row["fidelity"] == "full" and row["violated"]]
        display += [row for row in violations if row["key"] not in designated][:500]
        headings = ["Candidate", "Segment", "Fold", "Gini", "Required minimum", "Violation", "Status"]
        table_rows = []
        for row in display:
            values = [row["key"][:14], f"{row['column']}={row['value']}", row["fold"],
                      row["candidate_gini"], row["minimum_gini"], row["violation"],
                      "VIOLATION" if row["violated"] else ("UNDEFINED" if row["violated"] is None else "Pass")]
            cells = "".join(f"<td>{escape(f'{value:.5f}' if isinstance(value, float) else str(value))}</td>" for value in values)
            table_rows.append(f'<tr class="{"violation" if row["violated"] else ""}">{cells}</tr>')
        constraint_body = (f'<p class="section-intro">{len(violations)} violated full-CV comparisons. Selected, best and reference comparisons are shown, '
                           'followed by up to 500 other violations. Complete comparisons, support counts and fidelity tags '
                           f'are in <code>segment_constraints.{escape(str(payload.get("output_format", "parquet")))}</code>.</p>'
                           '<div class="table-scroll"><table class="constraints"><thead><tr>' +
                           ''.join(f'<th scope="col">{h}</th>' for h in headings) +
                           '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div>')
    else:
        constraint_body = ('<p class="section-intro">No active segment hard constraints were configured. '
                           'Any exclusions for insufficient event support are recorded in the experiment manifest.</p>')
    sections.append('<section class="report-section" id="constraint-table">'
                    '<div class="section-heading"><h2>Per-fold segment constraints</h2><a href="#top">Back to top</a></div>'
                    f'{constraint_body}</section>')

    names = payload["feature_names"]
    selected_mask = [name in payload["best_features"] for name in names]
    candidates = [r for r in payload["results"] if r["fidelity"] == "full"]
    selected_id = payload.get("selected_candidate_id")
    selected = next((r for r in candidates if r["key"] == selected_id), None) if selected_id else None
    if selected is None:
        selected = next((r for r in candidates if r["mask"] == selected_mask), None)
    reference = payload["reference"]
    def gini(result: dict | None) -> str:
        value = result.get("mean_gini") if result else None
        return f"{value:.4f}" if isinstance(value, (int, float)) else "—"

    converged = any(row.get("converged", False) for row in payload.get("history", []))
    links = [("frontier", "Frontier"), ("selected-features", "Selected features"),
             ("cardinality", "Subset size"), ("frequency", "Frequency"),
             ("coselection", "Co-selection"), ("temporal", "Over time"),
             ("segments", "Segments"), ("history", "Search history"),
             ("constraint-table", "Constraints")]
    navigation = ''.join(f'<a href="#{target}">{label}</a>' for target, label in links)
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feature selection | Development validation report</title><style>{_REPORT_CSS}</style>
<script>{get_plotlyjs()}</script></head><body id="top"><a class="skip-link" href="#report">Skip to report</a>
<header class="masthead"><div class="wrap"><p class="identity">Memetic NSGA-II / experiment report</p>
<h1>Feature selection under temporal validation</h1>
<p>Observed trade-offs from the development search. {len(payload['pareto'])} confirmed Pareto subsets; numerical convergence {'observed' if converged else 'not observed'}.</p>
</div></header><div class="overview"><div class="wrap"><div><h2>Read this result</h2>
<p>The selected subset is a candidate for an independent external OOT assessment. These plots describe evaluated subsets and do not establish a global optimum.</p>
<p class="caution">Repeated search creates selection optimism. Provisional fold-reduced measurements have a separate evaluation basis. Feature frequency and co-selection describe searched solution families, not statistical feature significance.</p></div>
<table class="comparison"><caption>Selected versus full-feature reference</caption><thead><tr><th scope="col">Measure</th><th scope="col">Selected</th><th scope="col">Reference</th></tr></thead>
<tbody><tr><th scope="row">Features</th><td>{len(payload['best_features'])} / {len(names)}</td><td>{reference['n_features']} / {len(names)}</td></tr>
<tr><th scope="row">Mean validation Gini</th><td>{gini(selected)}</td><td>{gini(reference)}</td></tr>
<tr><th scope="row">Evaluation</th><td>{'Full CV' if selected else 'Unavailable'}</td><td>Full CV</td></tr></tbody></table></div></div>
<nav class="contents" aria-label="Report sections"><div class="wrap"><strong>Explore</strong>{navigation}</div></nav>
<main class="wrap" id="report">{''.join(sections)}<p class="footer">Development validation evidence only. Review the saved tables and external OOT results before using a subset in production.</p></main>
<script>
(() => {{
  const charts = ['frontier', 'cardinality', 'temporal', 'segments', 'history'];
  const compact = matchMedia('(max-width: 640px)');
  function arrangeCharts() {{
    for (const name of charts) {{
      const plot = document.getElementById('chart-' + name);
      if (!plot || !plot.data) continue;
      const small = compact.matches;
      Plotly.relayout(plot, {{
        legend: small ? {{orientation:'h', x:0, y:-0.26, font:{{size:11}}}} :
                        {{orientation:'v', x:1.02, y:1, font:{{size:12}}}},
        margin: small ? {{l:55, r:12, t:25, b:name === 'history' ? 210 : 130}} :
                        {{l:70, r:30, t:35, b:70}}
      }});
    }}
    const frequency = document.getElementById('chart-frequency');
    if (frequency && frequency.data) Plotly.relayout(frequency, {{showlegend:!compact.matches}});
  }}
  arrangeCharts();
  compact.addEventListener('change', arrangeCharts);
}})();
</script></body></html>"""
    path.write_text(text, encoding="utf-8")
    return path.resolve()

def run_selector(X, y, *, id_col=None, segment_cols=None, time_values=None,
                 ids=None, feature_names=None, **selector_options):
    """Fit and return a selector in one notebook-friendly call.

    ``selector_options`` are MemeticNSGA2Selector constructor options. For
    arrays, supply time_values and ids (or set allow_repeated_ids=True).
    """
    selector = MemeticNSGA2Selector(**selector_options)
    return selector.fit(X, y, id_col=id_col, segment_cols=segment_cols,
                        time_values=time_values, ids=ids, feature_names=feature_names)

__version__ = "0.1.0"
__all__ = ["MemeticNSGA2Selector", "run_selector", "CVConfig", "OptimizationConfig",
           "ResourceConfig", "EvaluationConfig", "SegmentConstraint", "SelectionConfig", "WarmupResult",
           "make_synthetic_data", "load_experiment", "main", "rebuild_report",
           "aggregate_run_frequencies"]

if __name__ == "__main__":
    raise SystemExit(main())
