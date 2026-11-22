"""Full-row monthly folds, independent of the optimizer and estimator."""
from __future__ import annotations

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
