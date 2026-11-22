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
