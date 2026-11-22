"""Cardinality-aware binary proposals used by initialization and local search."""
from __future__ import annotations

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
