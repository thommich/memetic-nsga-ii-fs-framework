"""Diverse random, importance, family and recursive-elimination seeds."""
from __future__ import annotations
from math import comb
import time
import numpy as np
from ..progress import get_progress
from .operators import validate_mask, cardinality_probabilities


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
