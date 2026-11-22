"""Transparent final-selection policies on a common confirmed CV basis."""
from __future__ import annotations
import warnings
import numpy as np
from .config import SelectionConfig

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
