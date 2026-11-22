"""Budgeted diverse local refinement allowing removed features to return."""
from __future__ import annotations
import numpy as np
from .operators import propose_neighbor


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
