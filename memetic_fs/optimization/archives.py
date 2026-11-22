"""Common-fidelity feasible Pareto archives backed by pymoo."""
from __future__ import annotations
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
