"""Vectorized pymoo problem; evaluation and scheduling belong to the evaluator."""
from __future__ import annotations
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
