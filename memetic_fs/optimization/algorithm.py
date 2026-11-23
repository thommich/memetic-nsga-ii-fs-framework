"""Restartable pymoo NSGA-II with full-fidelity confirmation and local search."""
from __future__ import annotations

from collections.abc import Callable
from math import comb
import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.pntx import TwoPointCrossover

from ..progress import get_progress
from .archives import pareto_front, normalized_hypervolume
from .initialization import initial_population, normalize_subsets, rfe_seeds
from .local_search import local_neighbors
from .operators import CardinalityRepair, SubsetMutation
from .problem import FeatureSubsetProblem


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
