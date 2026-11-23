"""Cheap deterministic fitness isolates optimizer behavior from model training."""
from copy import deepcopy
import json
import numpy as np
import pytest
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.operators.crossover.pntx import TwoPointCrossover

from memetic_fs.config import OptimizationConfig
from memetic_fs.optimization import SearchEngine, pareto_front
from memetic_fs.optimization.operators import (validate_mask, repair_mask, propose_neighbor,
    cardinality_probabilities, CardinalityRepair, SubsetMutation)
from memetic_fs.optimization.initialization import normalize_subsets, initial_population
from memetic_fs.optimization.local_search import local_neighbors


class FakeEvaluator:
    def __init__(self, n=8, constraints=False):
        self.n = n
        self.calls = []
        self.all_results = {}
        self.constrained = constraints
        self.reference = self.evaluate_many([np.ones(n, dtype=bool)])[0]

    def evaluate_many(self, masks, fidelity="full"):
        results = []
        for mask in masks:
            mask = np.asarray(mask, dtype=bool)
            key = fidelity + "".join(str(int(i)) for i in mask)
            if key not in self.all_results:
                self.calls.append((tuple(mask), fidelity))
                # Sparse complementary signal; irrelevant features mildly hurt.
                gini = .2*mask[0] + .25*mask[1] + .1*(mask[2] and mask[3]) - .01*sum(mask[4:])
                if fidelity != "full":
                    gini += .02*mask[-1]
                self.all_results[key] = {"key": key, "mask": mask.tolist(), "n_features": int(mask.sum()),
                    "mean_gini": float(gini), "fidelity": fidelity,
                    "importance": [8, 7, 6, 5] + [1]*(self.n-4),
                    "constraints": ([float(.3-gini)] if self.constrained else []),
                    "folds": [{"gini": float(gini)}]*3}
            results.append(self.all_results[key])
        return results


def settings(**kwargs):
    return OptimizationConfig(population_size=10, max_generations=6, min_generations=20,
        local_search_interval=2, local_search_budget=12, reduced_generations=2, **kwargs)


def test_mask_validation_and_repair():
    with pytest.raises(ValueError, match="binary"):
        validate_mask([0, 2, 1], 3, (1, 2))
    with pytest.raises(ValueError, match="cardinality"):
        validate_mask([1, 1, 1], 3, (1, 2))
    rng = np.random.default_rng(4)
    assert repair_mask(np.zeros(10), (3, 6), rng).sum() == 3
    assert repair_mask(np.ones(10), (3, 6), rng).sum() == 6


def test_mutation_operations_and_feature_reintroduction():
    mask = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    rng = np.random.default_rng(8)
    for operation, cardinality, distance in [("add", 5, 1), ("remove", 3, 1), ("swap", 4, 2), ("swap2", 4, 4)]:
        neighbor = propose_neighbor(mask, (1, 7), rng, operation)
        assert neighbor.sum() == cardinality
        assert np.count_nonzero(mask != neighbor) == distance
    neighbor = propose_neighbor(mask, (1, 7), rng, "group", groups=[[4, 5, 6]])
    assert neighbor[[4, 5, 6]].sum() == 3
    assert neighbor.sum() == 4


def test_cardinality_bias_keeps_exploration_floor():
    counts, probabilities = cardinality_probabilities((1, 200))
    assert probabilities[0] > probabilities[-1] > 0
    assert probabilities.sum() == pytest.approx(1)
    assert (probabilities >= .15/len(counts)).all()


def test_pymoo_binary_crossover_mutation_and_repair():
    rng = np.random.default_rng(42)
    problem = Problem(n_var=8, n_obj=2, xl=0, xu=1, vtype=bool)
    pop = Population.new("X", np.array([[1]*4+[0]*4, [0]*4+[1]*4], dtype=bool))
    offspring = TwoPointCrossover(prob=1).do(problem, pop, parents=np.array([[0, 1]]), random_state=rng)
    mutated = SubsetMutation((2, 6), settings()).do(problem, offspring, random_state=rng)
    repaired = CardinalityRepair((2, 6)).do(problem, mutated, random_state=rng)
    assert repaired.get("X").dtype == bool
    assert all(2 <= mask.sum() <= 6 for mask in repaired.get("X"))


def test_user_seeds_invalid_unknown_and_deduplicated():
    names = ["a", "b", "c"]
    assert len(normalize_subsets([["a"], [1, 0, 0]], names, (1, 2))) == 1
    with pytest.raises(ValueError, match="Unknown"):
        normalize_subsets([["missing"]], names, (1, 2))
    with pytest.raises(ValueError, match="duplicated"):
        normalize_subsets([["a", "a"]], names, (1, 2))


def test_pareto_archive_never_mixes_fidelity_or_infeasibility():
    ev = FakeEvaluator()
    a, b, c = ev.evaluate_many([[1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0], [1,1,0,0,1,0,0,0]])
    cheap = dict(c, mean_gini=1., fidelity="last_1")
    infeasible = dict(a, mean_gini=1., constraints=[1.])
    assert [r["n_features"] for r in pareto_front([a,b,c,cheap,infeasible])] == [1,2]


def test_search_reproducible_and_local_search_operational():
    ev1, ev2 = FakeEvaluator(), FakeEvaluator()
    result1 = SearchEngine(ev1, 8, (1, 6), settings(), use_rfe_initialization=False).run()
    result2 = SearchEngine(ev2, 8, (1, 6), settings(), use_rfe_initialization=False).run()
    assert result1["history"] == result2["history"]
    assert result1["confirmed"] == result2["confirmed"]
    assert result1["confirmed"]
    assert all(r["fidelity"] == "full" for r in result1["confirmed"])
    assert max(r["mean_gini"] for r in result1["confirmed"]) >= .45
    assert any(h["local_search_evaluations"] > 0 for h in result1["history"])
    assert all(len(set(tuple(r["mask"]) for r in result1["confirmed"])) == len(result1["confirmed"]) for _ in [0])


def test_no_rfe_fits_when_disabled_and_rfe_checkpoints_when_enabled():
    disabled = SearchEngine(FakeEvaluator(), 8, (1,6), settings(), use_rfe_initialization=False).run()
    enabled = SearchEngine(FakeEvaluator(), 8, (1,6), settings(), use_rfe_initialization=True).run()
    assert disabled["rfe_trajectory"] == []
    assert enabled["rfe_trajectory"]
    assert all(r["fidelity"] == "full" for r in enabled["rfe_trajectory"])


@pytest.mark.parametrize("generation", [1, 2, 3, 4, 5])
def test_json_checkpoint_resume_matches_uninterrupted_run(generation):
    ev = FakeEvaluator()
    uninterrupted = SearchEngine(ev, 8, (1,6), settings(), use_rfe_initialization=False).run()
    states = []
    other = FakeEvaluator()
    SearchEngine(other, 8, (1,6), settings(), use_rfe_initialization=False).run(
        checkpoint_callback=lambda state: states.append(json.loads(json.dumps(deepcopy(state), allow_nan=False))))
    checkpoint = next(s for s in states if s["generation"] == generation)
    resumed = SearchEngine(other, 8, (1,6), settings(), use_rfe_initialization=False).run(state=checkpoint)
    assert resumed["confirmed"] == uninterrupted["confirmed"]
    assert resumed["history"] == uninterrupted["history"]
    assert resumed["state"]["optimizer_rng_state"] == uninterrupted["state"]["optimizer_rng_state"]


def test_constraints_applied_to_final_archive():
    ev = FakeEvaluator(constraints=True)
    result = SearchEngine(ev, 8, (1,6), settings(), use_rfe_initialization=False).run()
    assert result["confirmed"]
    assert all(max(r["constraints"]) <= 0 for r in result["confirmed"])


def test_small_search_space_exits_without_duplicate_loop():
    ev = FakeEvaluator(n=4)
    result = SearchEngine(ev, 4, (4,4), settings(), use_rfe_initialization=False).run()
    assert len(result["confirmed"]) == 1
    assert result["stop_reason"] == "search_space_exhausted"


def test_cancellation_keeps_completed_boundary():
    ev = FakeEvaluator()
    counter = {"count": 0}
    def checkpoint(state):
        counter["count"] += 1
    result = SearchEngine(ev,8,(1,6),settings(),use_rfe_initialization=False).run(
        checkpoint_callback=checkpoint, cancel=lambda: counter["count"] >= 2)
    assert result["stop_reason"] == "cancelled"
    assert result["state"]["generation"] == 2


def test_cancellation_before_initialization_is_actionable():
    with pytest.raises(InterruptedError, match="before initialization"):
        SearchEngine(FakeEvaluator(),8,(1,6),settings(),use_rfe_initialization=False).run(cancel=lambda: True)


def test_keyboard_interrupt_before_first_boundary_propagates():
    evaluator = FakeEvaluator()
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    evaluator.evaluate_many = interrupt
    with pytest.raises(KeyboardInterrupt):
        SearchEngine(evaluator,8,(1,6),settings(),use_rfe_initialization=False).run()


def test_converged_checkpoint_resume_is_noop():
    config = OptimizationConfig(population_size=6,max_generations=6,min_generations=1,
                                patience=1,min_improvement=100.,reduced_generations=0)
    evaluator = FakeEvaluator()
    completed = SearchEngine(evaluator,8,(1,6),config,use_rfe_initialization=False).run()
    assert completed["stop_reason"] == "pareto_convergence"
    resumed = SearchEngine(evaluator,8,(1,6),config,use_rfe_initialization=False).run(state=completed["state"])
    assert resumed["history"] == completed["history"]
    assert resumed["confirmed"] == completed["confirmed"]


def test_single_predictor_search_space():
    class OneFeatureEvaluator:
        def __init__(self):
            self.reference = {"key":"one", "mask":[True], "mean_gini":.5,"n_features":1,
                              "importance":[1.],"constraints":[],"fidelity":"full"}
        def evaluate_many(self,masks,fidelity="full"):
            return [dict(self.reference,fidelity=fidelity) for _ in masks]
    completed = SearchEngine(OneFeatureEvaluator(),1,(1,1),settings(),use_rfe_initialization=False).run()
    assert len(completed["confirmed"]) == 1
    assert completed["stop_reason"] == "search_space_exhausted"


def test_all_user_subsets_evaluated_when_more_than_population_size():
    evaluator = FakeEvaluator()
    seeds = np.eye(8,dtype=bool).tolist()
    config = OptimizationConfig(population_size=2,max_generations=1,reduced_generations=0,
                                local_search_budget=0)
    SearchEngine(evaluator,8,(1,6),config,use_rfe_initialization=False,initial_subsets=seeds).run()
    evaluated = {mask for mask, fidelity in evaluator.calls if fidelity == "full"}
    assert {tuple(mask) for mask in seeds} <= evaluated


class ProgressRecorder:
    def __init__(self):
        self.records = []
    def info(self,message,**fields):
        self.records.append(("info",message,fields))
    milestone = info
    def generation(self, number):
        pass
    def generation_completed(self, number):
        pass
    def debug(self,message,**fields):
        self.records.append(("debug",message,fields))
    def warning(self,message,**fields):
        self.records.append(("warning",message,fields))
    def error(self,message,**fields):
        self.records.append(("error",message,fields))


def test_progress_describes_generation_fidelity_and_local_search_without_changing_results():
    evaluator = FakeEvaluator()
    evaluator.progress = ProgressRecorder()
    engine = SearchEngine(evaluator,8,(1,6),settings(),use_rfe_initialization=False)
    result = engine.run()
    messages = [message for _,message,_ in evaluator.progress.records]
    assert messages[0] == "Optimization started"
    assert "Population initialization completed" in messages
    assert sum(message.startswith("Running generation ") for message in messages) == len(result["history"])
    ends = [fields for _,message,fields in evaluator.progress.records if message == "Generation completed"]
    assert len(ends) == len(result["history"])
    assert all({"best_confirmed_gini","confirmed_hypervolume","confirmed_count","provisional_count",
                "min_features","mean_features","max_features","unique_evaluations",
                "generation_seconds","elapsed_seconds"} <= set(fields) for fields in ends)
    transitions = [fields for _,message,fields in evaluator.progress.records if message.startswith("Changing fidelity")]
    assert len(transitions) == 1
    assert transitions[0]["previous_fidelity"] == "last_1"
    assert transitions[0]["fidelity"] == "full"
    assert "Local search started" in messages
    assert "Local search completed" in messages
    assert "Final population confirmation started" in messages
    assert messages[-1] == "Optimization completed"
    assert not any("RFE" in message for message in messages)
    assert all(not {"mask","feature_names","segment_values","rows"} & set(fields)
               for _,_,fields in evaluator.progress.records)
    assert all(not any("seconds" in key for key in row) for row in result["history"])
    quiet = SearchEngine(FakeEvaluator(),8,(1,6),settings(),use_rfe_initialization=False).run()
    assert result["history"] == quiet["history"]
    assert result["confirmed"] == quiet["confirmed"]


def test_rfe_progress_covers_every_batch_and_target():
    evaluator = FakeEvaluator()
    evaluator.progress = ProgressRecorder()
    SearchEngine(evaluator,8,(1,6),settings(),use_rfe_initialization=True).run()
    records = evaluator.progress.records
    messages = [message for _,message,_ in records]
    assert "RFE initialization started" in messages
    assert "RFE initialization completed" in messages
    assert messages.count("RFE target started") == messages.count("RFE target completed")
    assert messages.count("RFE batch started") == messages.count("RFE batch completed") > 0
    for _,message,fields in records:
        if message == "RFE batch started":
            assert fields["features_before"] - fields["features_after"] == fields["removed_features"]
            assert fields["fidelity"] == "full"


def test_progress_reports_resume_convergence_and_cancellation():
    evaluator = FakeEvaluator()
    evaluator.progress = ProgressRecorder()
    config = OptimizationConfig(population_size=6,max_generations=6,min_generations=1,
                                patience=1,min_improvement=100.,reduced_generations=0)
    first = SearchEngine(evaluator,8,(1,6),config,use_rfe_initialization=False).run()
    assert any("frontier converged" in message for _,message,_ in evaluator.progress.records)
    evaluator.progress.records.clear()
    SearchEngine(evaluator,8,(1,6),config,use_rfe_initialization=False).run(state=first["state"])
    messages = [message for _,message,_ in evaluator.progress.records]
    assert "Optimization checkpoint restored" in messages
    assert "Optimization already completed; retaining checkpoint result" in messages
    assert "Generation started" not in messages
    evaluator.progress.records.clear()
    with pytest.raises(InterruptedError):
        SearchEngine(evaluator,8,(1,6),config,use_rfe_initialization=False).run(cancel=lambda:True)
    assert any(level == "warning" and "cancelled before initialization" in message
               for level,message,_ in evaluator.progress.records)


def test_progress_reports_interruption_before_completed_generation():
    evaluator = FakeEvaluator()
    evaluator.progress = ProgressRecorder()
    def interrupt(*args,**kwargs):
        raise KeyboardInterrupt
    evaluator.evaluate_many = interrupt
    with pytest.raises(KeyboardInterrupt):
        SearchEngine(evaluator,8,(1,6),settings(),use_rfe_initialization=False).run()
    warnings = [(message,fields) for level,message,fields in evaluator.progress.records if level == "warning"]
    assert len(warnings) == 1
    assert "completed evaluations remain cached" in warnings[0][0]
    assert warnings[0][1]["completed_generation"] == 0
