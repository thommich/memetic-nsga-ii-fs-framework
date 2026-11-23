from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from memetic_fs.config import CVConfig, EvaluationConfig, ResourceConfig, SegmentConstraint
from memetic_fs.cv import build_folds
from memetic_fs.data import prepare_data, save_dataset
from memetic_fs.evaluation.lightgbm import Evaluator, score, selected_matrix, strict_params
from memetic_fs.evaluation import lightgbm as module


@pytest.fixture
def inputs(tmp_path):
    n = 240
    rng = np.random.default_rng(45)
    y = np.tile([0, 0, 1, 1], n // 4)
    frame = pd.DataFrame({"id": np.arange(n), "target": y,
                          "month": np.repeat(np.arange(202601, 202607), 40),
                          "segment": np.tile(["a", "b", "b", "a"], n // 4),
                          "signal": y.astype(np.float32) + rng.normal(0, 0.2, n).astype(np.float32),
                          "noise": rng.normal(size=n).astype(np.float32),
                          "row_marker": np.arange(n, dtype=np.float32)})
    data = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    path = save_dataset(data, tmp_path / "data")
    folds = build_folds(data.months, data.y, CVConfig())
    return data, path, folds


def make_evaluator(inputs, directory, **kwargs):
    _, path, folds = inputs
    return Evaluator(path, folds, directory, model_params={"min_data_in_leaf": 3, "num_leaves": 5},
                     num_boost_round=5, config=EvaluationConfig(min_segment_events=3, min_segment_nonevents=3),
                     resources=ResourceConfig(max_workers=1, threads_per_worker=1), **kwargs)


@pytest.mark.parametrize("params", [
    {"bagging_fraction": .8}, {"subsample": .9}, {"bagging_freq": 1},
    {"feature_fraction": .9}, {"colsample_bytree": .8}, {"feature_fraction_bynode": .5},
    {"pos_bagging_fraction": .8}, {"neg_bagging_fraction": .8},
    {"data_sample_strategy": "goss"}, {"boosting_type": "goss"},
    {"num_iterations": 4}, {"n_estimators": 4}, {"early_stopping_rounds": 1},
    {"subsample_for_bin": 5}, {"bin_construct_sample_cnt": 5},
    {"num_threads": 4}, {"random_state": 2}, {"device": "gpu"},
    {"feature_fraction": 1.0, "colsample_bytree": .9},
    {"mc": [1, 0]}, {"fc": [1, 1]}, {"categorical_features": [0]},
    {"forced_splits_file": "external.json"}, {"ignore_feature": "0"},
    {"interaction_constraints": [[0, 1]]}, {"tree_learner": "data"},
    {"pred_early_stop": True}, {"valid_data": "external.parquet"},
])
def test_strict_params_reject_sampling_and_alias_conflicts(params):
    with pytest.raises(ValueError):
        strict_params(params, seed=3, threads=2, n_rows=200)


def test_strict_full_row_settings():
    params = strict_params(None, seed=11, threads=2, n_rows=999_999)
    assert params["bin_construct_sample_cnt"] == 999_999
    assert params["bagging_fraction"] == params["feature_fraction"] == params["feature_fraction_bynode"] == 1
    assert params["bagging_freq"] == 0
    assert params["deterministic"] and params["force_col_wise"]
    assert params["num_threads"] == 2


def test_native_lightgbm_histogram_sample_indices_include_every_row():
    """Exercise native bin-sampling logic beyond LightGBM's default 200k rows.

    Signatures: https://lightgbm.readthedocs.io/en/stable/C-API.html
    This does not allocate a large predictor matrix or run a large optimization.
    """
    import ctypes
    from lightgbm.basic import _LIB, _safe_call
    try:
        get_count = _LIB.LGBM_GetSampleCount
        sample_indices = _LIB.LGBM_SampleIndices
    except AttributeError:
        pytest.skip("Installed LightGBM native library does not expose the documented sample-index C API")
    n = 200_001
    strict = strict_params(None, seed=123, threads=1, n_rows=n)
    parameters = ctypes.c_char_p(
        f"bin_construct_sample_cnt={strict['bin_construct_sample_cnt']} data_random_seed={strict['data_random_seed']}".encode("ascii"))
    count = ctypes.c_int()
    _safe_call(get_count(ctypes.c_int32(n), parameters, ctypes.byref(count)))
    assert count.value == n
    indices = np.empty(count.value, dtype=np.int32)
    actual_length = ctypes.c_int32()
    _safe_call(sample_indices(ctypes.c_int32(n), parameters, indices.ctypes.data_as(ctypes.c_void_p),
                             ctypes.byref(actual_length)))
    assert actual_length.value == n
    np.testing.assert_array_equal(indices, np.arange(n, dtype=np.int32))


def test_auc_gini_all_observations():
    result = score(np.array([0, 0, 1, 1]), np.array([.1, .2, .3, .4]))
    assert result == {"auc": 1.0, "gini": 1.0, "n": 4, "events": 2, "nonevents": 2}
    with pytest.raises(ValueError, match="both target classes"):
        score(np.ones(3), np.ones(3))


def test_every_full_fold_row_reaches_training_and_metrics(inputs, tmp_path, monkeypatch):
    data, _, folds = inputs
    seen = []
    class FakeModel:
        def __init__(self, columns):
            self.columns = columns
        def predict(self, matrix, **kwargs):
            return 1 / (1 + np.exp(-matrix[:, 0]))
        def feature_importance(self, **kwargs):
            return np.ones(self.columns)
    def train(params, dataset, num_boost_round):
        assert params["bin_construct_sample_cnt"] == len(dataset.data)
        assert params["bagging_freq"] == 0
        assert num_boost_round == 5
        seen.append(dataset.data.copy())
        return FakeModel(dataset.data.shape[1])
    monkeypatch.setattr(module.lgb, "train", train)
    evaluator = make_evaluator(inputs, tmp_path / "run")
    assert len(seen) == 3
    for trained, fold in zip(seen, folds):
        np.testing.assert_array_equal(trained, data.X[fold.train])
    assert [row["n"] for row in evaluator.reference["folds"]] == [40, 40, 40]
    result = evaluator.evaluate_many([[1, 0, 0]], "last_1")[0]
    assert len(result["folds"]) == 1
    assert result["folds"][0]["fold"] == folds[-1].label
    assert result["folds"][0]["validation_n"] == 40
    np.testing.assert_array_equal(seen[-1][:, 0], data.X[folds[-1].train, 0])
    full = evaluator.evaluate_many([[1, 0, 0]], "full")[0]
    assert len(full["folds"]) == len(folds)
    assert result["key"] != full["key"]
    assert full["mean_gini"] == np.mean([f["gini"] for f in full["folds"]])


def test_cache_reuses_completed_evaluations_and_deduplicates(inputs, tmp_path, monkeypatch):
    evaluator = make_evaluator(inputs, tmp_path / "run")
    mask = [1, 0, 0]
    results = evaluator.evaluate_many([mask, mask], "full")
    assert results[0] == results[1]
    calls = len(evaluator.cache.completed())
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed model evaluation was repeated")
    monkeypatch.setattr(module.lgb, "train", forbidden)
    assert evaluator.evaluate_many([mask])[0] == results[0]
    restarted = make_evaluator(inputs, tmp_path / "run")
    assert restarted.evaluate_many([mask])[0] == results[0]
    assert len(evaluator.cache.completed()) == calls


def test_full_feature_candidate_reuses_reference_fits(inputs, tmp_path, monkeypatch):
    evaluator = make_evaluator(inputs, tmp_path / "run", segment_constraints=[SegmentConstraint("segment", "a", max_gini_drop=.03)])
    def forbidden(*args, **kwargs):
        raise AssertionError("Full-feature reference model was needlessly fitted again")
    monkeypatch.setattr(module.lgb, "train", forbidden)
    candidate = evaluator.evaluate_many([[1, 1, 1]])[0]
    assert candidate["mean_gini"] == evaluator.reference["mean_gini"]
    assert candidate["folds"] == evaluator.reference["folds"]
    assert candidate["constraints"] == [-.03]
    assert candidate["key"] != evaluator.reference["key"]
    assert evaluator.cache.get(candidate["key"]) == candidate


def test_thread_count_participates_in_evaluator_identity(inputs, tmp_path):
    from memetic_fs.execution import resources
    if resources.detect_resources(tmp_path).cpus < 2:
        pytest.skip("This host exposes only one effective CPU")
    _, path, folds = inputs
    one = make_evaluator(inputs, tmp_path / "one")
    two = Evaluator(path, folds, tmp_path / "two", model_params={"min_data_in_leaf": 3, "num_leaves": 5},
                    num_boost_round=5, config=EvaluationConfig(min_segment_events=3, min_segment_nonevents=3),
                    resources=ResourceConfig(max_workers=1, threads_per_worker=2))
    assert one.training_threads == 1 and two.training_threads == 2
    assert one.identity != two.identity


def test_cache_identity_changes_with_model_and_rounds(inputs, tmp_path):
    _, path, folds = inputs
    baseline = make_evaluator(inputs, tmp_path / "run")
    changed_rounds = Evaluator(path, folds, tmp_path / "run", num_boost_round=2,
                              model_params={"min_data_in_leaf": 3, "num_leaves": 5},
                              config=EvaluationConfig(min_segment_events=3, min_segment_nonevents=3),
                              resources=ResourceConfig(max_workers=1, threads_per_worker=1))
    changed_params = Evaluator(path, folds, tmp_path / "run", num_boost_round=5,
                              model_params={"min_data_in_leaf": 3, "num_leaves": 7},
                              config=EvaluationConfig(min_segment_events=3, min_segment_nonevents=3),
                              resources=ResourceConfig(max_workers=1, threads_per_worker=1))
    assert len({baseline.identity, changed_rounds.identity, changed_params.identity}) == 3
    assert len({baseline.reference["key"], changed_rounds.reference["key"], changed_params.reference["key"]}) == 3


def test_categorical_indices_remapped_for_selected_subset(tmp_path, monkeypatch):
    n = 240
    frame = pd.DataFrame({"id": np.arange(n), "target": np.tile([0, 1], n // 2),
                          "month": np.repeat(np.arange(202601, 202607), 40),
                          "numeric": np.arange(n, dtype=np.float32),
                          "category": pd.Categorical(np.tile(["red", "blue"], n // 2))})
    data = prepare_data(frame, "target", "month", id_col="id")
    path = save_dataset(data, tmp_path / "data")
    folds = build_folds(data.months, data.y, CVConfig())
    seen = []
    real_train = module.lgb.train
    def train(params, dataset, **kwargs):
        seen.append(dataset.categorical_feature)
        return real_train(params, dataset, **kwargs)
    monkeypatch.setattr(module.lgb, "train", train)
    evaluator = Evaluator(path, folds, tmp_path / "run", num_boost_round=2,
                          resources=ResourceConfig(max_workers=1, threads_per_worker=1))
    assert seen == [[1], [1], [1]]
    evaluator.evaluate_many([[0, 1]])
    assert seen[-3:] == [[0], [0], [0]]


def test_segment_constraints_and_insufficient_support_warn_once(inputs, tmp_path):
    constraint = SegmentConstraint("segment", "a", max_gini_drop=2)
    evaluator = make_evaluator(inputs, tmp_path / "eligible", segment_constraints=[constraint])
    assert len(evaluator.active_constraints) == 1
    result = evaluator.evaluate_many([[1, 0, 0]])[0]
    assert len(result["constraints"]) == 1
    assert result["constraints"][0] <= 0
    assert len(result["segments"]) == 6
    assert all(row["events"] == row["nonevents"] == 10 for row in result["segments"])
    _, path, folds = inputs
    with pytest.warns(UserWarning, match="Disabling segment constraint") as warnings:
        unsupported = Evaluator(path, folds, tmp_path / "unsupported", num_boost_round=2,
                                resources=ResourceConfig(max_workers=1, threads_per_worker=1),
                                segment_constraints=[constraint])
        unsupported.evaluate_many([[1, 0, 0]])
        unsupported.evaluate_many([[0, 1, 0]])
    assert sum("Disabling segment constraint" in str(w.message) for w in warnings) == 1
    assert unsupported.active_constraints == []
    assert len(unsupported.excluded_constraints) == 1
    assert unsupported.evaluate_many([[1, 0, 0]])[0]["constraints"] == []


def test_constraint_violation_is_explicit(inputs, tmp_path, monkeypatch):
    evaluator = make_evaluator(inputs, tmp_path / "run", segment_constraints=[SegmentConstraint("segment", "a", max_gini_drop=0)])
    class ConstantModel:
        def predict(self, data, **kwargs):
            return np.full(len(data), .5)
        def feature_importance(self, **kwargs):
            return np.ones(1)
    monkeypatch.setattr(module.lgb, "train", lambda *a, **kw: ConstantModel())
    candidate = evaluator.evaluate_many([[0, 1, 0]])[0]
    assert candidate["constraints"][0] > 0


def test_calibration_respects_resource_admission(inputs, tmp_path, monkeypatch):
    from memetic_fs.execution import resources
    actual = resources.detect_resources(tmp_path)
    monkeypatch.setattr(resources, "detect_resources", lambda _: replace(actual, available_bytes=0))
    _, path, folds = inputs
    def forbidden(*args, **kwargs):
        raise AssertionError("Training bypassed resource admission")
    monkeypatch.setattr(module.lgb, "train", forbidden)
    with pytest.raises(MemoryError, match="No full-fold worker fits"):
        Evaluator(path, folds, tmp_path / "run", num_boost_round=2,
                  resources=ResourceConfig(max_workers=1, threads_per_worker=1, memory_wait_seconds=0))


def test_deterministic_sequential_parallel(inputs, tmp_path):
    _, path, folds = inputs
    params = {"min_data_in_leaf": 3, "num_leaves": 5}
    sequential = make_evaluator(inputs, tmp_path / "serial")
    parallel = Evaluator(path, folds, tmp_path / "parallel", model_params=params, num_boost_round=5,
                         config=EvaluationConfig(min_segment_events=3, min_segment_nonevents=3),
                         resources=ResourceConfig(max_workers=2, threads_per_worker=1))
    masks = [[1, 0, 0], [1, 1, 0]]
    left = sequential.evaluate_many(masks)
    right = parallel.evaluate_many(masks)
    for a, b in zip(left, right):
        assert a["mean_gini"] == b["mean_gini"]
        assert a["folds"] == b["folds"]


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.bool_, np.float16])
def test_integer_and_low_precision_matrix_conversion(dtype):
    data = np.array([[0, 1], [1, 0]], dtype=dtype)
    matrix = selected_matrix(data, np.array([0, 1]), np.array([0]))
    assert matrix.dtype in (np.float32, np.float64)
    np.testing.assert_array_equal(matrix, [[0], [1]])


def test_integer_precision_loss_rejected():
    data = np.array([[2**53 + 1], [0]], dtype=np.int64)
    with pytest.raises(ValueError, match="exact float64 range"):
        selected_matrix(data, np.array([0, 1]), np.array([0]))


def test_missing_segment_values_and_constraints(tmp_path):
    n = 240
    y = np.tile([0, 0, 1, 1], n // 4)
    frame = pd.DataFrame({"id": np.arange(n), "target": y,
                          "month": np.repeat(np.arange(202601, 202607), 40),
                          "segment": np.tile([None, "a", "a", pd.NA], n // 4),
                          "signal": y.astype(np.float32), "noise": np.zeros(n, dtype=np.float32)})
    data = prepare_data(frame, "target", "month", id_col="id", segment_cols=["segment"])
    path = save_dataset(data, tmp_path / "data")
    folds = build_folds(data.months, data.y, CVConfig())
    evaluator = Evaluator(path, folds, tmp_path / "run", num_boost_round=2,
                          resources=ResourceConfig(max_workers=1, threads_per_worker=1),
                          config=EvaluationConfig(min_segment_events=3, min_segment_nonevents=3),
                          segment_constraints=[SegmentConstraint("segment", np.nan, max_gini_drop=2)])
    assert evaluator.active_constraints[0]["value"] is None
    candidate = evaluator.evaluate_many([[1, 0]])[0]
    assert len(candidate["segments"]) == 6
    assert len([r for r in candidate["segments"] if r["value"] is None]) == 3
    assert candidate["constraints"][0] <= 0


def test_worker_failure_is_persisted(inputs, tmp_path, monkeypatch):
    evaluator = make_evaluator(inputs, tmp_path / "run")
    def bad_train(*args, **kwargs):
        raise ValueError("broken model setting")
    monkeypatch.setattr(module.lgb, "train", bad_train)
    with pytest.raises(ValueError, match="broken model setting"):
        evaluator.evaluate_many([[0, 1, 0]])
    incomplete = evaluator.cache.incomplete()
    assert len(incomplete) == 1
    assert incomplete[0]["status"] == "failed"
    assert "broken model setting" in incomplete[0]["error"]
    assert evaluator.cache.get(evaluator.reference["key"]) is not None


def test_lightgbm_native_memory_failure_retries(inputs, tmp_path, monkeypatch):
    evaluator = make_evaluator(inputs, tmp_path / "run")
    real_train = module.lgb.train
    attempts = []
    def transient(params, dataset, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise module.lgb.basic.LightGBMError("std::bad_alloc")
        return real_train(params, dataset, **kwargs)
    monkeypatch.setattr(module.lgb, "train", transient)
    result = evaluator.evaluate_many([[1, 0, 0]])[0]
    assert result["fidelity"] == "full"
    assert len(attempts) == 4
    assert evaluator.cache.incomplete() == []


def test_evaluator_reports_fold_cache_and_duplicate_progress(inputs, tmp_path, monkeypatch):
    class Reporter:
        descriptor = None
        def __init__(self):
            self.events = []
        def info(self, message, **fields):
            self.events.append(("info", message, fields))
        milestone = info
        def worker_status(self, **fields):
            self.events.append(("workers", "Worker status", fields))
        def debug(self, message, **fields):
            self.events.append(("debug", message, fields))
        def warning(self, message, **fields):
            self.events.append(("warning", message, fields))
        def error(self, message, **fields):
            self.events.append(("error", message, fields))
    reporter = Reporter()
    monkeypatch.setattr(module, "progress_from_descriptor", lambda descriptor: reporter)
    evaluator = make_evaluator(inputs, tmp_path / "run", progress=reporter)
    evaluator.evaluate_many([[1, 0, 0], [1, 0, 0]])
    evaluator.evaluate_many([[1, 0, 0]])
    reference_starts = [e for e in reporter.events if e[:2] == ("info", "Temporal fold started")]
    candidate_ends = [e for e in reporter.events if e[:2] == ("debug", "Temporal fold completed")]
    assert len(reference_starts) == len(candidate_ends) == 3
    assert all(e[2]["validation_n"] == 40 for e in reference_starts)
    assert all("gini" in e[2] and e[2]["elapsed_seconds"] >= 0 for e in candidate_ends)
    summaries = [e[2] for e in reporter.events if e[1] == "Evaluation batch prepared"]
    assert summaries[-2]["requested_candidates"] == 2
    assert summaries[-2]["unique_candidates"] == 1
    assert summaries[-2]["pending_duplicates"] == 1
    assert summaries[-1]["cache_hits"] == 1
    assert any(e[1] == "Worker batch admitted" for e in reporter.events)
    assert any(e[1] == "Full-feature reference calibration completed" for e in reporter.events)


def test_parallel_workers_write_separate_progress_files(inputs, tmp_path):
    import os
    from memetic_fs.execution.resources import detect_resources
    from memetic_fs.progress import ProgressReporter
    if detect_resources(tmp_path).cpus < 2:
        pytest.skip("This host exposes only one effective CPU")
    _, path, folds = inputs
    directory = tmp_path / "logged-run"
    reporter = ProgressReporter(directory, verbose=0)
    evaluator = Evaluator(path, folds, directory, model_params={"min_data_in_leaf": 3, "num_leaves": 5},
                          num_boost_round=3, config=EvaluationConfig(min_segment_events=3, min_segment_nonevents=3),
                          resources=ResourceConfig(max_workers=2, threads_per_worker=1), progress=reporter)
    evaluator.evaluate_many([[1, 0, 0], [0, 1, 0]])
    assert "Evaluation batch prepared" in (directory / "logs" / "run.log").read_text()
    workers = list((directory / "logs").glob("worker-*.log"))
    assert workers
    assert all(path.name != f"worker-{os.getpid()}.log" for path in workers)
    details = "\n".join(path.read_text() for path in workers)
    assert "Candidate evaluation started" in details
    assert "Temporal fold completed" in details
    assert "mean_gini=" in details
