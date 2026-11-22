"""Real full-row LightGBM integration, including durable restart and CLI paths."""
from __future__ import annotations
import subprocess
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml

from memetic_fs import MemeticNSGA2Selector
from memetic_fs.config import OptimizationConfig, ResourceConfig
from memetic_fs.persistence.checkpoints import read_json
from memetic_fs.synthetic import make_synthetic_data


@pytest.fixture
def development():
    return make_synthetic_data(n_rows=240, n_features=8, n_months=6, random_state=31)


def selector(directory, **overrides):
    settings = dict(checkpoint_dir=directory, time_cv="reference_month", feature_range=(1,6),
        model_params={"min_data_in_leaf": 5, "num_leaves": 7, "learning_rate": .15},
        num_boost_round=4, resources=ResourceConfig(max_workers=1, threads_per_worker=1),
        optimization=OptimizationConfig(population_size=6, max_generations=3,
            local_search_interval=2, local_search_budget=4, reduced_generations=1,
            rfe_seed_sizes=(2,4,6)), use_rfe_initialization=False, interactive_plots=False)
    settings.update(overrides)
    return MemeticNSGA2Selector(**settings)


def fit(selector, frame):
    return selector.fit(frame, "target", id_col="applicant_id",
                        segment_cols=["customer_type", "age_group"])


def frontier_summary(model):
    return [(r["key"], tuple(r["mask"]), r["mean_gini"], tuple(r["constraints"])) for r in model.confirmed_]


def test_real_search_reports_valid_frontier_and_beats_noise(tmp_path, development):
    model = fit(selector(tmp_path / "real"), development)
    assert len(model.feature_names_in_) == 8
    assert not set(model.feature_names_in_) & {"target", "reference_month", "applicant_id", "customer_type", "age_group"}
    assert model.best_features_
    assert model.reference_["n_features"] == 8
    assert all(r["fidelity"] == "full" and len(r["folds"]) == 3 for r in model.confirmed_)
    assert all(1 <= r["n_features"] <= 6 for r in model.confirmed_)
    assert all(r["mean_gini"] == pytest.approx(np.mean([f["gini"] for f in r["folds"]])) for r in model.confirmed_)
    assert any(model.history_["local_search_evaluations"] > 0)
    assert "last_1" in model.history_["fidelity"].tolist()
    noise = [name == "noise_0" for name in model.feature_names_in_]
    baseline = model.evaluator_.evaluate_many([noise])[0]
    assert max(r["mean_gini"] for r in model.confirmed_) > baseline["mean_gini"]
    assert all(f["validation_n"] == 40 for r in model.confirmed_ for f in r["folds"])
    assert all([f["train_n"] for f in r["folds"]] == [120,160,200] for r in model.confirmed_)
    before = len(model.evaluator_.cache.completed())
    original = model.evaluator_.evaluate_many([model.confirmed_[0]["mask"]])[0]
    again = model.evaluator_.evaluate_many([model.confirmed_[0]["mask"]])[0]
    assert again == original
    assert len(model.evaluator_.cache.completed()) == before
    assert isinstance(model.temporal_performance_, pd.DataFrame)
    assert isinstance(model.segment_performance_, pd.DataFrame)
    assert (model.directory_ / "checkpoint.json").is_file()
    assert (model.directory_ / "manifest.json").is_file()
    assert list(model.directory_.glob("*.parquet"))
    assert not list(model.directory_.glob("*.csv"))


def test_transform_manual_selection_and_final_fit(tmp_path, development):
    model = fit(selector(tmp_path / "manual", selection={"policy":"manual"}), development)
    assert model.best_features_ is None
    with pytest.raises(ValueError, match="select"):
        model.transform(development)
    candidate = model.confirmed_[0]
    chosen = model.select("manual", candidate_id=candidate["key"])
    pd.testing.assert_frame_equal(model.transform(development), development.loc[:, chosen])
    assert model.get_support().sum() == len(chosen)
    matrix = development.loc[:, list(model.feature_names_in_)].to_numpy()
    np.testing.assert_array_equal(model.transform(matrix), matrix[:, model.get_support()])
    booster = model.fit_final_model()
    assert booster.num_feature() == len(chosen)
    assert (model.directory_ / "final_model.txt").exists()
    assert model.select("knee")
    with pytest.warns(UserWarning, match="fewer than six"):
        assert model.select("statistical")
    assert model.select("tolerance", gini_tolerance=.005)


def test_completed_resume_reuses_all_model_evaluations(tmp_path, development):
    model = fit(selector(tmp_path / "resume"), development)
    count = len(model.evaluator_.cache.completed())
    resumed = MemeticNSGA2Selector.resume(model.directory_)
    assert frontier_summary(resumed) == frontier_summary(model)
    assert len(resumed.evaluator_.cache.completed()) == count
    pd.testing.assert_frame_equal(resumed.history_, model.history_)


def test_mid_generation_boundary_resume_matches_complete_run(tmp_path, development, monkeypatch):
    original = fit(selector(tmp_path / "complete"), development)
    interrupted = selector(tmp_path / "interrupted")
    old_checkpoint = interrupted._checkpoint
    def crash_after_checkpoint(state):
        old_checkpoint(state)
        if state["generation"] == 1:
            raise RuntimeError("simulated process interruption")
    monkeypatch.setattr(interrupted, "_checkpoint", crash_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated"):
        fit(interrupted, development)
    assert read_json(interrupted.directory_ / "checkpoint.json")["state"]["generation"] == 1
    resumed = MemeticNSGA2Selector.resume(interrupted.directory_)
    assert frontier_summary(resumed) == frontier_summary(original)
    pd.testing.assert_frame_equal(resumed.history_, original.history_)


def test_resume_rejects_mutated_persisted_data(tmp_path, development):
    model = fit(selector(tmp_path / "tampered"), development)
    path = next((model.directory_ / "data").glob("*.npy"))
    with path.open("ab") as stream:
        stream.write(b"unexpected change")
    with pytest.raises(ValueError, match="changed"):
        MemeticNSGA2Selector.resume(model.directory_)


def test_rfe_toggle_and_full_fidelity_trajectory(tmp_path, development):
    model = fit(selector(tmp_path / "rfe", use_rfe_initialization=True), development)
    assert model.rfe_trajectory_
    assert {2,4,6} <= {r["n_features"] for r in model.rfe_trajectory_}
    assert all(r["fidelity"] == "full" for r in model.rfe_trajectory_)


def test_real_parallel_matches_sequential(tmp_path, development):
    sequential = fit(selector(tmp_path / "sequential"), development)
    parallel = fit(selector(tmp_path / "parallel", resources=ResourceConfig(max_workers=2, threads_per_worker=1)), development)
    assert frontier_summary(parallel) == frontier_summary(sequential)
    assert parallel.best_features_ == sequential.best_features_
    pd.testing.assert_frame_equal(parallel.history_, sequential.history_)


def test_segment_constraints_and_small_support_exclusion(tmp_path, development):
    constraints = [{"column":"customer_type", "value":"client", "max_gini_drop":2.0},
                   {"column":"customer_type", "value":"prospect", "max_gini_drop":.01}]
    with pytest.warns(UserWarning, match="Disabling segment constraint") as records:
        model = fit(selector(tmp_path / "constraints", segment_constraints=constraints,
            evaluation={"min_segment_events":8, "min_segment_nonevents":8}), development)
    assert sum("Disabling segment constraint" in str(record.message) for record in records) == 1
    assert len(model.evaluator_.active_constraints) == 1
    assert len(model.evaluator_.excluded_constraints) == 1
    assert all(len(r["constraints"]) == 1 and r["constraints"][0] <= 0 for r in model.confirmed_)
    assert model.engine_.problem.n_ieq_constr == 1


def test_yaml_cli_run_report_resume(tmp_path, development):
    source = tmp_path / "dataset.parquet"
    development.to_parquet(source, index=False)
    directory = tmp_path / "cli_run"
    config = {"data":{"path":str(source), "target":"target", "time_column":"reference_month",
                "id_column":"applicant_id", "segment_columns":["customer_type", "age_group"]},
        "model":{"estimator":"lightgbm", "num_boost_round":4,
                 "params":{"min_data_in_leaf":5,"num_leaves":7}},
        "optimization":{"feature_range":[1,6], "population_size":6,"max_generations":2,
                        "use_rfe_initialization":False,"local_search_budget":4,
                        "local_search_interval":2,"reduced_generations":1},
        "resources":{"mode":"auto", "max_workers":1,"threads_per_worker":1},
        "output":{"experiment_dir":str(directory),"interactive_plots":False}}
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    for arguments in [["run","--config",str(yaml_path)], ["report","--experiment-dir",str(directory)],
                      ["resume","--experiment-dir",str(directory)]]:
        result = subprocess.run([sys.executable,"-m","memetic_fs.cli",*arguments], capture_output=True,
                                text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert (directory / "manifest.json").exists()
    assert (directory / "checkpoint.json").exists()


def test_synthetic_signal_components_are_reproducible():
    first = make_synthetic_data(n_rows=240,n_features=8,n_months=6,random_state=9)
    pd.testing.assert_frame_equal(first, make_synthetic_data(240,8,6,9))
    assert first["applicant_id"].is_unique
    assert first.groupby("reference_month")["target"].nunique().eq(2).all()
    assert first["signal_0"].corr(first["redundant_0"]) > .99
    assert 0 < first["customer_type"].eq("prospect").mean() < .2
    with pytest.raises(ValueError):
        make_synthetic_data(n_rows=20,n_months=6)


def test_manual_selection_survives_resume(tmp_path, development):
    fast = OptimizationConfig(population_size=2,max_generations=1,reduced_generations=0,
                              local_search_budget=0)
    model = fit(selector(tmp_path / "manual_resume",optimization=fast), development)
    candidate = model.confirmed_[0]
    selected = model.select("manual",candidate_id=candidate["key"])
    manifest = read_json(model.directory_ / "manifest.json")
    assert manifest["config"]["selection"]["policy"] == "manual"
    assert manifest["selected_candidate_id"] == candidate["key"]
    resumed = MemeticNSGA2Selector.resume(model.directory_)
    assert resumed.selection_.policy == "manual"
    assert resumed.selected_candidate_["key"] == candidate["key"]
    assert resumed.best_features_ == selected
    pd.testing.assert_frame_equal(resumed.transform(development),model.transform(development))


def test_final_categorical_model_predictions_survive_category_reorder_and_reload(tmp_path, development):
    frame = development.loc[:,["signal_1","target","reference_month","applicant_id",
                                "customer_type","age_group"]].copy()
    category_order = ["middle","high","low"]
    frame.insert(1,"risk_band",pd.cut(development["signal_0"],bins=[-np.inf,-.4,.4,np.inf],
                                     labels=["low","middle","high"]).cat.reorder_categories(category_order))
    fast = OptimizationConfig(population_size=2,max_generations=1,reduced_generations=0,
                              local_search_budget=0)
    model = fit(selector(tmp_path / "categorical",optimization=fast,feature_range=(2,2)),frame)
    assert model.best_features_ == ["signal_1","risk_band"]
    booster = model.fit_final_model()
    assert booster.pandas_categorical == [category_order]
    predictors = model.transform(frame)
    expected = booster.predict(predictors,num_threads=1)
    assert np.ptp(expected) > .01
    reordered = predictors.copy()
    reordered["risk_band"] = reordered["risk_band"].cat.reorder_categories(["high","low","middle"])
    assert not np.array_equal(predictors["risk_band"].cat.codes,reordered["risk_band"].cat.codes)
    np.testing.assert_array_equal(booster.predict(reordered,num_threads=1),expected)
    loaded = lgb.Booster(model_file=str(model.directory_ / "final_model.txt"))
    assert loaded.pandas_categorical == [category_order]
    np.testing.assert_array_equal(loaded.predict(reordered,num_threads=1),expected)
    np.testing.assert_array_equal(loaded.predict(predictors,num_threads=1),expected)
    assert read_json(model.directory_ / "final_model_metadata.json")["categories"]["risk_band"] == category_order


@pytest.mark.parametrize("subsets,message",[
    ([["not_a_feature"]],"Unknown"),
    ([[1,2,0,0,0,0,0,0]],"binary"),
    ([[1,1,1,1,1,1,1,1]],"cardinality"),
    ([["signal_0","signal_0"]],"duplicated"),
])
def test_invalid_initial_subsets_rejected_before_any_model_fit(tmp_path, development, monkeypatch, subsets, message):
    calls = []
    def forbidden_train(*args,**kwargs):
        calls.append(True)
        raise AssertionError("Invalid initialization must be rejected before any LightGBM training")
    monkeypatch.setattr(lgb,"train",forbidden_train)
    with pytest.raises(ValueError,match=message):
        fit(selector(tmp_path / "invalid_seed",initial_subsets=subsets),development)
    assert calls == []
