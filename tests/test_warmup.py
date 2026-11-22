"""Warm-up recommendations, YAML precedence, and search compatibility."""
from pathlib import Path

import pytest
import yaml

from memetic_fs import MemeticNSGA2Selector, WarmupResult
from memetic_fs.synthetic import make_synthetic_data


@pytest.fixture
def frame():
    return make_synthetic_data(n_rows=240, n_features=8, n_months=6, random_state=31)


def warm(frame, path, **kwargs):
    return MemeticNSGA2Selector.warmup_settings_params(
        frame, "target", output_path=path, id_col="applicant_id",
        segment_cols=["customer_type", "age_group"], **kwargs)


def test_quick_writes_separate_yaml_and_merges_inline_overrides(tmp_path, frame):
    source = tmp_path / "original.yaml"
    source.write_text(yaml.safe_dump({"time_cv": "reference_month", "feature_range": [2, 8],
        "model_params": {"num_leaves": 7, "min_data_in_leaf": 5},
        "optimization": {"population_size": 6},
        "checkpoint_dir": "run"}), encoding="utf-8")
    original = source.read_bytes()
    result = warm(frame, tmp_path / "recommended.yaml", depth="quick", base_config=source)
    assert isinstance(result, WarmupResult)
    assert source.read_bytes() == original
    assert result.diagnostics["planned_fits"] == 0
    assert len(result.diagnostics["months"]) == 6
    assert result.suggestions["feature_range"] == [1, 8]
    assert "feature_range" in result.changes
    assert result.suggestions["selection"]["gini_tolerance"] <= .001
    assert any("heuristic" in message.lower() for message in result.warnings)
    selector = MemeticNSGA2Selector.from_config(result.settings_path,
        optimization={"max_generations": 3}, model_params={"num_leaves": 11})
    assert selector.optimization_.population_size == 6
    assert selector.optimization_.max_generations == 3
    assert selector.model_params["num_leaves"] == 11
    assert selector.model_params["min_data_in_leaf"] == 5
    assert selector.checkpoint_dir == str((tmp_path / "run").resolve())
    result.settings_path.write_text("invalid: true\n", encoding="utf-8")
    assert selector.optimization_.population_size == 6  # configuration is resolved once
    with pytest.raises(FileExistsError):
        warm(frame, result.settings_path, depth="quick")


def test_standard_pilots_every_planned_fit_on_chronological_full_rows(tmp_path, frame, monkeypatch):
    import memetic_fs.warmup as warmup
    real_fit = warmup._fit
    seen = []
    def checked_fit(data, fold, columns, candidate, threads, seed):
        assert data.months[fold.train].max() < data.months[fold.valid].min()
        assert len(fold.train) == int((data.months <= data.months[fold.train].max()).sum())
        assert len(fold.valid) == int((data.months == data.months[fold.valid].min()).sum())
        seen.append((fold.label, len(columns)))
        return real_fit(data, fold, columns, candidate, threads, seed)
    monkeypatch.setattr(warmup, "_fit", checked_fit)
    result = warm(frame, tmp_path / "standard.yaml", depth="standard")
    assert len(seen) == result.diagnostics["planned_fits"]
    assert len(result.diagnostics["pilot_folds"]) == 3
    assert result.pilots["selected_model"]
    assert result.pilots["subset_score_gaps"]
    assert result.suggestions["resources"]["max_workers"] >= 1
    assert result.suggestions["resources"]["threads_per_worker"] >= 1


def test_deep_uses_all_folds_and_complete_optimization_config(tmp_path, frame, monkeypatch):
    import memetic_fs.warmup as warmup
    monkeypatch.setattr(warmup, "_candidates", lambda depth, n, params, rounds:
        [{"params": {"learning_rate": .1, "num_leaves": 7, "min_data_in_leaf": 5},
          "num_boost_round": 3}])
    result = warm(frame, tmp_path / "deep.yaml", depth="deep")
    assert len(result.diagnostics["pilot_folds"]) == len(result.diagnostics["folds"])
    assert len(result.pilots["model_candidates"]) == 1
    assert result.diagnostics["planned_fits"] == (
        1 + len(result.pilots["subset_scores"]) - 1) * len(result.diagnostics["folds"])
    assert result.suggestions["optimization"]["rfe_seed_sizes"] == [
        r["n_features"] for r in result.pilots["subset_scores"]]
    assert "inverse_exponent" in result.suggestions["optimization"]


def test_from_config_accepts_existing_experiment_yaml_without_creating_run(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({"data": {"path": "absent.csv", "target": "y",
        "time_column": "month"}, "output": {"experiment_dir": "run"},
        "model": {"params": {"num_leaves": 15}, "num_boost_round": 20},
        "optimization": {"population_size": 8}}), encoding="utf-8")
    selector = MemeticNSGA2Selector.from_config(path, model_params={"learning_rate": .05},
                                                optimization={"max_generations": 3})
    assert selector.time_cv == "month"
    assert selector.model_params == {"num_leaves": 15, "learning_rate": .05}
    assert selector.optimization_.population_size == 8
    assert selector.optimization_.max_generations == 3
    assert not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="Unknown selector overrides"):
        MemeticNSGA2Selector.from_config(path, nonexistent=1)
    with pytest.raises(ValueError, match="num_boost_round"):
        MemeticNSGA2Selector.from_config(path, num_boost_round=0)


def test_generated_yaml_fit_and_resume(tmp_path, frame):
    result = warm(frame, tmp_path / "quick.yaml", depth="quick")
    selector = MemeticNSGA2Selector.from_config(result.settings_path,
        checkpoint_dir=tmp_path / "run", num_boost_round=3,
        model_params={"num_leaves": 7, "min_data_in_leaf": 5},
        optimization={"population_size": 4, "max_generations": 1,
                      "reduced_generations": 0, "local_search_budget": 0},
        resources={"max_workers": 1, "threads_per_worker": 1},
        use_rfe_initialization=False, interactive_plots=False, verbose=0)
    selector.fit(frame, "target", id_col="applicant_id",
                 segment_cols=["customer_type", "age_group"])
    assert selector.best_features_
    resumed = MemeticNSGA2Selector.resume(tmp_path / "run", verbose=0)
    assert resumed.best_features_ == selector.best_features_
