"""Configuration typos and transparent model-selection behavior."""
import numpy as np
import pytest
import yaml
from memetic_fs.config import (CVConfig, OptimizationConfig, ResourceConfig,
                               SelectionConfig, SegmentConstraint, from_dict)
from memetic_fs.selection import select_candidate
from memetic_fs.cli import load_experiment

def candidate(key, n, values, constraints=None, fidelity="full"):
    return {"key": key, "n_features": n, "mean_gini": np.mean(values),
            "folds": [{"fold": str(i), "gini": value} for i, value in enumerate(values)],
            "constraints": constraints or [], "fidelity": fidelity}

def test_tolerance_is_absolute_gini_scale_and_respects_constraints():
    front = [candidate("small", 2, [.7, .7, .7]), candidate("best", 7, [.704, .704, .704]),
             candidate("invalid", 1, [.9, .9, .9], [0.1])]
    assert select_candidate(front, SelectionConfig())["key"] == "small"
    assert select_candidate(front, SelectionConfig(gini_tolerance=.001))["key"] == "best"
    with pytest.raises(ValueError, match="No feasible"):
        select_candidate([front[2]], SelectionConfig())

def test_manual_and_knee_and_temporal_paired_uncertainty():
    front = [candidate("small", 1, [.1]*6), candidate("knee", 3, [.65]*6),
             candidate("near", 4, [.704]*6), candidate("best", 8, [.707]*6)]
    assert select_candidate(front, SelectionConfig(policy="knee"))["key"] == "knee"
    assert select_candidate(front, SelectionConfig(policy="statistical"))["key"] == "near"
    assert select_candidate(front, SelectionConfig(policy="manual"), candidate_id="best")["key"] == "best"
    with pytest.raises(ValueError, match="candidate_id"):
        select_candidate(front, SelectionConfig(policy="manual"))
    with pytest.raises(ValueError, match="not a feasible"):
        select_candidate(front, SelectionConfig(policy="manual"), candidate_id="missing")

@pytest.mark.parametrize("factory,kwargs", [(CVConfig,{"min_train_months":2}),
    (CVConfig,{"min_total_months":5}), (CVConfig,{"validation_months":2,"step_months":1}),
    (CVConfig,{"gap_months":1}), (OptimizationConfig,{"population_size":1}),
    (OptimizationConfig,{"uniform_cardinality_probability":0}),
    (ResourceConfig,{"max_workers":0}), (SelectionConfig,{"gini_tolerance":-1}),
    (SegmentConstraint,{"column":"group","value":"a","reference":"unknown"})])
def test_invalid_typed_config(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)

def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        from_dict(OptimizationConfig, {"populaton_size":12})

@pytest.mark.parametrize("evaluation", [{"row_sampling":True},{"feature_sampling":True},
                                        {"use_sample_weights":True},{"metric":"accuracy"}])
def test_cli_forbids_incompatible_evaluation_before_loading_data(tmp_path,evaluation):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"data":{"path":"absent.parquet","target":"y","time_column":"t"},
                                   "output":{"experiment_dir":"run"}, "evaluation":evaluation}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment(path)

def test_yaml_rejects_unknown_nested_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"data":{"path":"absent.parquet","target":"y","time_column":"t"},
                                   "output":{"experiment_dir":"run"}, "resources":{"max_worker":2}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown ResourceConfig"):
        load_experiment(path)


@pytest.mark.parametrize("output_format", ["csv", "json", "parquet"])
def test_yaml_output_format(tmp_path, output_format):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"data": {"path": "absent.parquet", "target": "y", "time_column": "t"},
                                    "output": {"experiment_dir": "run", "format": output_format}}), encoding="utf-8")
    _, kwargs = load_experiment(path)
    assert kwargs["output_format"] == output_format
