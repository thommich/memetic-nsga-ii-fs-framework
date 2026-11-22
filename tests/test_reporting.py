import json
from html.parser import HTMLParser

import pandas as pd
import pytest

from memetic_fs.reporting import aggregate_run_frequencies, export_results, rebuild_report, selection_frequency
from memetic_fs.reporting.visualizations import build_figures, segment_constraint_comparisons


def result(key, mask, gini, fidelity="full"):
    folds = [{"fold": f"2026-0{i}", "gini": gini + offset, "auc": (gini + offset + 1) / 2,
              "n": 100, "events": 50, "nonevents": 50} for i, offset in [(4, -0.01), (5, 0.), (6, 0.01)]]
    return {"key": key, "mask": mask, "n_features": sum(mask), "mean_gini": gini,
            "std_gini": 0.01, "min_gini": gini - 0.01, "pooled_gini": gini,
            "folds": folds, "segments": [{"column": "customer", "value": "prospect", "fold": f["fold"],
                "gini": gini - 0.05, "auc": (gini + 0.95) / 2, "n": 100, "events": 50, "nonevents": 50} for f in folds],
            "constraints": [], "fidelity": fidelity}


def sample():
    first = result("one", [True, False, False], .40)
    second = result("two", [True, True, False], .41)
    provisional = result("cheap", [False, True, False], .8, "last_1")
    reference = result("reference", [True, True, True], .405)
    return first, second, provisional, reference


def test_selection_frequency_deduplicates_masks():
    first, second, _, _ = sample()
    table = selection_frequency([first, first, second], ["a", "b", "c"], {"a": "signal", "b": "signal"}).set_index("feature")
    assert table.loc["a", "count"] == 2
    assert table.loc["b", "frequency"] == .5
    assert table.loc["c", "frequency"] == 0
    assert table.loc["a", "family"] == "signal"
    assert table.loc["a", "n_solutions"] == 2
    assert selection_frequency([], ["a"]).iloc[0].frequency == 0


def test_exports_rebuild_and_full_fidelity_frontier(tmp_path):
    first, second, provisional, reference = sample()
    kwargs = dict(directory=tmp_path, results=[first, second, provisional, reference], feature_names=["a", "b", "c"],
                  best_features=["a", "b"], history=[{"generation": 1, "fidelity": "last_1", "best_confirmed_gini": .41,
                    "confirmed_hypervolume": .8, "unique_evaluations": 3, "confirmed_count": 2, "provisional_count": 1,
                    "feature_count_min": 1, "feature_count_mean": 1.5, "feature_count_max": 2, "converged": False}],
                  reference=reference, pareto=[first, second], rfe_trajectory=[first], feature_groups={"a": "signal", "b": "signal"},
                  segment_constraints=[{"column": "customer", "value": "prospect", "max_gini_drop": .01}])
    outputs = export_results(**kwargs)
    assert outputs["report"].exists()
    payload = json.loads((tmp_path / "results.json").read_text())
    assert len(payload["pareto"]) == 2
    assert json.loads((tmp_path / "selected_features.json").read_text()) == ["a", "b"]
    assert pd.read_parquet(tmp_path / "confirmed_masks.parquet").shape == (3, 4)
    assert pd.read_parquet(tmp_path / "evaluations.parquet").fidelity.tolist().count("last_1") == 1
    assert pd.read_parquet(tmp_path / "fold_performance.parquet").shape[0] == 12
    assert pd.read_parquet(tmp_path / "family_frequency.parquet").iloc[0].frequency == 1
    assert pd.read_parquet(tmp_path / "selection_frequency.parquet").set_index("feature").loc["b", "frequency"] == .5
    assert pd.read_parquet(tmp_path / "high_performing_frequency.parquet").set_index("feature").loc["b", "frequency"] == 1
    assert not list(tmp_path.glob("*.csv"))
    html = outputs["report"].read_text(encoding="utf-8")
    assert "plotly.js" in html
    class ScriptParser(HTMLParser):
        sources = []
        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self.sources.extend(value for key, value in attrs if key == "src")
    parser = ScriptParser()
    parser.feed(html)
    assert not parser.sources
    assert "Provisional (reduced folds)" in html
    assert "independent external OOT" in html
    outputs["report"].unlink()
    assert rebuild_report(tmp_path).exists()
    with pytest.raises(ValueError, match="full-fidelity"):
        export_results(**(kwargs | {"pareto": [provisional]}))


def test_aggregate_independent_runs_keeps_matching_masks(tmp_path):
    first, second, _, reference = sample()
    for name, front in [("one", [first]), ("two", [first, second])]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "results.json").write_text(json.dumps({"feature_names": ["a", "b", "c"], "pareto": front}))
    table = aggregate_run_frequencies([tmp_path / "one", tmp_path / "two"]).set_index("feature")
    assert table.loc["a", "count"] == 3
    assert table.loc["b", "frequency"] == 1 / 3
    assert table.loc["b", "run_frequency"] == .5
    assert table.loc["a", "run_frequency"] == 1
    assert table.loc["a", "n_solutions"] == 3


def test_figure_coselection_is_fraction_of_pareto_subsets():
    first, second, provisional, reference = sample()
    payload = dict(results=[first, second, provisional, reference], pareto=[first, second], reference=reference,
                   feature_names=["a", "b", "c"], best_features=["a"], history=[])
    figures = build_figures(payload)
    assert set(figures) == {"frontier", "cardinality", "frequency", "coselection", "temporal", "segments", "history"}
    assert figures["coselection"].data[0].z[0, 1] == .5
    assert figures["coselection"].data[0].z[0, 0] == 1


def test_empty_segment_front_and_history_exports(tmp_path):
    _, _, _, reference = sample()
    reference["segments"] = []
    result = export_results(tmp_path, [reference], ["a", "b", "c"], [], [], reference, pareto=[])
    assert result["report"].exists()
    assert pd.read_parquet(tmp_path / "segment_performance.parquet").empty
    assert not (tmp_path / "segment_performance.csv").exists()


@pytest.mark.parametrize("output_format", ["csv", "json", "parquet"])
def test_exports_one_selected_table_format(tmp_path, output_format):
    first, second, _, reference = sample()
    export_results(tmp_path, [first, second, reference], ["a", "b", "c"], ["a", "b"], [], reference,
                   pareto=[first, second], output_format=output_format, interactive_plots=False)
    for name in ("evaluations", "confirmed_masks", "fold_performance", "segment_performance",
                 "segment_constraints", "selection_frequency", "high_performing_frequency",
                 "history", "rfe_trajectory"):
        assert (tmp_path / f"{name}.{output_format}").exists()
        assert not any((tmp_path / f"{name}.{other}").exists()
                       for other in {"csv", "json", "parquet"} - {output_format})
    path = tmp_path / f"selection_frequency.{output_format}"
    if output_format == "csv":
        rows = pd.read_csv(path).to_dict("records")
    elif output_format == "json":
        rows = json.loads(path.read_text())
    else:
        rows = pd.read_parquet(path).to_dict("records")
    assert next(row for row in rows if row["feature"] == "b")["frequency"] == .5
    assert rebuild_report(tmp_path).exists()


def test_invalid_output_format(tmp_path):
    _, _, _, reference = sample()
    with pytest.raises(ValueError, match="output_format"):
        export_results(tmp_path, [reference], ["a", "b", "c"], [], [], reference, output_format="xml")


def test_reexport_removes_previous_table_format(tmp_path):
    first, second, _, reference = sample()
    args = (tmp_path, [first, second, reference], ["a", "b", "c"], ["a", "b"], [], reference)
    export_results(*args, pareto=[first, second], output_format="csv", interactive_plots=False)
    export_results(*args, pareto=[first, second], output_format="json", interactive_plots=False)
    assert (tmp_path / "evaluations.json").exists()
    assert not (tmp_path / "evaluations.csv").exists()


def test_configurable_tolerance_and_per_fold_constraint_violations(tmp_path):
    first, second, _, reference = sample()
    first["segments"][0]["gini"] = .1
    first["constraints"] = [.245]
    export_results(tmp_path, [first, second, reference], ["a", "b", "c"], ["a", "b"], [], reference,
                   pareto=[second], segment_constraints=[{"column": "customer", "value": "prospect", "max_gini_drop": .01}],
                   gini_tolerance=.02, selected_candidate_id="two")
    comparisons = pd.read_parquet(tmp_path / "segment_constraints.parquet")
    violation = comparisons.loc[(comparisons.key == "one") & (comparisons.fold == "2026-04")].iloc[0]
    assert bool(violation.violated)
    assert violation.violation == pytest.approx(.245)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "VIOLATION" in html
    assert 'class="violation"' in html
    assert "within 0.02 Gini" in html
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["selected_candidate_id"] == "two"
    assert payload["gini_tolerance"] == .02


def test_missing_segment_gini_is_undefined_in_constraint_comparison():
    first, _, _, reference = sample()
    first["segments"][0]["gini"] = None
    payload = {"results": [first], "reference": reference,
               "segment_constraints": [{"column": "customer", "value": "prospect", "max_gini_drop": .01}]}
    row = segment_constraint_comparisons(payload)[0]
    assert row["violated"] is None
    assert row["violation"] is None
