"""Standardized metrics exports; no estimator objects or predictions are persisted."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _clean(value: Any) -> Any:
    if is_dataclass(value):
        return _clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if k not in {"predictions", "oof_predictions", "models"}}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp, Path)):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _unique(results: Sequence[dict]) -> list[dict]:
    return list({tuple(result["mask"]): result for result in results}.values())


def _families(feature_names: Sequence[str], feature_groups: dict | None) -> dict[str, str]:
    if not feature_groups:
        return {}
    if all(name in feature_names for name in feature_groups):
        return {name: str(family) for name, family in feature_groups.items()}
    result = {}
    for family, features in feature_groups.items():
        for feature in features:
            name = feature_names[feature] if isinstance(feature, int) else feature
            if name not in feature_names:
                raise ValueError(f"Unknown grouped feature {name!r}")
            result[name] = str(family)
    return result


def selection_frequency(
    results: Sequence[dict], feature_names: Sequence[str], feature_groups: dict | None = None,
) -> pd.DataFrame:
    """Count inclusion across distinct supplied subsets, without score weighting.

    Supply a confirmed frontier or a chosen high-performing solution family.
    Frequencies describe the searched solutions and are not feature significance.
    """
    results = _unique(results)
    masks = np.asarray([result["mask"] for result in results], dtype=bool)
    if results and (masks.ndim != 2 or masks.shape[1] != len(feature_names)):
        raise ValueError("Frequency masks must match feature_names")
    counts = masks.sum(axis=0) if len(results) else np.zeros(len(feature_names), dtype=int)
    table = pd.DataFrame({"feature": list(feature_names), "count": counts,
                          "frequency": counts / len(results) if results else np.zeros(len(feature_names)),
                          "n_solutions": len(results)})
    families = _families(feature_names, feature_groups)
    if families:
        table["family"] = table.feature.map(families).fillna("ungrouped")
    return table.sort_values(["frequency", "feature"], ascending=[False, True]).reset_index(drop=True)


def aggregate_run_frequencies(directories: Sequence[str | Path]) -> pd.DataFrame:
    """Aggregate confirmed Pareto frequencies across independent saved runs.

    ``run_frequency`` is the fraction of runs in which a feature appears in at
    least one Pareto subset. ``frequency`` pools all subsets without collapsing
    identical masks from different runs. Runs must use the same predictor names.
    """
    if not directories:
        raise ValueError("At least one experiment directory is required")
    names = None
    counts, run_counts, total = None, None, 0
    for directory in directories:
        payload = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
        if names is None:
            names = payload["feature_names"]
            counts = np.zeros(len(names), dtype=int)
            run_counts = np.zeros(len(names), dtype=int)
        elif names != payload["feature_names"]:
            raise ValueError("Independent runs must use identical predictor names and order")
        front = _unique(payload["pareto"])
        if front:
            masks = np.array([r["mask"] for r in front], dtype=bool)
            counts += masks.sum(axis=0)
            run_counts += masks.any(axis=0)
            total += len(front)
    return pd.DataFrame({"feature": names, "count": counts,
                         "frequency": counts / total if total else np.zeros(len(names)),
                         "run_count": run_counts, "run_frequency": run_counts / len(directories),
                         "n_solutions": total, "n_runs": len(directories)}).sort_values(
                             ["frequency", "feature"], ascending=[False, True]).reset_index(drop=True)


def _table(directory: Path, name: str, rows: list | pd.DataFrame, output_format: str) -> pd.DataFrame:
    empty_columns = {
        "evaluations": ["key", "fidelity", "n_features", "mean_gini", "std_gini", "min_gini", "pooled_gini", "pareto", "feasible", "max_constraint_violation"],
        "confirmed_masks": ["key"],
        "fold_performance": ["key", "fidelity", "fold", "auc", "gini", "n", "events", "nonevents"],
        "segment_performance": ["key", "fidelity", "column", "value", "fold", "auc", "gini", "n", "events", "nonevents"],
        "history": ["generation", "fidelity", "confirmed_hypervolume", "best_confirmed_gini", "unique_evaluations", "converged"],
        "rfe_trajectory": ["key", "n_features", "mean_gini", "std_gini"],
        "segment_constraints": ["key", "fidelity", "column", "value", "fold", "candidate_gini", "reference_gini", "minimum_gini", "violation", "violated", "n", "events", "nonevents"],
    }
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows if rows else None, columns=None if rows else empty_columns.get(name))
    path = directory / f"{name}.{output_format}"
    if output_format == "csv":
        frame.to_csv(path, index=False)
    elif output_format == "json":
        _json(path, frame.to_dict(orient="records"))
    else:
        frame.to_parquet(path, index=False)
    for other in {"csv", "json", "parquet"} - {output_format}:
        (directory / f"{name}.{other}").unlink(missing_ok=True)
    return frame


def export_results(
    directory: str | Path,
    results: Sequence[dict],
    feature_names: Sequence[str],
    best_features: Sequence[str],
    history: Sequence[dict],
    reference: dict,
    rfe_trajectory: Sequence[dict] | None = None,
    feature_groups: dict | None = None,
    pareto: Sequence[dict] | None = None,
    segment_constraints: Sequence[dict] | None = None,
    interactive_plots: bool = True,
    gini_tolerance: float = 0.005,
    selected_candidate_id: str | None = None,
    output_format: str = "parquet",
) -> dict:
    """Export reproducible metrics tables and an interactive offline HTML report.

    All metrics are development validation metrics. The caller supplies the
    canonical confirmed frontier; otherwise pymoo constructs one from full-CV
    feasible results. The full-feature reference is shown separately. The
    function returns exported paths and the Pareto selection-frequency table.
    """
    from ..optimization.archives import pareto_front
    from .visualizations import segment_constraint_comparisons, write_report

    if not isinstance(gini_tolerance, (int, float)) or not math.isfinite(gini_tolerance) or gini_tolerance < 0:
        raise ValueError("Reporting gini_tolerance must be finite and nonnegative on the Gini coefficient scale")
    if not isinstance(output_format, str) or output_format not in {"csv", "json", "parquet"}:
        raise ValueError("output_format must be 'csv', 'json', or 'parquet'")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = _clean(list(results))
    front = _clean(list(pareto)) if pareto is not None else pareto_front(results)
    if any(r.get("fidelity") != "full" for r in front):
        raise ValueError("Exported confirmed Pareto frontier must contain full-fidelity results only")
    if any(any(c > 0 for c in r.get("constraints", [])) or r.get("mean_gini") is None for r in front):
        raise ValueError("Exported confirmed Pareto frontier must contain feasible, defined metrics only")
    payload = {"schema": 1, "feature_names": list(feature_names), "best_features": list(best_features),
               "results": results, "pareto": front, "history": _clean(list(history)),
               "reference": _clean(reference), "rfe_trajectory": _clean(list(rfe_trajectory or [])),
               "feature_groups": _clean(feature_groups), "segment_constraints": _clean(list(segment_constraints or [])),
               "gini_tolerance": gini_tolerance, "selected_candidate_id": selected_candidate_id,
               "output_format": output_format}
    _json(directory / "results.json", payload)
    _json(directory / "selected_features.json", list(best_features))
    _json(directory / "pareto.json", [{**r, "features": [f for f, flag in zip(feature_names, r["mask"]) if flag]} for r in front])
    _json(directory / "feature_masks.json", [{"key": r["key"], "fidelity": r["fidelity"], "mask": r["mask"]} for r in results])
    confirmed = _unique([r for r in results if r["fidelity"] == "full"])
    keys = {r["key"] for r in front}
    summary = [{k: r.get(k) for k in ["key", "fidelity", "n_features", "mean_gini", "std_gini", "min_gini", "pooled_gini"]} | {
                "pareto": r["key"] in keys, "feasible": all(c <= 0 for c in r.get("constraints", [])),
                "max_constraint_violation": max([0.] + r.get("constraints", []))} for r in results]
    _table(directory, "evaluations", summary, output_format)
    _table(directory, "confirmed_masks", [{"key": r["key"], **{f"mask::{f}": flag for f, flag in zip(feature_names, r["mask"])}} for r in confirmed], output_format)
    _table(directory, "fold_performance", [{"key": r["key"], "fidelity": r["fidelity"], **f} for r in results for f in r["folds"]], output_format)
    _table(directory, "segment_performance", [{"key": r["key"], "fidelity": r["fidelity"], **s} for r in results for s in r["segments"]], output_format)
    _table(directory, "segment_constraints", segment_constraint_comparisons(payload), output_format)
    frequency = selection_frequency(front, feature_names, feature_groups)
    _table(directory, "selection_frequency", frequency, output_format)
    high = [r for r in front if r["mean_gini"] >= max((v["mean_gini"] for v in front), default=0) - gini_tolerance]
    _table(directory, "high_performing_frequency", selection_frequency(high, feature_names, feature_groups), output_format)
    if feature_groups:
        family_map = _families(feature_names, feature_groups)
        families = sorted(set(family_map.values()))
        rows = []
        for family in families:
            indices = [i for i, name in enumerate(feature_names) if family_map.get(name) == family]
            count = sum(any(r["mask"][i] for i in indices) for r in front)
            rows.append({"family": family, "count": count, "frequency": count / len(front) if front else 0., "n_solutions": len(front)})
        _table(directory, "family_frequency", rows, output_format)
    _table(directory, "history", list(history), output_format)
    _table(directory, "rfe_trajectory", [{k: r.get(k) for k in ["key", "n_features", "mean_gini", "std_gini"]} for r in (rfe_trajectory or [])], output_format)
    if interactive_plots:
        write_report(directory / "report.html", payload)
    return {"directory": directory.resolve(), "report": (directory / "report.html").resolve(),
            "frequency": frequency, "results": (directory / "results.json").resolve()}


def rebuild_report(directory: str | Path) -> Path:
    """Rebuild offline plots from saved metrics without loading data or models."""
    from .visualizations import write_report
    directory = Path(directory)
    payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise ValueError("Unsupported results export schema")
    path = directory / "report.html"
    write_report(path, payload)
    return path.resolve()
