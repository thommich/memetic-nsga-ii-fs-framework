"""YAML-driven experiment execution; no executable Python is loaded from YAML."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import logging
import pandas as pd
import yaml
from .api import MemeticNSGA2Selector

def _check(values: dict, allowed: set[str], label: str):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")

def load_experiment(path: str | Path) -> tuple[dict, dict]:
    """Read strict YAML and map it onto the public Python constructor."""
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    _check(config, {"data", "model", "cross_validation", "optimization", "evaluation", "resources", "output", "selection", "segment_constraints", "initial_subsets", "feature_groups"}, "root")
    for key in ("data", "output"):
        if key not in config:
            raise ValueError(f"Missing YAML section {key}")
    data = dict(config["data"])
    _check(data, {"path", "target", "time_column", "id_column", "allow_repeated_ids", "segment_columns"}, "data")
    for key in ("path", "target", "time_column"):
        if key not in data:
            raise ValueError(f"Missing data.{key}")
    data["path"] = str((source.parent / data["path"]).resolve())
    model = dict(config.get("model", {}))
    _check(model, {"estimator", "params", "num_boost_round"}, "model")
    if model.get("estimator", "lightgbm") != "lightgbm":
        raise ValueError("model.estimator must be lightgbm")
    cv = dict(config.get("cross_validation", {}))
    if cv.pop("strategy", "expanding") != "expanding":
        raise ValueError("Only expanding auto CV is supported; use Python callable API for custom windows")
    optimization = dict(config.get("optimization", {}))
    if optimization.pop("algorithm", "memetic_nsga2") != "memetic_nsga2":
        raise ValueError("optimization.algorithm must be memetic_nsga2")
    special = {key: optimization.pop(key) for key in ("feature_range", "random_state", "use_rfe_initialization") if key in optimization}
    evaluation = dict(config.get("evaluation", {}))
    if evaluation.pop("metric", "gini") != "gini":
        raise ValueError("evaluation.metric must be gini")
    for name in ("row_sampling", "feature_sampling", "use_sample_weights"):
        if evaluation.pop(name, False) is not False:
            raise ValueError(f"evaluation.{name} must be false")
    reduced = evaluation.pop("allow_reduced_fold_fidelity", True)
    if not isinstance(reduced, bool):
        raise ValueError("allow_reduced_fold_fidelity must be boolean")
    if not reduced:
        optimization["reduced_generations"] = 0
    resources = dict(config.get("resources", {}))
    if resources.pop("mode", "auto") != "auto":
        raise ValueError("resources.mode must be auto; apply limits with max_workers/max_ram_gb/threads_per_worker")
    output = dict(config["output"])
    _check(output, {"experiment_dir", "checkpoints", "interactive_plots", "verbose", "format"}, "output")
    if "experiment_dir" not in output:
        raise ValueError("Missing output.experiment_dir")
    if output.get("checkpoints", True) is not True:
        raise ValueError("Durable checkpoints are required and cannot be disabled")
    kwargs = {"model_params": model.get("params"), "num_boost_round": model.get("num_boost_round", 100),
              "time_cv": data["time_column"], "allow_repeated_ids": data.get("allow_repeated_ids", False),
              "cv_config": cv, "optimization": optimization, "evaluation": evaluation,
              "resources": resources, "selection": config.get("selection"),
              "segment_constraints": config.get("segment_constraints"),
              "initial_subsets": config.get("initial_subsets"), "feature_groups": config.get("feature_groups"),
              "checkpoint_dir": str((source.parent / output["experiment_dir"]).resolve()),
              "interactive_plots": output.get("interactive_plots", True),
              "verbose": output.get("verbose", 1),
              "output_format": output.get("format", "parquet"), **special}
    # Validate nested settings before loading a potentially huge dataset.
    MemeticNSGA2Selector(**kwargs)._validate_options()
    return data, kwargs

def main(argv: list[str] | None = None) -> int:
    """Run, resume or rebuild an experiment report."""
    parser = argparse.ArgumentParser(prog="memetic-fs")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", required=True)
    commands = [run_parser]
    for command in ("resume", "report"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--experiment-dir", required=True)
        commands.append(command_parser)
    for command_parser in commands:
        verbosity = command_parser.add_mutually_exclusive_group()
        verbosity.add_argument("-v", "--verbose", action="store_true", help="Detailed candidate, fold and cache progress")
        verbosity.add_argument("-q", "--quiet", action="store_true", help="Warnings/errors only; retain detailed files")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            data, kwargs = load_experiment(args.config)
            if args.verbose or args.quiet:
                kwargs["verbose"] = 2 if args.verbose else 0
            path = Path(data["path"])
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
            elif path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
            else:
                raise ValueError("CLI data.path must point to .parquet or .csv; use Python API for arrays")
            selector = MemeticNSGA2Selector(**kwargs).fit(frame, data["target"], id_col=data.get("id_column"), segment_cols=data.get("segment_columns"))
            print(f"Completed: {selector.directory_}; selected {len(selector.best_features_ or [])} predictors; {selector.stop_reason_}")
        elif args.command == "resume":
            selector = MemeticNSGA2Selector.resume(args.experiment_dir,
                verbose=2 if args.verbose else (0 if args.quiet else None))
            print(f"Resumed: {selector.directory_}; {selector.stop_reason_}")
        else:
            from .reporting.exports import rebuild_report
            from .progress import ProgressReporter
            progress = ProgressReporter(args.experiment_dir, verbose=2 if args.verbose else (0 if args.quiet else 1))
            progress.info("Rebuilding interactive report from saved metrics", phase="reporting")
            rebuild_report(args.experiment_dir)
            progress.info("Interactive report saved", phase="reporting")
            print(f"Report: {Path(args.experiment_dir).resolve() / 'report.html'}")
        return 0
    except (ValueError, FileNotFoundError, MemoryError) as error:
        parser.exit(2, f"memetic-fs: {error}\n")
    except (KeyboardInterrupt, InterruptedError):
        print("Cancelled; completed evaluations are cached. Use memetic-fs resume.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
