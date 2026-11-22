"""Read selector settings without initializing an experiment."""
from __future__ import annotations

import inspect
from pathlib import Path

import yaml


def load_selector_settings(path: str | Path) -> dict:
    """Accept flat selector YAML or the existing CLI experiment YAML."""
    source = Path(path).resolve()
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Selector YAML must be a mapping")
    if "data" in document or "model" in document or "output" in document:
        from .cli import load_experiment
        _, settings = load_experiment(source)
        return settings
    from .api import MemeticNSGA2Selector
    allowed = set(inspect.signature(MemeticNSGA2Selector).parameters)
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"Unknown selector YAML keys: {sorted(unknown)}")
    if "checkpoint_dir" in document:
        directory = Path(document["checkpoint_dir"])
        if not directory.is_absolute():
            document["checkpoint_dir"] = str((source.parent / directory).resolve())
    return document
