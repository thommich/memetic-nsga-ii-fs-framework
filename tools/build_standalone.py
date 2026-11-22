"""Rebuild the single-file module from the maintained package sources."""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "memetic_fs"
OUTPUT = ROOT / "memetic_fs_standalone.py"

def local(node):
    return bool(node.level or (node.module and (node.module == "memetic_fs" or node.module.startswith("memetic_fs."))))

def name(path):
    return ".".join(path.relative_to(SOURCE).with_suffix("").parts)

def dependency(path, node):
    if node.level:
        parts = name(path).split(".")[:-1]
        return ".".join(parts[:len(parts)-node.level+1] + (node.module.split(".") if node.module else []))
    return node.module.removeprefix("memetic_fs.")

def build():
    paths = {name(path): path for path in SOURCE.rglob("*.py") if path.name not in {"__init__.py", "__main__.py"}}
    graph = {}
    for key, path in paths.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        graph[key] = [dependency(path, n) for n in tree.body if isinstance(n, ast.ImportFrom) and local(n)]
        if set(graph[key]) - paths.keys():
            raise ValueError(f"Unknown dependency in {path}: {set(graph[key]) - paths.keys()}")
    order, seen, active = [], set(), set()
    def visit(key):
        if key in active:
            raise ValueError(f"Top-level import cycle: {key}")
        if key in seen:
            return
        active.add(key)
        for required in graph[key]:
            visit(required)
        active.remove(key)
        seen.add(key)
        order.append(key)
    for key in paths:
        visit(key)
    parts = ['''"""Single-file temporal memetic NSGA-II feature selection.

Copy this file beside a notebook and import it. Third-party dependencies must be
installed separately. Generated from the maintained memetic_fs package; all
runtime framework code is in this module. Experiment artifacts remain in
checkpoint_dir.
"""
from __future__ import annotations
''']
    for key in order:
        path = paths[key]
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source)
        remove = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (local(n) or n.module == "__future__")]
        for node in sorted(remove, key=lambda n: n.lineno, reverse=True):
            start, end = node.lineno - 1, node.end_lineno
            indentation = len(lines[start]) - len(lines[start].lstrip())
            lines[start:end] = [" " * indentation + "pass\n"] if indentation else ["\n"]
        parts.append(f"\n\n# === Source: memetic_fs/{path.relative_to(SOURCE).as_posix()} ===\n" + "".join(lines))
    parts.append('''
def run_selector(X, y, *, id_col=None, segment_cols=None, time_values=None,
                 ids=None, feature_names=None, **selector_options):
    """Fit and return a selector in one notebook-friendly call.

    ``selector_options`` are MemeticNSGA2Selector constructor options. For
    arrays, supply time_values and ids (or set allow_repeated_ids=True).
    """
    selector = MemeticNSGA2Selector(**selector_options)
    return selector.fit(X, y, id_col=id_col, segment_cols=segment_cols,
                        time_values=time_values, ids=ids, feature_names=feature_names)

__version__ = "0.1.0"
__all__ = ["MemeticNSGA2Selector", "run_selector", "CVConfig", "OptimizationConfig",
           "ResourceConfig", "EvaluationConfig", "SegmentConstraint", "SelectionConfig", "WarmupResult",
           "make_synthetic_data", "load_experiment", "main", "rebuild_report",
           "aggregate_run_frequencies"]

if __name__ == "__main__":
    raise SystemExit(main())
''')
    OUTPUT.write_text("".join(parts), encoding="utf-8")
    compile(OUTPUT.read_text(encoding="utf-8"), str(OUTPUT), "exec")
    return OUTPUT

if __name__ == "__main__":
    output = build()
    print(f"Built {output} ({output.stat().st_size:,} bytes)")
