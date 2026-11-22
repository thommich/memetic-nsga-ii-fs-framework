"""Exercise the deliverable from a directory containing only its one module."""
from pathlib import Path
import shutil
import subprocess
import sys


def test_single_file_parallel_fit_report_and_resume(tmp_path):
    source = Path(__file__).resolve().parents[1] / "memetic_fs_standalone.py"
    shutil.copyfile(source, tmp_path / source.name)
    script = '''from pathlib import Path
import sys
import memetic_fs_standalone as fs
frame = fs.make_synthetic_data(n_rows=240, n_features=8, n_months=6, random_state=31)
selector = fs.run_selector(frame, "target", time_cv="reference_month",
    id_col="applicant_id", segment_cols=["customer_type", "age_group"],
    feature_range=(1, 5), use_rfe_initialization=True, num_boost_round=3,
    model_params={"min_data_in_leaf": 5, "num_leaves": 7},
    optimization=fs.OptimizationConfig(population_size=4, max_generations=2,
        local_search_interval=2, local_search_budget=2, reduced_generations=1,
        rfe_seed_sizes=(2, 4)),
    resources=fs.ResourceConfig(max_workers=2, threads_per_worker=1),
    checkpoint_dir="run", verbose=0)
assert selector.best_features_
assert all(r["fidelity"] == "full" for r in selector.confirmed_)
assert Path("run/report.html").is_file()
assert Path("run/checkpoint.json").is_file()
assert Path("run/logs/run.log").is_file()
assert list(Path("run/logs").glob("worker-*.log"))
resumed = fs.MemeticNSGA2Selector.resume("run", verbose=0)
assert resumed.best_features_ == selector.best_features_
assert "memetic_fs" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_single_file_matches_generated_package_sources():
    from tools.build_standalone import build, OUTPUT
    before = OUTPUT.read_bytes()
    build()
    assert OUTPUT.read_bytes() == before


def test_single_file_warmup_and_config_without_package(tmp_path):
    source = Path(__file__).resolve().parents[1] / "memetic_fs_standalone.py"
    shutil.copyfile(source, tmp_path / source.name)
    script = '''from pathlib import Path
import sys
import memetic_fs_standalone as fs
frame = fs.make_synthetic_data(n_rows=240, n_features=8, n_months=6)
result = fs.MemeticNSGA2Selector.warmup_settings_params(
    frame, "target", depth="quick", output_path="warmup.yaml",
    id_col="applicant_id", segment_cols=["customer_type", "age_group"])
selector = fs.MemeticNSGA2Selector.from_config(result.settings_path)
assert selector.feature_range == [1, 8]
assert Path("warmup.yaml").is_file()
assert "memetic_fs" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
