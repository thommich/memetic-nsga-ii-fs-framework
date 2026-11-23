"""Observable progress output, durable detail, verbosity and heartbeat behavior."""
import io
import logging
import os
import threading
import pytest
from memetic_fs.progress import ProgressReporter, progress_from_descriptor, progress_activity, NULL_PROGRESS

def test_console_verbosity_and_detailed_file_output(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    reporter = ProgressReporter(tmp_path, verbose=1)
    reporter.milestone("Generation completed", generation=3, best_confirmed_gini=.71)
    reporter.info("Evaluation batch prepared", requested_candidates=8)
    reporter.debug("Temporal fold started", train_n=1000, validation_n=200)
    console = capsys.readouterr().err
    assert "Generation completed" in console and "generation=3" in console
    assert "Temporal fold started" not in console
    assert "Evaluation batch prepared" not in console
    output = reporter.path.read_text(encoding="utf-8")
    assert "Generation completed" in output and "Temporal fold started" in output
    assert "Evaluation batch prepared" in output
    assert "train_n=1000" in output and "validation_n=200" in output
    assert "Z INFO pid=" in output
    reporter.close()
    reporter.milestone("Resume completed")
    reporter.close()
    assert "Resume completed" in reporter.path.read_text(encoding="utf-8")

def test_quiet_mode_keeps_warnings_and_detailed_logs(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    reporter = ProgressReporter(tmp_path, verbose=0)
    reporter.info("Cached result reused")
    reporter.warning("Memory pressure detected")
    console = capsys.readouterr().err
    assert "Cached result" not in console and "Memory pressure" in console
    reporter.close()
    assert "Cached result" in reporter.path.read_text(encoding="utf-8")

def test_existing_application_logging_receives_structured_records_without_duplicate_console(tmp_path, capsys, monkeypatch):
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
    monkeypatch.setattr(logging.getLogger(), "handlers", [Capture()])
    monkeypatch.setattr(logging.getLogger(), "level", logging.INFO)
    reporter = ProgressReporter(tmp_path, verbose=1)
    reporter.milestone("Model batch complete", cache_hits=4, unique=7)
    reporter.info("Routine batch detail", tasks=7)
    reporter.debug("Detailed internal state")
    assert len(records) == 1
    assert records[0].progress_fields == {"cache_hits":4, "unique":7}
    assert "cache_hits=4" in records[0].getMessage()
    assert not capsys.readouterr().err
    assert "Routine batch detail" in reporter.path.read_text(encoding="utf-8")
    reporter.close()


def test_generation_bar_updates_worker_counts_and_closes(tmp_path, monkeypatch):
    from memetic_fs import progress as progress_module

    bars = []
    class Bar:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.n = kwargs["initial"]
            self.postfix = None
            self.closed = False
            bars.append(self)
        def set_description_str(self, value):
            self.description = value
        def set_postfix(self, value, refresh=True):
            self.postfix = value.copy()
        def update(self, amount):
            self.n += amount
        def close(self):
            self.closed = True

    monkeypatch.setattr(progress_module, "tqdm", Bar)
    reporter = ProgressReporter(tmp_path, verbose=1)
    reporter.start_generations(5, completed=2)
    reporter.generation(3)
    reporter.worker_status(total=4, running=2, completed=1, failed=1)
    reporter.generation_completed(3)
    reporter.finish_generations()
    assert bars[0].description == "Running generation 3/5"
    assert bars[0].n == 3
    assert bars[0].postfix == {"workers_total": 4, "running": 2, "completed": 1, "failed": 1}
    assert bars[0].closed
    reporter.close()

def test_descriptors_reconstruct_worker_specific_file_and_validate_verbosity(tmp_path):
    reporter = progress_from_descriptor({"directory":str(tmp_path), "verbose":0, "parent_pid":os.getpid()+1})
    reporter.info("Worker task complete")
    reporter.close()
    assert reporter.path.name == f"worker-{os.getpid()}.log"
    assert "Worker task complete" in reporter.path.read_text(encoding="utf-8")
    assert progress_from_descriptor(None) is NULL_PROGRESS
    for value in (-1,3,True,"info"):
        with pytest.raises(ValueError, match="verbose"):
            ProgressReporter(tmp_path, verbose=value)

def test_long_activity_emits_heartbeat_and_stops_on_error():
    events, received = [], threading.Event()
    class Reporter:
        def info(self, message, **fields):
            events.append((message, fields))
            received.set()
    with pytest.raises(RuntimeError):
        with progress_activity(Reporter(), "Fit still running", interval_seconds=.01, phase="model"):
            assert received.wait(1)
            raise RuntimeError("interrupted")
    assert events and events[0][1]["phase"] == "model"
    assert events[0][1]["elapsed_seconds"] >= 0
    assert not any(t.name == "memetic-fs-progress" for t in threading.enumerate())

def test_api_progress_covers_pipeline_and_persists_across_quiet_resume(tmp_path):
    from memetic_fs import MemeticNSGA2Selector, OptimizationConfig, ResourceConfig
    from memetic_fs.synthetic import make_synthetic_data
    frame = make_synthetic_data(n_rows=240, n_features=8, n_months=6)
    selector = MemeticNSGA2Selector(checkpoint_dir=tmp_path, feature_range=(2,4), verbose=0,
        resources=ResourceConfig(max_workers=1, threads_per_worker=1), num_boost_round=2,
        optimization=OptimizationConfig(population_size=4, max_generations=2,
            rfe_seed_sizes=(2,4), local_search_interval=2, local_search_budget=2,
            reduced_generations=1))
    selector.fit(frame, "target", id_col="applicant_id", segment_cols=["customer_type","age_group"])
    output = selector.log_path_.read_text(encoding="utf-8")
    for marker in ("Development data validated", "Temporal folds validated", "reference calibration",
                   "RFE", "Generation completed", "Local search", "Evaluation batch", "cache_hits",
                   "Checkpoint saved", "Final candidate selected", "Result artifacts saved", "Feature selection completed"):
        assert marker in output
    resumed = MemeticNSGA2Selector.resume(tmp_path, verbose=0)
    resumed.close()
    assert "Resuming feature subset search" in selector.log_path_.read_text(encoding="utf-8")
    assert resumed.best_features_ == selector.best_features_
