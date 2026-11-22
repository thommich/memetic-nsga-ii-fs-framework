from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from memetic_fs.execution import MemoryMonitor, ResourceManager, ResourceScheduler, ResourceSnapshot, detect_resources
from memetic_fs.execution import resources
from memetic_fs.persistence import EvaluationCache, atomic_json, read_json, stable_hash


def config(**overrides):
    values = dict(memory_reserve_fraction=0.25, max_ram_gb=None, max_workers=4,
                  threads_per_worker=1, max_retries=2, memory_wait_seconds=0)
    values.update(overrides)
    return SimpleNamespace(**values)


def snapshot():
    return ResourceSnapshot(cpus=8, capacity_bytes=8 * 1024**3, available_bytes=6 * 1024**3,
                            private_bytes=1024**3, temporary_free_bytes=20 * 1024**3)


def square(value):
    return value * value


def mmap_task(path):
    data = np.load(path, mmap_mode="r")
    return {"writable": data.flags.writeable, "sum": int(data.sum())}


def crash_once_task(task):
    cache = EvaluationCache(task["cache"])
    cached = cache.get(task["key"])
    if cached:
        return cached
    marker = Path(task["marker"])
    if task["key"] == "crash" and not marker.exists():
        deadline = time.monotonic() + 5
        while cache.get("success") is None and time.monotonic() < deadline:
            time.sleep(.01)
        marker.write_text("crashed once")
        os._exit(23)
    value = {"key": task["key"]}
    cache.put(task["key"], value)
    return value


def test_atomic_checkpoint_numpy_and_hash(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"generation": np.int64(3), "population": np.array([[True, False]])})
    assert read_json(path) == {"generation": 3, "population": [[True, False]]}
    atomic_json(path, {"generation": 4})
    assert read_json(path)["generation"] == 4
    assert not list(tmp_path.glob("*.tmp"))
    assert stable_hash({"b": 2, "a": np.int64(1)}) == stable_hash({"a": 1, "b": 2})
    assert stable_hash({"fidelity": [1]}) != stable_hash({"fidelity": [0, 1]})
    assert stable_hash(float("nan")) != stable_hash(None)


def test_cache_lifecycle_and_restart(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache = EvaluationCache(path)
    cache.pending("a")
    assert cache.get("a") is None
    cache.failed("a", "worker terminated")
    assert cache.incomplete()[0]["attempts"] == 1
    cache.pending("a")
    assert cache.incomplete()[0]["attempts"] == 2
    cache.put("a", {"gini": 0.6})
    cache.pending("a")
    cache.failed("a", "late error")
    assert EvaluationCache(path).get("a") == {"gini": 0.6}
    assert cache.incomplete() == []
    assert cache.completed() == [{"gini": 0.6}]


def test_cache_parallel_transactions(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache = EvaluationCache(path)
    def write(i):
        writer = EvaluationCache(path)
        writer.pending(str(i))
        writer.put(str(i), {"i": i})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(20)))
    assert len(cache.completed()) == 20


def test_resource_detection_cgroup_v2(tmp_path, monkeypatch):
    (tmp_path / "memory.max").write_text(str(4 * 1024**3))
    (tmp_path / "memory.current").write_text(str(3 * 1024**3))
    (tmp_path / "cpu.max").write_text("250000 100000")
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 1024**3, available=12 * 1024**3))
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 16)
    measured = detect_resources(tmp_path, cgroup_root=tmp_path)
    assert measured.capacity_bytes == 4 * 1024**3
    assert measured.available_bytes == 1024**3
    assert measured.cpus <= 2
    assert measured.temporary_free_bytes > 0


def test_resource_detection_cgroup_v1(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    cpu = tmp_path / "cpu"
    memory.mkdir()
    cpu.mkdir()
    (memory / "memory.limit_in_bytes").write_text(str(2 * 1024**3))
    (memory / "memory.usage_in_bytes").write_text(str(1024**3))
    (cpu / "cpu.cfs_quota_us").write_text("100000")
    (cpu / "cpu.cfs_period_us").write_text("100000")
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 1024**3, available=12 * 1024**3))
    measured = detect_resources(tmp_path, cgroup_root=tmp_path)
    assert measured.capacity_bytes == 2 * 1024**3
    assert measured.available_bytes == 1024**3
    assert measured.cpus == 1


def test_combined_cpu_cgroup_mount(tmp_path, monkeypatch):
    cpu = tmp_path / "cpu,cpuacct"
    cpu.mkdir()
    (cpu / "cpu.cfs_quota_us").write_text("100000")
    (cpu / "cpu.cfs_period_us").write_text("100000")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 8)
    assert detect_resources(tmp_path, cgroup_root=tmp_path).cpus == 1


def test_ram_constrained_plan_and_calibration(monkeypatch):
    monkeypatch.setattr(resources, "detect_resources", lambda _: snapshot())
    manager = ResourceManager(config(), estimated_task_bytes=2 * 1024**3)
    plan = manager.plan()
    assert plan.workers == 2
    assert plan.workers * plan.threads_per_worker <= plan.snapshot.cpus
    manager.calibrate(4 * 1024**3)
    assert manager.plan().workers == 0
    manager = ResourceManager(config(max_ram_gb=3), estimated_task_bytes=1024**3)
    assert manager.plan().workers == 1


def test_worker_backoff_keeps_estimator_thread_budget(monkeypatch):
    monkeypatch.setattr(resources, "detect_resources", lambda _: snapshot())
    manager = ResourceManager(config(threads_per_worker=None), estimated_task_bytes=100)
    before = manager.plan()
    manager.backoff(before.workers)
    after = manager.plan()
    assert before.threads_per_worker == after.threads_per_worker
    assert after.workers < before.workers


def test_bound_thread_budget_rejects_cpu_capacity_drop(monkeypatch):
    monkeypatch.setattr(resources, "detect_resources", lambda _: snapshot())
    manager = ResourceManager(config(threads_per_worker=4), estimated_task_bytes=100)
    manager.bound_threads = manager.plan().threads_per_worker
    monkeypatch.setattr(resources, "detect_resources", lambda _: replace(snapshot(), cpus=2))
    with pytest.raises(RuntimeError, match="fixed LightGBM thread budget"):
        manager.plan()


def test_scheduler_parallel_matches_sequential(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "detect_resources", lambda _: snapshot())
    sequential = ResourceScheduler(ResourceManager(config(max_workers=1), 100, tmp_path))
    parallel = ResourceScheduler(ResourceManager(config(max_workers=2), 100, tmp_path))
    assert parallel.map(square, range(6)) == sequential.map(square, range(6)) == [0, 1, 4, 9, 16, 25]


def test_shared_data_remains_readonly(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "detect_resources", lambda _: snapshot())
    path = tmp_path / "shared.npy"
    np.save(path, np.arange(50, dtype=np.float32))
    scheduler = ResourceScheduler(ResourceManager(config(max_workers=2), 100, tmp_path))
    assert scheduler.map(mmap_task, [path, path]) == [{"writable": False, "sum": 1225}] * 2


def test_retry_memory_failure_and_preserve_previous_results(monkeypatch):
    monkeypatch.setattr(resources, "detect_resources", lambda _: snapshot())
    calls = []
    def flaky(value):
        calls.append(value)
        if value == 1 and calls.count(1) == 1:
            raise MemoryError("allocation")
        return value
    manager = ResourceManager(config(max_workers=1), 100)
    assert ResourceScheduler(manager).map(flaky, [0, 1, 2]) == [0, 1, 2]
    assert calls == [0, 1, 1, 2]
    assert manager.estimated_task_bytes > 100


def test_memory_pressure_backoff_and_cancellation(monkeypatch):
    monkeypatch.setattr(resources, "detect_resources", lambda _: replace(snapshot(), available_bytes=0))
    manager = ResourceManager(config(), 1024)
    assert manager.plan().workers == 0
    with pytest.raises(MemoryError, match="observations will not be sampled"):
        ResourceScheduler(manager).map(square, [1])
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        ResourceScheduler(manager, cancelled=lambda: True).map(square, [1])


def test_worker_failure_exhausts_retries(monkeypatch):
    monkeypatch.setattr(resources, "detect_resources", lambda _: snapshot())
    from joblib.externals.loky.process_executor import TerminatedWorkerError
    def dies(value):
        raise TerminatedWorkerError("terminated")
    with pytest.raises(RuntimeError, match="completed cached results"):
        ResourceScheduler(ResourceManager(config(max_workers=1, max_retries=1), 100)).map(dies, [1])


def test_actual_worker_crash_preserves_completed_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "detect_resources", lambda _: snapshot())
    path = tmp_path / "cache.sqlite"
    cache = EvaluationCache(path)
    common = {"cache": str(path), "marker": str(tmp_path / "crash.marker")}
    tasks = [{**common, "key": "success"}, {**common, "key": "crash"}]
    result = ResourceScheduler(ResourceManager(config(max_workers=2), 100)).map(crash_once_task, tasks)
    assert result == [{"key": "success"}, {"key": "crash"}]
    assert len(cache.completed()) == 2


def test_repeated_batch_shutdown_has_no_memmapping_tracker_errors(tmp_path):
    script = """
from memetic_fs.config import ResourceConfig
from memetic_fs.execution import ResourceManager, ResourceScheduler
scheduler = ResourceScheduler(ResourceManager(ResourceConfig(max_workers=2, threads_per_worker=1), 100))
for i in range(4):
    assert scheduler.map(lambda value: value * value, range(6)) == [0, 1, 4, 9, 16, 25]
print('clean shutdown')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert "clean shutdown" in result.stdout
    assert "resource_tracker" not in result.stderr
    assert "KeyError" not in result.stderr


def test_private_memory_monitor():
    with MemoryMonitor(interval=0.001) as monitor:
        allocation = np.ones(100_000, dtype=np.float64)
        assert allocation.sum() == 100_000
    assert monitor.peak_private_bytes >= monitor.baseline_private_bytes > 0
    assert monitor.peak_increment_bytes >= 0
