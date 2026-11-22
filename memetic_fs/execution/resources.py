"""Conservative admission based on currently available host/container resources."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    """Effective CPU and memory limits at an admission boundary, in bytes."""
    cpus: int
    capacity_bytes: int
    available_bytes: int
    private_bytes: int
    temporary_free_bytes: int


@dataclass(frozen=True)
class ResourcePlan:
    """Safe next-dispatch limit; zero workers means wait for memory."""
    workers: int
    threads_per_worker: int
    memory_budget_bytes: int
    estimated_task_bytes: int
    snapshot: ResourceSnapshot


def _read_number(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _cgroup_paths(root: Path) -> list[Path]:
    """Include delegated cgroup directories and every enclosing limit."""
    paths = [root, root / "memory", root / "cpu", root / "cpu,cpuacct"]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            bases = [root] if not controllers else [root / controllers, *[root / name for name in controllers.split(",")]]
            for base in bases:
                candidate = base / relative.lstrip("/")
                if candidate.is_relative_to(root):
                    paths.extend([candidate, *[p for p in candidate.parents if p.is_relative_to(root)]])
    except (OSError, ValueError):
        pass
    return list(dict.fromkeys(paths))


def private_process_bytes(process: psutil.Process | None = None) -> int:
    """Use USS (excluding shared mapped pages); RSS is only a fallback."""
    process = process or psutil.Process()
    try:
        info = process.memory_full_info()
        return int(getattr(info, "uss", getattr(info, "pss", info.rss)))
    except (psutil.Error, OSError):
        try:
            return int(process.memory_info().rss)
        except psutil.Error:
            return 0


def detect_resources(temp_dir: str | Path | None = None, cgroup_root: str | Path = "/sys/fs/cgroup") -> ResourceSnapshot:
    """Inspect affinity, CPU quotas, host availability and cgroup v1/v2 limits."""
    memory = psutil.virtual_memory()
    capacity, available = int(memory.total), int(memory.available)
    process = psutil.Process()
    cpus = os.cpu_count() or 1
    try:
        cpus = min(cpus, len(process.cpu_affinity()))
    except (AttributeError, psutil.Error, OSError):
        pass
    for path in _cgroup_paths(Path(cgroup_root)):
        limit = _read_number(path / "memory.max")
        used = _read_number(path / "memory.current")
        if limit is None:
            limit = _read_number(path / "memory.limit_in_bytes")
            used = _read_number(path / "memory.usage_in_bytes")
        if limit is not None and limit > 0:
            capacity = min(capacity, limit)
            # Missing usage is not evidence the cgroup is empty.
            available = min(available, max(0, limit - (used if used is not None else memory.total - memory.available)))
        quota, period = None, None
        try:
            quota_text, period_text = (path / "cpu.max").read_text().split()
            quota = None if quota_text == "max" else int(quota_text)
            period = int(period_text)
        except (OSError, ValueError):
            quota = _read_number(path / "cpu.cfs_quota_us")
            period = _read_number(path / "cpu.cfs_period_us")
        if quota is not None and quota > 0 and period is not None and period > 0:
            cpus = min(cpus, max(1, math.floor(quota / period)))
    private = private_process_bytes(process)
    try:
        private += sum(private_process_bytes(child) for child in process.children(recursive=True))
    except psutil.Error:
        pass
    location = Path(temp_dir or tempfile.gettempdir())
    location.mkdir(parents=True, exist_ok=True)
    return ResourceSnapshot(max(1, cpus), capacity, max(0, min(capacity, available)), private, shutil.disk_usage(location).free)


class ResourceManager:
    """Estimate task admission, then refine it with full-row measured memory."""

    def __init__(self, config: Any, estimated_task_bytes: int, temp_dir: str | Path | None = None):
        self.config = config
        self.estimated_task_bytes = max(1, int(estimated_task_bytes))
        self.temp_dir = temp_dir
        self.worker_cap: int | None = None
        self.bound_threads: int | None = None

    def calibrate(self, measured_private_peak: int) -> None:
        """Reserve a 30% margin above observed private working allocation."""
        self.estimated_task_bytes = max(self.estimated_task_bytes, int(measured_private_peak * 1.3))

    def backoff(self, previous_workers: int) -> None:
        """Reduce future concurrency and increase memory estimate after failure."""
        self.worker_cap = max(1, previous_workers // 2)
        self.estimated_task_bytes = max(1, int(self.estimated_task_bytes * 1.5))

    def plan(self) -> ResourcePlan:
        """Return safe concurrency using available memory after the safety reserve."""
        snapshot = detect_resources(self.temp_dir)
        capacity, available = snapshot.capacity_bytes, snapshot.available_bytes
        max_ram = getattr(self.config, "max_ram_gb", None)
        if max_ram is not None:
            cap = int(max_ram * 1024**3)
            capacity = min(capacity, cap)
            available = min(available, max(0, cap - snapshot.private_bytes))
        reserve = float(getattr(self.config, "memory_reserve_fraction", 0.25))
        budget = max(0, int(available - capacity * reserve))
        specified_threads = getattr(self.config, "threads_per_worker", None)
        configured_workers = getattr(self.config, "max_workers", None) or snapshot.cpus
        # Worker admission/backoff must not change the estimator's thread budget:
        # references and candidates use the same effective training settings.
        threads = min(snapshot.cpus, int(specified_threads or max(1, snapshot.cpus // min(4, configured_workers))))
        if self.bound_threads is not None:
            threads = self.bound_threads
            if snapshot.cpus < threads:
                raise RuntimeError(f"Available CPU capacity ({snapshot.cpus}) fell below this run's fixed LightGBM thread budget ({threads}); restore CPU capacity before resuming")
        max_workers = configured_workers
        if self.worker_cap is not None:
            max_workers = min(max_workers, self.worker_cap)
        workers = min(int(max_workers), max(1, snapshot.cpus // threads), budget // self.estimated_task_bytes)
        return ResourcePlan(max(0, workers), max(1, threads), budget, self.estimated_task_bytes, snapshot)
