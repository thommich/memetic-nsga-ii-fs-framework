"""Bounded batch scheduling with memory admission and recoverable retries."""
from __future__ import annotations

import logging
import time
from concurrent.futures import as_completed
from typing import Callable, Iterable, TypeVar

from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.process_executor import TerminatedWorkerError
from threadpoolctl import threadpool_limits

from .resources import ResourceManager
from ..progress import NULL_PROGRESS, progress_activity

T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger(__name__)


def _limited_call(fn: Callable[[T], R], task: T, threads: int) -> R:
    with threadpool_limits(limits=threads):
        return fn(task)


class ResourceScheduler:
    """Reusable loky execution with resource checks before each admitted batch.

    Tasks must carry lightweight read-only mmap descriptors rather than full
    datasets. Workers should commit results to the cache before returning, so
    completed evaluations survive a different worker terminating in that batch.
    """

    def __init__(self, manager: ResourceManager, cancelled: Callable[[], bool] | None = None, progress=None):
        self.manager = manager
        self.cancelled = cancelled or (lambda: False)
        self.progress = progress or NULL_PROGRESS

    def _admit(self):
        deadline = time.monotonic() + float(getattr(self.manager.config, "memory_wait_seconds", 30))
        while True:
            if self.cancelled():
                self.progress.warning("Worker dispatch cancelled")
                raise KeyboardInterrupt("Feature-selection run cancelled before dispatch")
            plan = self.manager.plan()
            if plan.workers > 0:
                return plan
            if time.monotonic() >= deadline:
                self.progress.error("Worker admission failed under memory pressure", memory_budget_bytes=plan.memory_budget_bytes,
                                    estimated_task_bytes=plan.estimated_task_bytes)
                raise MemoryError("No full-fold worker fits the current memory budget. Free memory, increase the RAM limit, or use a larger instance; observations will not be sampled.")
            logger.warning("Pausing new model dispatch while available memory is below the admission reserve")
            self.progress.warning("Worker dispatch paused under memory pressure", memory_budget_bytes=plan.memory_budget_bytes,
                                  estimated_task_bytes=plan.estimated_task_bytes)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    def map(self, fn: Callable[[T], R], tasks: Iterable[T]) -> list[R]:
        """Evaluate tasks in order, retrying memory failures with safer concurrency."""
        task_list = list(tasks)
        if not task_list:
            return []
        results: list[R] = []
        cursor = 0
        retries = 0
        while cursor < len(task_list):
            plan = self._admit()
            workers = min(plan.workers, len(task_list) - cursor)
            batch = task_list[cursor:cursor + workers]
            started = time.perf_counter()
            self.progress.info("Worker batch admitted", tasks=len(batch), workers=workers,
                               threads_per_worker=plan.threads_per_worker, completed_tasks=cursor,
                               total_tasks=len(task_list), memory_budget_bytes=plan.memory_budget_bytes,
                               estimated_task_bytes=plan.estimated_task_bytes)
            self.progress.worker_status(total=len(batch), running=len(batch), completed=0, failed=0)
            try:
                if workers == 1:
                    try:
                        batch_results = [_limited_call(fn, batch[0], plan.threads_per_worker)]
                    except BaseException:
                        self.progress.worker_status(total=1, running=0, completed=0, failed=1)
                        raise
                    self.progress.worker_status(total=1, running=0, completed=1, failed=0)
                else:
                    # Descriptors contain paths, not matrices. Using loky
                    # directly avoids creating redundant joblib temporary
                    # memmapping-folder contexts for every batch.
                    environment = {name: str(plan.threads_per_worker) for name in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS")}
                    executor = get_reusable_executor(max_workers=workers, timeout=300, env=environment)
                    futures = [executor.submit(_limited_call, fn, task, plan.threads_per_worker) for task in batch]
                    batch_results, errors = [None] * len(futures), []
                    completed, failed = 0, 0
                    try:
                        # Drain the bounded batch before retrying: surviving
                        # workers can commit their completed cache records.
                        with progress_activity(self.progress, "Model batch still running", workers=workers,
                                               completed_tasks=cursor, total_tasks=len(task_list)):
                            positions = {future: index for index, future in enumerate(futures)}
                            for future in as_completed(futures):
                                try:
                                    batch_results[positions[future]] = future.result()
                                    completed += 1
                                except Exception as error:
                                    errors.append(error)
                                    failed += 1
                                self.progress.worker_status(total=len(batch),
                                    running=len(batch)-completed-failed, completed=completed, failed=failed)
                        if errors:
                            raise errors[0]
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise
                results.extend(batch_results)
                cursor += len(batch)
                retries = 0
                # A successful safer batch may regain concurrency at later
                # boundaries as resource availability improves.
                self.manager.worker_cap = None
                for result in batch_results:
                    if isinstance(result, dict) and result.get("memory_peak_bytes"):
                        self.manager.calibrate(int(result["memory_peak_bytes"]))
                self.progress.info("Worker batch completed", tasks=len(batch_results), workers=workers,
                                   completed_tasks=cursor, total_tasks=len(task_list),
                                   elapsed_seconds=time.perf_counter() - started)
            except (MemoryError, TerminatedWorkerError) as error:
                retries += 1
                self.manager.backoff(workers)
                if retries > int(getattr(self.manager.config, "max_retries", 2)):
                    self.progress.error("Worker batch retries exhausted", attempts=retries,
                                        error_type=type(error).__name__, completed_tasks=cursor)
                    raise RuntimeError(f"Full-row worker evaluation failed after {retries} attempts; completed cached results remain reusable") from error
                logger.warning("Retrying model batch after %s with reduced concurrency", type(error).__name__)
                self.progress.warning("Retrying worker batch with reduced concurrency", attempt=retries,
                                      error_type=type(error).__name__, previous_workers=workers,
                                      estimated_task_bytes=self.manager.estimated_task_bytes)
        return results
