"""Timestamped progress using standard Python logging, without global setup.

The parent writes ``logs/run.log``. Every worker writes its own process log so
multiple processes never share a FileHandler. Console output is concise at
verbosity 1 and includes candidate/fold details at verbosity 2. Record fields
are available as ``record.progress_fields`` to existing logging integrations.
"""
from __future__ import annotations

from functools import lru_cache
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import threading
from typing import Any

from tqdm.auto import tqdm


class _ProgressFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record)


def _ancestor_has_output(logger: logging.Logger) -> bool:
    current = logger.parent
    while current is not None:
        if any(not isinstance(handler, logging.NullHandler) for handler in current.handlers):
            return True
        current = current.parent
    return False


class ProgressReporter:
    """Per-experiment progress logger, safe for notebooks and reusable workers.

    ``verbose=0`` shows warnings/errors, 1 shows milestones and a generation bar,
    and 2 additionally shows routine progress and candidate/fold detail. The files
    retain DEBUG details regardless of console verbosity. Existing ancestor
    logging handlers receive records; an extra console handler is added only
    when the application has not configured logging already.
    """

    def __init__(self, directory: str | Path, verbose: int = 1, parent_pid: int | None = None,
                 log_name: str = "run.log"):
        if not isinstance(verbose, int) or isinstance(verbose, bool) or verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0 (quiet), 1 (progress), or 2 (detailed)")
        self.directory = Path(directory).resolve()
        self.verbose = verbose
        self.parent_pid = os.getpid() if parent_pid is None else parent_pid
        self._bar = None
        self._worker_status = {"workers_total": 0, "running": 0, "completed": 0, "failed": 0}
        logs = self.directory / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        process = os.getpid()
        self.path = logs / (log_name if process == self.parent_pid else f"worker-{process}.log")
        identity = hashlib.sha256(str(self.path).encode()).hexdigest()[:12]
        self.logger = logging.getLogger(f"memetic_fs.progress.{identity}.{self.parent_pid}.{process}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = True
        formatter = _ProgressFormatter("%(asctime)sZ %(levelname)s pid=%(process)d %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        own_files = [handler for handler in self.logger.handlers if getattr(handler, "_memetic_file", False)]
        if not own_files:
            handler = logging.FileHandler(self.path, encoding="utf-8")
            handler._memetic_file = True
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        console = [handler for handler in self.logger.handlers if getattr(handler, "_memetic_console", False)]
        if process == self.parent_pid and not _ancestor_has_output(self.logger) and not console:
            handler = logging.StreamHandler()
            handler._memetic_console = True
            handler.setFormatter(formatter)
            handler.addFilter(lambda record: not _ancestor_has_output(self.logger))
            self.logger.addHandler(handler)
            console = [handler]
        for handler in console:
            handler.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[verbose])

    @property
    def descriptor(self) -> dict:
        """Small serializable worker settings, containing no data or handlers."""
        return {"directory": str(self.directory), "verbose": self.verbose, "parent_pid": self.parent_pid}

    def _emit(self, level: int, message: str, fields: dict[str, Any], *, screen: bool = True) -> None:
        if fields:
            message += " | " + " ".join(f"{key}={json.dumps(value, default=str, ensure_ascii=True)}" for key, value in sorted(fields.items()))
        # Parent logger levels are not automatically applied to propagated
        # records. Keep DEBUG detail in our files without flooding an existing
        # application's INFO console handler.
        process_is_worker = os.getpid() != self.parent_pid
        if (not screen or process_is_worker
                or (_ancestor_has_output(self.logger) and level < logging.getLogger("memetic_fs").getEffectiveLevel())):
            record = self.logger.makeRecord(self.logger.name, level, __file__, 0, message, (), None,
                                            extra={"progress_fields": fields})
            for handler in self.logger.handlers:
                if level >= handler.level and (screen and not process_is_worker or getattr(handler, "_memetic_file", False)):
                    handler.handle(record)
            return
        self.logger.log(level, message, extra={"progress_fields": fields})

    def info(self, message: str, **fields: Any) -> None:
        """Record routine progress in files and at console verbosity 2."""
        self._emit(logging.INFO, message, fields, screen=self.verbose == 2)

    def milestone(self, message: str, **fields: Any) -> None:
        """Show a concise milestone at verbosity 1 or 2."""
        self._emit(logging.INFO, message, fields, screen=self.verbose > 0)

    def debug(self, message: str, **fields: Any) -> None:
        """Record detailed candidate, fold or cache progress."""
        self._emit(logging.DEBUG, message, fields, screen=self.verbose == 2)

    def start_generations(self, total: int, completed: int = 0) -> None:
        if self.verbose == 1 and os.getpid() == self.parent_pid:
            self._bar = tqdm(total=total, initial=completed, desc="Generations", unit="gen",
                             dynamic_ncols=True, leave=True)
            self._refresh_bar()

    def generation(self, number: int) -> None:
        if self._bar is not None:
            self._bar.set_description_str(f"Running generation {number}/{self._bar.total}")
            self._refresh_bar()

    def generation_completed(self, number: int) -> None:
        if self._bar is not None:
            self._bar.update(max(0, number - self._bar.n))
            self._refresh_bar()

    def worker_status(self, total: int, running: int, completed: int, failed: int) -> None:
        self._worker_status = dict(workers_total=total, running=running, completed=completed, failed=failed)
        self._refresh_bar()

    def _refresh_bar(self) -> None:
        if self._bar is not None:
            self._bar.set_postfix(self._worker_status, refresh=True)

    def finish_generations(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def warning(self, message: str, **fields: Any) -> None:
        """Record operational warnings and recoverable failures."""
        self._emit(logging.WARNING, message, fields)

    def error(self, message: str, **fields: Any) -> None:
        """Record an actionable failure."""
        self._emit(logging.ERROR, message, fields)

    def close(self) -> None:
        """Flush and release log files; later messages reopen append handlers."""
        for handler in self.logger.handlers:
            if getattr(handler, "_memetic_file", False):
                handler.flush()
                handler.close()


class _NullProgress:
    descriptor = None

    def info(self, message: str, **fields: Any) -> None:
        pass

    debug = info
    milestone = info
    warning = info
    error = info
    def start_generations(self, *args: Any, **fields: Any) -> None:
        pass

    generation = start_generations
    generation_completed = start_generations
    worker_status = start_generations
    finish_generations = start_generations

    def close(self) -> None:
        pass


NULL_PROGRESS = _NullProgress()


def get_progress(owner):
    """Use an object's reporter when present; preserve lightweight test doubles."""
    return getattr(owner, "progress", None) or NULL_PROGRESS


@lru_cache(maxsize=32)
def _process_reporter(directory: str, verbose: int, parent_pid: int) -> ProgressReporter:
    return ProgressReporter(directory, verbose, parent_pid)


def progress_from_descriptor(descriptor: dict | None):
    """Open the correct per-process logger once for a reusable worker."""
    return _process_reporter(**descriptor) if descriptor else NULL_PROGRESS


@contextmanager
def progress_activity(reporter, message: str, interval_seconds: float = 30, **fields: Any):
    """Emit a heartbeat during a long native fit or a pending model batch.

    The timer never changes model parameters or inspects observations. It stops
    immediately when the surrounding activity finishes or raises.
    """
    if reporter is NULL_PROGRESS:
        yield
        return
    stopped = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not stopped.wait(interval_seconds):
            reporter.info(message, elapsed_seconds=round(time.monotonic() - started, 1), **fields)

    thread = threading.Thread(target=heartbeat, name="memetic-fs-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
