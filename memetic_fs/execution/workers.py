"""Worker-side memory measurements, excluding shared file-backed pages."""
from __future__ import annotations

import threading

from .resources import private_process_bytes


class MemoryMonitor:
    """Sample private memory during an actual full-fold evaluation.

    The peak is sampled rather than an OS-enforced allocation limit. Very short
    allocation spikes can be missed; admission adds a conservative margin.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_private_bytes = 0
        self.baseline_private_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_private_bytes = max(self.peak_private_bytes, private_process_bytes())

    def __enter__(self) -> "MemoryMonitor":
        self.baseline_private_bytes = private_process_bytes()
        self.peak_private_bytes = self.baseline_private_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.peak_private_bytes = max(self.peak_private_bytes, private_process_bytes())
        self._stop.set()
        if self._thread:
            self._thread.join()

    @property
    def peak_increment_bytes(self) -> int:
        """Peak additional private allocation above the pre-evaluation baseline."""
        return max(0, self.peak_private_bytes - self.baseline_private_bytes)
