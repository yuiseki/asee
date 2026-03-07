"""Persistent diagnostics logging and memory sampling for the asee runtime."""

from __future__ import annotations

import gc
import json
import threading
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TextIO


class DiagnosticsLogger(Protocol):
    """Minimal diagnostics logger surface used by the video server."""

    @property
    def path(self) -> Path | None: ...

    def log_event(self, event: str, **fields: Any) -> None: ...
    def open_fault_handler_stream(self) -> TextIO | None: ...
    def close(self) -> None: ...


class NullDiagnosticsLogger:
    """No-op logger used by tests or in-process callers that do not want files."""

    @property
    def path(self) -> Path | None:
        return None

    def log_event(self, event: str, **fields: Any) -> None:
        return None

    def open_fault_handler_stream(self) -> TextIO | None:
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class ProcessMetrics:
    """Single point-in-time view of lightweight process memory metrics."""

    rss_kib: int | None
    hwm_kib: int | None
    thread_count: int | None
    python_thread_count: int
    open_fd_count: int | None
    gc_gen0: int
    gc_gen1: int
    gc_gen2: int
    tracemalloc_current_kib: int | None
    tracemalloc_peak_kib: int | None


def build_default_diagnostics_log_path() -> Path:
    """Return a persistent default JSONL path under the user's state directory."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        Path.home()
        / ".local"
        / "state"
        / "asee"
        / "video-server"
        / f"diagnostics-{timestamp}.jsonl"
    )


def _read_status_kib(status_path: Path, key: str) -> int | None:
    if not status_path.exists():
        return None
    prefix = f"{key}:"
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    return None


def _read_status_int(status_path: Path, key: str) -> int | None:
    if not status_path.exists():
        return None
    prefix = f"{key}:"
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    return None


def read_process_metrics(proc_root: Path = Path("/proc/self")) -> ProcessMetrics:
    """Read Linux process metrics suitable for leak/regression tracking."""
    status_path = proc_root / "status"
    fd_dir = proc_root / "fd"
    open_fd_count = len(list(fd_dir.iterdir())) if fd_dir.exists() else None
    tracemalloc_current_kib: int | None = None
    tracemalloc_peak_kib: int | None = None
    if tracemalloc.is_tracing():
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc_current_kib = current_bytes // 1024
        tracemalloc_peak_kib = peak_bytes // 1024
    gc_gen0, gc_gen1, gc_gen2 = gc.get_count()
    return ProcessMetrics(
        rss_kib=_read_status_kib(status_path, "VmRSS"),
        hwm_kib=_read_status_kib(status_path, "VmHWM"),
        thread_count=_read_status_int(status_path, "Threads"),
        python_thread_count=threading.active_count(),
        open_fd_count=open_fd_count,
        gc_gen0=gc_gen0,
        gc_gen1=gc_gen1,
        gc_gen2=gc_gen2,
        tracemalloc_current_kib=tracemalloc_current_kib,
        tracemalloc_peak_kib=tracemalloc_peak_kib,
    )


class JsonlDiagnosticsLogger:
    """Append-only JSONL logger that survives across reboots."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._stream = self._path.open("a", encoding="utf-8", buffering=1)
        self._fault_stream: TextIO | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    def log_event(self, event: str, **fields: Any) -> None:
        record = {
            "ts": self._clock(),
            "event": event,
            **fields,
        }
        with self._lock:
            self._stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            self._stream.write("\n")
            self._stream.flush()

    def open_fault_handler_stream(self) -> TextIO | None:
        if self._fault_stream is None:
            fault_path = self._path.with_suffix(".fault.log")
            self._fault_stream = fault_path.open("a", encoding="utf-8", buffering=1)
        return self._fault_stream

    def close(self) -> None:
        with self._lock:
            if self._fault_stream is not None:
                self._fault_stream.close()
                self._fault_stream = None
            self._stream.close()


class MemoryMonitor:
    """Periodic process metrics sampler for leak and hang analysis."""

    def __init__(
        self,
        logger: DiagnosticsLogger,
        *,
        interval_sec: float = 30.0,
        enable_tracemalloc: bool = True,
        metrics_reader: Callable[[], ProcessMetrics] = read_process_metrics,
    ) -> None:
        self._logger = logger
        self._interval_sec = interval_sec
        self._enable_tracemalloc = enable_tracemalloc
        self._metrics_reader = metrics_reader
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: ProcessMetrics | None = None

    def start(self) -> None:
        if self._interval_sec <= 0:
            return
        if self._thread is not None:
            return
        if self._enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)
        self._logger.log_event(
            "memory_monitor_started",
            interval_sec=self._interval_sec,
            tracemalloc_enabled=tracemalloc.is_tracing(),
        )
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="memory_monitor",
        )
        self._thread.start()

    def sample_once(self) -> ProcessMetrics:
        metrics = self._metrics_reader()
        baseline = self._baseline
        if baseline is None:
            self._baseline = metrics
        self._logger.log_event(
            "memory_sample",
            rss_kib=metrics.rss_kib,
            hwm_kib=metrics.hwm_kib,
            thread_count=metrics.thread_count,
            python_thread_count=metrics.python_thread_count,
            native_thread_surplus=_delta(
                metrics.thread_count,
                metrics.python_thread_count,
            ),
            open_fd_count=metrics.open_fd_count,
            gc_gen0=metrics.gc_gen0,
            gc_gen1=metrics.gc_gen1,
            gc_gen2=metrics.gc_gen2,
            tracemalloc_current_kib=metrics.tracemalloc_current_kib,
            tracemalloc_peak_kib=metrics.tracemalloc_peak_kib,
            rss_delta_from_baseline_kib=_delta(
                metrics.rss_kib,
                baseline.rss_kib if baseline else None,
            ),
            tracemalloc_delta_from_baseline_kib=_delta(
                metrics.tracemalloc_current_kib,
                baseline.tracemalloc_current_kib if baseline else None,
            ),
        )
        return metrics

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=max(1.0, self._interval_sec + 0.5))
        self._logger.log_event("memory_monitor_stopped")
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            if self._stop_event.wait(self._interval_sec):
                break


def _delta(current: int | None, baseline: int | None) -> int | None:
    if current is None or baseline is None:
        return None
    return current - baseline
