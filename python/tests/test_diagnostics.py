"""Unit tests for persistent diagnostics logging and memory sampling."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from asee.diagnostics import (
    JsonlDiagnosticsLogger,
    MemoryMonitor,
    ProcessMetrics,
    read_process_metrics,
)


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    @property
    def path(self) -> Path | None:
        return None

    def log_event(self, event: str, **fields: object) -> None:
        self.events.append((event, dict(fields)))

    def open_fault_handler_stream(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_jsonl_diagnostics_logger_writes_structured_events(tmp_path: Path) -> None:
    log_path = tmp_path / "video-server.jsonl"
    logger = JsonlDiagnosticsLogger(log_path, clock=lambda: 123.456)

    logger.log_event("server_starting", pid=42, camera_list=[0, 2])
    fault_stream = logger.open_fault_handler_stream()
    logger.close()

    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload == {
        "camera_list": [0, 2],
        "event": "server_starting",
        "pid": 42,
        "ts": 123.456,
    }
    assert fault_stream is not None
    assert Path(fault_stream.name).name == "video-server.fault.log"


def test_jsonl_diagnostics_logger_ignores_events_after_close(tmp_path: Path) -> None:
    log_path = tmp_path / "video-server.jsonl"
    logger = JsonlDiagnosticsLogger(log_path, clock=lambda: 123.456)

    logger.close()
    logger.log_event("server_stop_requested")

    assert log_path.read_text(encoding="utf-8") == ""


def test_read_process_metrics_reads_linux_proc_status(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    (proc_root / "status").write_text(
        "VmRSS:\t1024 kB\nVmHWM:\t2048 kB\nThreads:\t12\n",
        encoding="utf-8",
    )
    fd_dir = proc_root / "fd"
    fd_dir.mkdir()
    for name in ("0", "1", "2"):
        (fd_dir / name).write_text("", encoding="utf-8")

    original_active_count = threading.active_count
    threading.active_count = lambda: 5
    try:
        metrics = read_process_metrics(proc_root)
    finally:
        threading.active_count = original_active_count

    assert metrics.rss_kib == 1024
    assert metrics.hwm_kib == 2048
    assert metrics.thread_count == 12
    assert metrics.python_thread_count == 5
    assert metrics.open_fd_count == 3


def test_memory_monitor_sample_once_logs_growth_from_baseline() -> None:
    samples = iter(
        [
            ProcessMetrics(
                rss_kib=1024,
                hwm_kib=2048,
                thread_count=3,
                python_thread_count=2,
                open_fd_count=10,
                gc_gen0=1,
                gc_gen1=2,
                gc_gen2=3,
                tracemalloc_current_kib=256,
                tracemalloc_peak_kib=512,
            ),
            ProcessMetrics(
                rss_kib=1280,
                hwm_kib=2304,
                thread_count=4,
                python_thread_count=2,
                open_fd_count=12,
                gc_gen0=4,
                gc_gen1=5,
                gc_gen2=6,
                tracemalloc_current_kib=320,
                tracemalloc_peak_kib=768,
            ),
        ]
    )
    logger = RecordingLogger()
    monitor = MemoryMonitor(
        logger,
        interval_sec=30.0,
        enable_tracemalloc=False,
        metrics_reader=lambda: next(samples),
    )

    monitor.sample_once()
    monitor.sample_once()

    assert [event for event, _ in logger.events] == ["memory_sample", "memory_sample"]
    first = logger.events[0][1]
    second = logger.events[1][1]
    assert first["rss_delta_from_baseline_kib"] is None
    assert first["tracemalloc_delta_from_baseline_kib"] is None
    assert first["native_thread_surplus"] == 1
    assert second["native_thread_surplus"] == 2
    assert second["rss_delta_from_baseline_kib"] == 256
    assert second["tracemalloc_delta_from_baseline_kib"] == 64
