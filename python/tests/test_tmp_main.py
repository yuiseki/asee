from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "tmp_main.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _build_stubbed_env(tmp_path: Path, *, port: int) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    invocation_log = tmp_path / "invocations.jsonl"

    _write_executable(
        bin_dir / "python-stub",
        """#!/usr/bin/env python3
import json
import os
import sys

log_path = os.environ["ASEE_TMP_MAIN_INVOCATIONS"]
with open(log_path, "a", encoding="utf-8") as handle:
    json.dump({"cmd": "python", "argv": sys.argv[1:]}, handle)
    handle.write("\\n")
""",
    )
    _write_executable(
        bin_dir / "npm",
        """#!/usr/bin/env python3
import json
import os
import sys

log_path = os.environ["ASEE_TMP_MAIN_INVOCATIONS"]
with open(log_path, "a", encoding="utf-8") as handle:
    json.dump(
        {
            "cmd": "npm",
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
            "backend_url": os.environ.get("ASEE_VIEWER_BACKEND_URL"),
            "viewer_title": os.environ.get("ASEE_VIEWER_TITLE"),
            "poll_interval_ms": os.environ.get("ASEE_VIEWER_POLL_INTERVAL_MS"),
        },
        handle,
    )
    handle.write("\\n")
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
printf '{"running":true}\\n'
""",
    )
    _write_executable(
        bin_dir / "wmctrl",
        """#!/usr/bin/env bash
printf '0x001  0 test ASEE Viewer\\n'
""",
    )
    _write_executable(
        bin_dir / "qdbus",
        """#!/usr/bin/env bash
exit 0
""",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["ASEE_PYTHON_BIN"] = str(bin_dir / "python-stub")
    env["ASEE_TMP_MAIN_INVOCATIONS"] = str(invocation_log)
    env["DISPLAY"] = ":0"
    return env, invocation_log


def _load_invocations(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _find_invocation(invocations: list[dict[str, object]], command: str) -> dict[str, object]:
    for invocation in invocations:
        if invocation.get("cmd") == command:
            return invocation
    raise AssertionError(f"{command} invocation was not captured")


def test_start_launches_backend_and_electron_viewer_with_720p_profile(tmp_path: Path) -> None:
    env, invocation_log = _build_stubbed_env(tmp_path, port=19140)

    result = subprocess.run(
        [
            str(SCRIPT),
            "start",
            "--port",
            "19140",
            "--cameras",
            "0,2,4,6",
            "--capture-profile",
            "720p",
            "--auto-shutdown-sec",
            "15",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    invocations = _load_invocations(invocation_log)
    python_invocation = _find_invocation(invocations, "python")
    python_argv = [str(item) for item in python_invocation["argv"]]
    assert "-m" in python_argv
    assert "asee.video_server" in python_argv
    assert "--capture-profile" in python_argv
    assert python_argv[python_argv.index("--capture-profile") + 1] == "720p"

    npm_invocation = _find_invocation(invocations, "npm")
    assert npm_invocation["argv"] == ["run", "start"]
    assert npm_invocation["backend_url"] == "http://127.0.0.1:19140"
    assert npm_invocation["viewer_title"] == "ASEE Viewer"


def test_start_accepts_legacy_noop_flags(tmp_path: Path) -> None:
    env, invocation_log = _build_stubbed_env(tmp_path, port=19142)

    result = subprocess.run(
        [
            str(SCRIPT),
            "start",
            "--port",
            "19142",
            "--device",
            "0",
            "--interval",
            "45",
            "--voice",
            "--chromium",
            "--pwa-installing",
            "--ollama-vlm",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    invocations = _load_invocations(invocation_log)
    python_invocation = _find_invocation(invocations, "python")
    python_argv = [str(item) for item in python_invocation["argv"]]
    assert "--device" in python_argv
    assert python_argv[python_argv.index("--device") + 1] == "0"

    npm_invocation = _find_invocation(invocations, "npm")
    assert npm_invocation["argv"] == ["run", "start"]
    assert npm_invocation["viewer_title"] == "ASEE Viewer"


def test_stop_removes_pid_file_and_terminates_recorded_processes(tmp_path: Path) -> None:
    server_proc = subprocess.Popen(["sleep", "300"], start_new_session=True)
    viewer_proc = subprocess.Popen(["sleep", "300"], start_new_session=True)
    pid_file = Path("/tmp/asee_tmp_main_19141.pids")
    pid_file.write_text(
        f"server={server_proc.pid}\nviewer={viewer_proc.pid}\n",
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [str(SCRIPT), "stop", "--port", "19141"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert not pid_file.exists()

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if server_proc.poll() is not None and viewer_proc.poll() is not None:
                break
            time.sleep(0.05)

        assert server_proc.poll() is not None
        assert viewer_proc.poll() is not None
    finally:
        for proc in (server_proc, viewer_proc):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        if pid_file.exists():
            pid_file.unlink()


def test_stop_removes_stale_pid_file_when_processes_are_gone() -> None:
    pid_file = Path("/tmp/asee_tmp_main_19143.pids")
    pid_file.write_text("server=999991\nviewer=999992\n", encoding="utf-8")

    try:
        result = subprocess.run(
            [str(SCRIPT), "stop", "--port", "19143"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert not pid_file.exists()
    finally:
        if pid_file.exists():
            pid_file.unlink()
