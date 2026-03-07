# asee

Agentic seeing backend for camera recognition and biometric status.

## Current Scope

- Python package scaffold with `pytest`, `mypy`, and `ruff`
- extracted pure helpers from `tmp/GOD_MODE`
  - camera layout parsing / optional camera selection
  - biometric status aggregation independent from HTTP / OpenCV runtime
  - web shell asset builders for the future Electron viewer
  - Flask HTTP shell contract for `god_mode_video_server.py`
  - DNN backend policy helpers for `god_mode_overlay.py`
- `god_mode_predictor.py` is intentionally excluded from migration for now

## Commands

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/mypy src
.venv/bin/ruff check
```

## Initial Modules

- `asee.camera_layout`
  - `parse_v4l2_devices()`
  - `extend_with_optional_camera()`
  - `build_camera_csv()`
- `asee.biometric_status`
  - `BiometricStatusTracker`
- `asee.web_shell`
  - `build_web_manifest()`
  - `build_service_worker_script()`
  - `build_icon_svg()`
- `asee.http_app`
  - `OverlayTextState`
  - `InMemoryHttpRuntime`
  - `create_http_app()`
- `asee.dnn_policy`
  - `should_use_opencl_dnn()`
  - `emit_opencl_nonfatal_warning_note()`

## Planned Next Slice

- extract more of the Python backend contract from `god_mode_video_server.py`
- extract backend adapters from `god_mode_overlay.py` without dragging OpenCV UI concerns
- add compatibility adapters around the extracted Flask app factory
- keep image processing and biometric inference in Python
- add an Electron viewer separately instead of pushing CV logic into TypeScript

## Notes

- `tmp/GOD_MODE` remains the source of truth until compatibility wrappers exist
- the future Electron UI belongs beside this backend, but not inside the CV/runtime core
