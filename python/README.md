# asee/python

Python backend for camera recognition and biometric status inside `repos/asee`.

## Current Scope

- Python package scaffold with `pytest`, `mypy`, and `ruff`
- extracted pure helpers from `tmp/GOD_MODE`
  - camera layout parsing / optional camera selection
  - biometric status aggregation independent from HTTP / OpenCV runtime
  - web shell asset builders for the future Electron viewer
  - Flask HTTP shell contract for `god_mode_video_server.py`
  - DNN backend policy helpers for `god_mode_overlay.py`
  - face tracking primitives and capture persistence helpers from `god_mode_overlay.py`
  - OWNER selection policy and YuNet detection pipeline from `god_mode_overlay.py`
  - `GodModeOverlay` runtime rebuilt on top of extracted `asee` primitives
  - viewer/server state holder rebuilt from `god_mode_video_server.py`
  - `GodModeVideoServer` compatibility server rebuilt from `god_mode_video_server.py`
  - OWNER enrollment flow rebuilt from `god_mode_enroll_owner.py`
- `god_mode_predictor.py` is intentionally excluded from migration for now

## Commands

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/mypy src
.venv/bin/ruff check
.venv/bin/python -m asee.video_server --port 8765 --device 0
```

## Initial Modules

- `asee.camera_layout`
  - `parse_camera_csv()`
  - `parse_v4l2_devices()`
  - `extend_with_optional_camera()`
  - `detect_v4l2_devices()`
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
- `asee.tracking`
  - `FaceBox`
  - `FaceTracker`
- `asee.capture_writer`
  - `FaceCaptureWriter`
- `asee.owner_policy`
  - `keep_largest_owner()`
  - `OWNER_TOPK`
  - `OWNER_COSINE_THRESHOLD`
- `asee.detection_runtime`
  - `to_square()`
  - `YunetDetectionPipeline`
- `asee.overlay`
  - `GodModeOverlay`
  - rebuilt overlay drawing, detection, classification, and capture-writer integration
- `asee.server_runtime`
  - `SeeingServerRuntime`
  - overlay state, biometric status, owner embedding load, snapshot/stream contract
- `asee.video_server`
  - `GodModeVideoServer`
  - `encode_frame_to_jpeg()`
  - camera-less/single-camera HTTP compatibility server on top of extracted modules
- `asee.enroll_owner`
  - `fetch_frame_from_server()`
  - `run_enrollment()`
  - OWNER embedding capture from server snapshot or direct camera fallback

## Planned Next Slice

- extract more of the Python backend contract from `god_mode_video_server.py`
- replace direct `tmp/GOD_MODE` detector/classifier orchestration with `asee.overlay.GodModeOverlay`
- add a compatibility runtime adapter so `tmp/GOD_MODE` can call `asee` modules as the new source of truth
- connect `tmp/GOD_MODE/god_mode_video_server.py` to `asee.server_runtime.SeeingServerRuntime`
- move more of `tmp/GOD_MODE/god_mode_video_server.py` onto `asee.video_server.GodModeVideoServer`
- add compatibility adapters around the extracted Flask app factory
- keep image processing and biometric inference in Python
- add an Electron viewer separately instead of pushing CV logic into TypeScript

## Notes

- `tmp/GOD_MODE/god_mode_overlay.py` and `tmp/GOD_MODE/god_mode_video_server.py` can now act as compatibility wrappers over `asee`
- `tmp/GOD_MODE/god_mode_camera_layout.py` can also act as a compatibility wrapper over `asee.camera_layout`
- `tmp/GOD_MODE/god_mode_enroll_owner.py` can also act as a compatibility wrapper over `asee.enroll_owner`
- the future Electron UI belongs beside this backend, but not inside the CV/runtime core
