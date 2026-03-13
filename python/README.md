# asee/python

Python backend for camera recognition and biometric status inside `repos/asee`.

## Current Scope

- Python package scaffold with `pytest`, `mypy`, and `ruff`
- extracted pure helpers from the legacy GOD MODE runtime, now archived under `tmp/_trash/GOD_MODE`
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
  - first staged WebRTC transport slice for the future low-latency viewer path
    - overlay payload schema
    - overlay broadcaster
    - aiohttp + aiortc signaling app factory
    - runtime-backed WebRTC video track over `SeeingServerRuntime`
- `god_mode_predictor.py` is intentionally excluded from migration for now

## Commands

```bash
cd ..
./tmp_main.sh start --port 8765 --cameras 0,2,4,6 --capture-profile 720p --auto-shutdown-sec 30
./tmp_main.sh status --port 8765
./tmp_main.sh stop --port 8765
```

- `../tmp_main.sh` is now the temporary canonical launcher when the backend should run together with the official Electron viewer.
- it now forwards both `--face-capture-dir` and `--subject-capture-dir`, with the default persistent SUBJECT dataset under `/home/yuiseki/Workspaces/private/datasets/faces/others`.
- before guests arrive, move the current single-owner false negatives out of `others`:
  `cd /home/yuiseki/Workspaces/repos/asee/python && PYTHONPATH=src python3 -m asee.relabel_owner_false_negatives`
- offline retrain/validation helper for owner embeddings:
  `cd /home/yuiseki/Workspaces/repos/asee/python && PYTHONPATH=src python3 -m asee.retrain_owner_embedding --negative-validation-dir /home/yuiseki/Workspaces/private/datasets/faces/others_guest_session/<session>`
  - default mode is validation-only; add `--apply` only after the before/after metrics look safe.
- mixed subject-session triage helper:
  `cd /home/yuiseki/Workspaces/repos/asee/python && GOD_MODE_DISABLE_OPENCL_DNN=1 PYTHONPATH=src python3 -m asee.triage_mixed_subject_session --session-dir /home/yuiseki/Workspaces/private/datasets/faces/others_guest_session/<session>`
  - writes conservative copies into `private/datasets/faces/others_guest_session_triaged/<session>/{likely_guest_negative,likely_owner_false_negative,uncertain}` plus `manifest.jsonl`
- owner-only false-negative triage helper:
  `cd /home/yuiseki/Workspaces/repos/asee/python && GOD_MODE_DISABLE_OPENCL_DNN=1 PYTHONPATH=src python3 -m asee.triage_owner_only_false_negatives --source /home/yuiseki/Workspaces/private/datasets/faces/others/2026/03/14`
  - writes conservative copies into `private/datasets/faces/others_owner_only_triaged/<source-leaf>/{likely_owner_false_negative,low_quality,uncertain}` plus `manifest.jsonl`
- tilted owner hard-positive selector:
  `cd /home/yuiseki/Workspaces/repos/asee/python && GOD_MODE_DISABLE_OPENCL_DNN=1 PYTHONPATH=src python3 -m asee.tilted_owner_hard_positive_selector --source /home/yuiseki/Workspaces/private/datasets/faces/others_owner_only_triaged/2026-03-14/likely_owner_false_negative`
  - re-detects eye-line roll on conservative owner-only false negatives, dedupes near-identical embeddings, and writes append-only tilted candidates into `private/datasets/faces/tilted_owner_hard_positive_candidates/<source-parent>/` plus `manifest.jsonl`
- during guest-time collection, both OWNER and SUBJECT crops now use a relaxed `10s` minimum interval plus large guard rails (`500000` files/day, `500000` total files, `50GB`) so rare false-positive / false-negative examples are retained.
- each captured face crop now writes a sidecar `.json` with timestamp, score, label, face box, and per-frame counts for later relabeling and session triage.
- the official Electron window caption is `ASEE Viewer`.
- `stop` performs process-group cleanup for both backend and viewer, tolerates stale pid files, and removes the launcher pid file even after bounded auto-shutdown runs.
- `tmp_main.sh` now builds the Electron app before launch and supervises the viewer process separately, so a viewer-only crash can be retried without restarting the backend.
- that viewer supervisor also reapplies the default left-bottom layout on each launch, so a respawned window snaps back instead of reopening in the center.
- direct Electron launch now also defaults to the left-bottom half of the primary work area, keeping standalone WebRTC experiments aligned with the supervised layout slot.
- `ASEE_VIEWER_RESPAWN=0` disables that retry loop for bounded experiments.
- GPU backend experiments and PRIME offload hints can now be injected through `tmp_main.sh` with env vars such as `ASEE_VIEWER_USE_GL=desktop`, `ASEE_VIEWER_USE_ANGLE=gl`, `ASEE_VIEWER_DISABLE_GPU_SANDBOX=1`, `ASEE_VIEWER_EXTRA_ARGS='--enable-logging=stderr --v=1'`, `__NV_PRIME_RENDER_OFFLOAD=1`, `__NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, and `DRI_PRIME=1`.
- `tmp_main.sh` logs those effective GPU env settings plus the resolved Electron args into `/tmp/asee_tmp_main_viewer_<port>.log` on each viewer launch.
- legacy `god_mode.sh`-style flags like `--chromium` are still accepted as compatibility no-ops by `tmp_main.sh`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/mypy src
.venv/bin/ruff check
.venv/bin/python -m asee.video_server --port 8765
# live camera is opt-in only
.venv/bin/python -m asee.video_server --port 8765 --device 0 --allow-live-camera
# WebRTC is now the default transport; MJPEG remains available as a fallback
.venv/bin/python -m asee.video_server --port 8765
.venv/bin/python -m asee.video_server --port 8765 --transport mjpeg
# bounded current multicamera default
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --auto-shutdown-sec 30
# isolate camera/native pressure from face-detect pressure
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --disable-face-detect --auto-shutdown-sec 30
# bounded 720p multi-camera trial
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --capture-profile 720p --opencv-threads 1 --auto-shutdown-sec 30
# explicit low-resolution fallback experiment
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --width 640 --height 360 --fps 10 --opencv-threads 1 --auto-shutdown-sec 30
# bounded live test window
.venv/bin/python -m asee.video_server --port 8765 --device 0 --allow-live-camera --auto-shutdown-sec 180
```

- WebRTC is now the default runtime path for direct CLI launches.
- The current staging modules are:
  - `asee.overlay_data`
  - `asee.overlay_broadcaster`
  - `asee.webrtc_signaling`
  - `asee.webrtc_video_track`
- `asee.video_server` now starts the staged signaling path by default against the same `SeeingServerRuntime`.
- `--transport mjpeg` keeps the old Flask/MJPEG path available as a compatibility fallback.
- The existing `asee.video_server` centralized detection/runtime remains the source of truth that the later WebRTC path should wrap, not replace.
- the official Electron viewer can now consume that staged path through the same preload snapshot contract, selecting WebRTC when `/status.transport` says so.
- its staged WebRTC overlay now uses source-frame dimensions from the backend so face boxes stay aligned under `object-fit: cover` and keep the legacy OWNER/SUBJECT HUD styling.

## Safety And Diagnostics

- `asee.video_server` defaults to no-camera mode. Passing `--device 0` is not enough; live capture also requires `--allow-live-camera`.
- single-camera default capture profile stays `1280x720 @ 30fps MJPG`.
- multi-camera default capture profile now returns to `1280x720 @ 30fps MJPG`.
- `--capture-profile 720p` remains the explicit shortcut for that multicamera operating point.
- `asee.enroll_owner` still captures OWNER embeddings through a `1280x720` overlay path, so the restored multi-camera default now matches enrollment conditions more closely.
- if load becomes unacceptable, the explicit fallback is now to lower `--width/--height/--fps` for the experiment instead of relying on the default path.
- `--width`, `--height`, `--fps`, and `--fourcc` can override the requested capture mode when a controlled experiment needs it.
- multi-camera runs also default OpenCV's internal worker pool to `1`; `--opencv-threads` can override that when a benchmark explicitly needs more.
- YuNet, SFace, and `owner_embedding.npy` are now resolved only from `python/src/asee/models/`.
- `python/src/asee/models/` is a private local cache and is gitignored on purpose; copied owner embeddings must never be published.
- `--disable-face-detect` lets us separate camera/native instability from detector/runtime instability.
- Every CLI launch now creates a persistent JSONL diagnostics log under `~/.local/state/asee/video-server/` unless `--diagnostic-log-path` is specified.
- Each run also enables `faulthandler` and writes a sibling `.fault.log`.
- The diagnostics stream records:
  - server start/stop and worker lifecycle
  - HTTP request summaries
  - camera open attempts, read failures, and capture/detection heartbeats
  - periodic memory samples with RSS/HWM, total/native-vs-Python thread counts, FD count, GC counters, and `tracemalloc`
- Use `--memory-log-interval-sec` to tighten or relax memory sampling.
- Use `--auto-shutdown-sec` to force a short-lived live-camera session for safer repro attempts.
- Camera-open diagnostics also record the negotiated width, height, fps, and FOURCC observed after `VideoCapture.set()` so mode negotiation stays auditable.
- Memory diagnostics now record both total threads and Python threads, which makes native thread surplus explicit in JSONL logs.
- the Electron viewer now keeps a single polling timer, so `ASEE_VIEWER_POLL_INTERVAL_MS` maps cleanly to backend request rate.
- `repos/asee/electron` is now the official viewer surface; `tmp_main.sh` launches it with `ASEE_VIEWER_BACKEND_URL` and `ASEE_VIEWER_TITLE`.

## Initial Modules

- `asee.camera_layout`
  - `parse_camera_csv()`
  - `parse_v4l2_devices()`
  - `extend_with_optional_camera()`
  - `detect_v4l2_devices()`
  - `build_camera_csv()`
- `asee.biometric_status`
  - `BiometricStatusTracker`
- `asee.biometric_client`
  - `RemoteBiometricStatusClient`
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
- `asee.diagnostics`
  - `JsonlDiagnosticsLogger`
  - `MemoryMonitor`
  - `read_process_metrics()`
  - `build_default_diagnostics_log_path()`
- `asee.video_server`
  - `GodModeVideoServer`
  - `LiveCameraDisabledError`
  - `CaptureSettings`
  - `decode_fourcc_value()`
  - `encode_frame_to_jpeg()`
  - per-camera latest-only MJPEG chunk cache so multiple viewers reuse overlay/JPEG work
  - `resolve_opencv_threads()`
  - `resolve_capture_settings()`
  - no-camera-safe HTTP compatibility server on top of extracted modules
- `asee.overlay_data`
  - `FaceDetection`
  - `OverlayFrame`
- `asee.overlay_broadcaster`
  - `OverlayBroadcaster`
- `asee.webrtc_signaling`
  - `create_webrtc_app()`
  - staged `/offer` + compatibility `/cameras` `/overlay_text` `/status` `/biometric_status`
- `asee.webrtc_video_track`
  - `RuntimeVideoTrack`
  - reads frames/faces/overlay text directly from `SeeingServerRuntime`
- `asee.enroll_owner`
  - `fetch_frame_from_server()`
  - `run_enrollment()`
  - OWNER embedding capture from server snapshot or direct camera fallback

## Planned Next Slice

- add codec selection inside `repos/asee/python`
- keep Flask/MJPEG as the safe fallback until WebRTC parity is proven
- replace direct detector/classifier orchestration from `tmp/_trash/GOD_MODE` with `asee.overlay.GodModeOverlay`
- add a compatibility runtime adapter so `tmp/_trash/GOD_MODE` can call `asee` modules as the new source of truth
- keep image processing and biometric inference in Python instead of moving recognition logic into TypeScript

## Notes

- `tmp/_trash/GOD_MODE/god_mode_overlay.py` and `tmp/_trash/GOD_MODE/god_mode_video_server.py` can now act as compatibility wrappers over `asee`
- `tmp/_trash/GOD_MODE/god_mode_camera_layout.py` can also act as a compatibility wrapper over `asee.camera_layout`
- `tmp/_trash/GOD_MODE/god_mode_enroll_owner.py` can also act as a compatibility wrapper over `asee.enroll_owner`
- the future Electron UI belongs beside this backend, but not inside the CV/runtime core
- `repos/asee/tmp_main.sh` now owns the temporary GOD MODE-style start/stop/layout lifecycle, while `god_mode_predictor.py` remains intentionally outside the repo boundary
