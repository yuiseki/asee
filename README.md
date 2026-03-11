# asee

Agentic seeing split into a Python backend and an Electron viewer surface.

## Layout

- `python/`
- extracted Python backend from the legacy GOD MODE runtime, now archived under `tmp/_trash/GOD_MODE`
  - camera layout, biometric status, HTTP shell contract, DNN backend policy
  - face tracking and capture persistence primitives
  - OWNER policy and YuNet detection pipeline
  - rebuilt `GodModeOverlay` runtime on top of extracted primitives
  - rebuilt viewer/server state holder for the GOD MODE HTTP contract
  - rebuilt `GodModeVideoServer` compatibility server for camera-less, single-camera, and safety-limited multi-camera paths
  - rebuilt OWNER enrollment flow
  - staged WebRTC transport primitives for the next viewer/backend migration slice
    - overlay JSON payloads
    - overlay broadcaster
    - aiohttp + aiortc signaling app factory
    - runtime-backed WebRTC video track that reads `SeeingServerRuntime`
- `electron/`
  - Electron + React + TypeScript viewer
  - already reads the existing GOD MODE HTTP contract through a preload bridge
  - future home for the full GOD MODE desktop surface
- `docs/ADR/`
  - repository-level architecture decisions

## Commands

### Temporary operator launcher

```bash
./tmp_main.sh start --port 8765 --cameras 0,2,4,6 --capture-profile 720p --auto-shutdown-sec 30
./tmp_main.sh status --port 8765
./tmp_main.sh layout --port 8765 --left-bottom
./tmp_main.sh stop --port 8765
```

- `tmp_main.sh` is now the canonical temporary operator launcher for `repos/asee`.
- It starts `python -m asee.video_server` and the official `electron/` viewer together.
- The official Electron window caption is `ASEE Viewer`.
- viewer startup now builds once up front, then runs Electron through a lightweight supervisor so an unexpected viewer exit can be restarted without bouncing the backend.
- the viewer supervisor now reapplies the default left-bottom KWin layout after each viewer launch, so a respawned Electron window does not drift back to the desktop center.
- `ASEE_VIEWER_RESPAWN=0` disables that respawn loop when a bounded test wants the viewer to exit only once.
- GPU experiments now flow through `tmp_main.sh` into the Electron runner via env vars such as `ASEE_VIEWER_USE_GL=desktop`, `ASEE_VIEWER_USE_ANGLE=gl`, `ASEE_VIEWER_DISABLE_GPU_SANDBOX=1`, `ASEE_VIEWER_EXTRA_ARGS='--enable-logging=stderr --v=1'`, and PRIME offload hints like `__NV_PRIME_RENDER_OFFLOAD=1`, `__NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, `DRI_PRIME=1`.
- each viewer restart logs the effective GPU env values and resolved Electron args into `/tmp/asee_tmp_main_viewer_<port>.log`.
- `stop` now terminates backend/viewer by process group, tolerates stale pid files, and always removes `/tmp/asee_tmp_main_<port>.pids`.
- legacy wrapper flags such as `--chromium`, `--pwa-installing`, `--voice`, and `--ollama-vlm` are accepted as no-op compatibility shims while callers migrate.

### Python backend

```bash
cd python
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/mypy src
.venv/bin/ruff check
.venv/bin/python -m asee.video_server --port 8765
# live camera is opt-in only
.venv/bin/python -m asee.video_server --port 8765 --device 0 --allow-live-camera
# staged WebRTC transport over the same runtime state
.venv/bin/python -m asee.video_server --port 8765 --transport webrtc
# safer multi-camera repro: lower-risk defaults + bounded lifetime
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --auto-shutdown-sec 30
# extra isolation: disable face-detect workers while checking capture stability
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --disable-face-detect --auto-shutdown-sec 30
# bounded 720p multi-camera trial
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --capture-profile 720p --opencv-threads 1 --auto-shutdown-sec 30
# bounded live test window
.venv/bin/python -m asee.video_server --port 8765 --device 0 --allow-live-camera --auto-shutdown-sec 180
```

- MJPEG/Flask remains the default production path for now.
- A first WebRTC migration slice now exists in the Python package:
  - `asee.overlay_data`
  - `asee.overlay_broadcaster`
  - `asee.webrtc_signaling`
  - `asee.webrtc_video_track`
- These modules are intentionally additive.
- `asee.video_server --transport webrtc` now wires the staged signaling path to the same `SeeingServerRuntime`.
- MJPEG/Flask still remains the production default.

## Safety Policy

- `asee.video_server` now defaults to no-camera mode. `--device` defaults to `-1`, and live capture is blocked unless `--allow-live-camera` is present.
- single-camera runs keep the higher-fidelity default request: `1280x720 @ 30fps MJPG`.
- multi-camera runs now default to a lower-risk request: `640x360 @ 10fps MJPG`.
- `--capture-profile 720p` is the simplest explicit opt-in when both rendering and recognition should stay at `1280x720`; for multi-camera it keeps `10fps MJPG`.
- current OWNER enrollment still collects embeddings through a `1280x720` overlay path, so the lower-risk multi-camera profile trades recognition quality for stability.
- if recognition quality matters more than load, either re-enroll under the same runtime conditions or move detection/embedding back to a higher-resolution capture path.
- operators can override the requested capture mode explicitly with `--width`, `--height`, `--fps`, and `--fourcc`.
- multi-camera runs also cap OpenCV's internal worker pool to `1` thread by default; `--opencv-threads` can override this when a controlled benchmark needs it.
- YuNet, SFace, and `owner_embedding.npy` are now resolved only from `repos/asee/python/src/asee/models/`.
- local copies under `python/src/asee/models/` are intentionally gitignored so private biometric assets never get pushed by accident.
- risky detection load can be isolated with `--disable-face-detect`.
- This is intentional. Direct migration to real webcams stays disabled by default until memory behavior and crash forensics are good enough.

## Diagnostics

- `asee.video_server` now writes persistent JSONL diagnostics logs by default under `~/.local/state/asee/video-server/`.
- Each run also gets a sibling `.fault.log` file for `faulthandler`.
- The log includes:
  - startup/shutdown and worker lifecycle
  - HTTP request summaries
  - camera open attempts, read failures, and periodic capture heartbeats
  - periodic process memory samples, including RSS/HWM, total/native-vs-Python thread counts, FD count, GC counters, and `tracemalloc`
- Override the destination with `--diagnostic-log-path`, or tune sampling with `--memory-log-interval-sec`.
- For risky live-camera repros, `--auto-shutdown-sec 60..180` gives the process a bounded lifetime even if the operator forgets to stop it.
- `camera_opened` diagnostics now include the negotiated width, height, fps, and FOURCC returned by OpenCV so capture-mode mismatches remain visible in logs.
- `memory_sample` diagnostics now distinguish total thread count from Python thread count, making native thread blow-ups visible without attaching a debugger.

### Electron viewer

```bash
cd electron
npm install
npm test
npm run build
npm run demo
```

- the Electron viewer now keeps exactly one polling interval alive, so `ASEE_VIEWER_POLL_INTERVAL_MS` is honored instead of multiplying backend request load on each refresh.
- `electron/` is now the official viewer surface for `asee`; Chromium/PWA is no longer the target UI path.
- `tmp_main.sh` now prebuilds the viewer before launch, then runs the Electron process through `scripts/run-electron-with-x11-env.mjs --skip-build` so viewer crashes can be supervised separately from build failures.

## Migration Notes

- `tmp/_trash/GOD_MODE/god_mode_overlay.py` and `tmp/_trash/GOD_MODE/god_mode_video_server.py` can now be thin compatibility facades over `asee`
- `god_mode_predictor.py` stays excluded as dead code
- `repos/asee/tmp_main.sh` is now the canonical operator entrypoint that replaced the legacy `tmp/_trash/GOD_MODE/god_mode.sh` flow
- image processing remains Python-first
- desktop rendering moves toward Electron instead of Tauri/WebKitGTK
- `electron/` can already act as a read-only viewer for the current backend at `http://127.0.0.1:8765`
- `python/asee.overlay.GodModeOverlay` is now the target runtime for future `tmp/_trash/GOD_MODE` compatibility wrappers
- `python/asee.server_runtime.SeeingServerRuntime` is now the target state holder for future `god_mode_video_server.py` wrappers
- `python/asee.video_server.GodModeVideoServer` is now the migration target for camera-less, single-camera, and safety-limited multi-camera server behavior
- `python/asee.enroll_owner` is now the migration target for OWNER enrollment
- remaining migration focus is wiring existing callers over to `repos/asee/tmp_main.sh`, reducing native-memory risk further, and retiring the tmp wrapper once callers move
- until live capture stability is proven, `asee` treats no-camera mode and persistent diagnostics as the safe default
