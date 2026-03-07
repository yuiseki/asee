# asee

Agentic seeing split into a Python backend and an Electron viewer surface.

## Layout

- `python/`
  - extracted Python backend from `tmp/GOD_MODE`
  - camera layout, biometric status, HTTP shell contract, DNN backend policy
  - face tracking and capture persistence primitives
  - OWNER policy and YuNet detection pipeline
  - rebuilt `GodModeOverlay` runtime on top of extracted primitives
  - rebuilt viewer/server state holder for the GOD MODE HTTP contract
  - rebuilt `GodModeVideoServer` compatibility server for camera-less, single-camera, and safety-limited multi-camera paths
  - rebuilt OWNER enrollment flow
- `electron/`
  - Electron + React + TypeScript viewer
  - already reads the existing GOD MODE HTTP contract through a preload bridge
  - future home for the full GOD MODE desktop surface
- `docs/ADR/`
  - repository-level architecture decisions

## Commands

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
# safer multi-camera repro: lower-risk defaults + bounded lifetime
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --auto-shutdown-sec 30
# extra isolation: disable face-detect workers while checking capture stability
.venv/bin/python -m asee.video_server --port 8765 --cameras 0,2,4,6 --allow-live-camera --disable-face-detect --auto-shutdown-sec 30
# bounded live test window
.venv/bin/python -m asee.video_server --port 8765 --device 0 --allow-live-camera --auto-shutdown-sec 180
```

## Safety Policy

- `asee.video_server` now defaults to no-camera mode. `--device` defaults to `-1`, and live capture is blocked unless `--allow-live-camera` is present.
- single-camera runs keep the higher-fidelity default request: `1280x720 @ 30fps MJPG`.
- multi-camera runs now default to a lower-risk request: `640x360 @ 10fps MJPG`.
- operators can override the requested capture mode explicitly with `--width`, `--height`, `--fps`, and `--fourcc`.
- multi-camera runs also cap OpenCV's internal worker pool to `1` thread by default; `--opencv-threads` can override this when a controlled benchmark needs it.
- when `repos/asee/python/src/asee/models/` is empty, `asee` now falls back to `tmp/GOD_MODE/models/` for YuNet, SFace, and `owner_embedding.npy`.
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

## Migration Notes

- `tmp/GOD_MODE/god_mode_overlay.py` and `tmp/GOD_MODE/god_mode_video_server.py` can now be thin compatibility facades over `asee`
- `god_mode_predictor.py` stays excluded as dead code
- image processing remains Python-first
- desktop rendering moves toward Electron instead of Tauri/WebKitGTK
- `electron/` can already act as a read-only viewer for the current backend at `http://127.0.0.1:8765`
- `python/asee.overlay.GodModeOverlay` is now the target runtime for future `tmp/GOD_MODE` compatibility wrappers
- `python/asee.server_runtime.SeeingServerRuntime` is now the target state holder for future `god_mode_video_server.py` wrappers
- `python/asee.video_server.GodModeVideoServer` is now the migration target for camera-less, single-camera, and safety-limited multi-camera server behavior
- `python/asee.enroll_owner` is now the migration target for OWNER enrollment
- remaining migration focus is multi-camera/live runtime glue, reducing native-memory risk, and replacing the current backend host with `asee/python`
- until live capture stability is proven, `asee` treats no-camera mode and persistent diagnostics as the safe default
