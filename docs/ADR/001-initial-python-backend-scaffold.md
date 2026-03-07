# ADR 001: Initial Python Backend Scaffold

## Status

Accepted

## Context

- `tmp/GOD_MODE` mixes camera capture, face recognition, HTTP serving, Chromium/PWA display, VOICEVOX, and shell orchestration
- the future replacement should use Electron for the visual surface, but image processing remains Python-first
- `god_mode_predictor.py` is currently dead code and should not shape the initial repository boundary
- `tmp/GOD_MODE/god_mode.sh` should migrate into `repos/asee/tmp_main.sh`, while `repos/asee/electron` becomes the official viewer surface

## Decision

- `repos/asee` starts as a Python package, not an Electron app
- this initial Python package now lives under `repos/asee/python`
- the first extracted slice is limited to pure backend logic:
  - camera layout helpers
  - biometric status aggregation
  - static web shell asset builders
- the second extracted slice adds a Flask app factory for the non-OpenCV HTTP shell contract:
  - `/`
  - `/manifest.webmanifest`
  - `/service-worker.js`
  - `/icon.svg`
  - `/update`
  - `/cameras`
  - `/snapshot`
  - `/overlay_text`
  - `/status`
  - `/biometric_status`
- the third extracted slice adds pure DNN backend policy helpers from `god_mode_overlay.py`
- the fourth extracted slice adds pure overlay-support primitives from `god_mode_overlay.py`
  - `FaceBox`
  - `FaceTracker`
  - `FaceCaptureWriter`
- the fifth extracted slice adds pure detection orchestration from `god_mode_overlay.py`
  - OWNER selection policy
  - YuNet detection normalization and square-box expansion
- the sixth extracted slice rebuilds the overlay runtime itself inside `asee.overlay`
  - `GodModeOverlay`
  - DNN backend selection
  - overlay drawing
  - detector/classifier orchestration
  - face-capture writer integration
- the seventh extracted slice rebuilds the viewer/server state holder inside `asee.server_runtime`
  - `SeeingServerRuntime`
  - overlay text state propagation
  - owner embedding loading
  - biometric status aggregation
  - snapshot/stream contract glue for the extracted Flask app
- the eighth extracted slice rebuilds the GOD MODE compatibility server inside `asee.video_server`
  - `GodModeVideoServer`
  - `encode_frame_to_jpeg()`
  - camera-less and single-camera HTTP compatibility behavior
  - MJPG camera-open policy and frame normalization helpers
- the ninth extracted slice rebuilds OWNER enrollment inside `asee.enroll_owner`
  - server snapshot fetch
  - direct camera fallback
  - embedding collection and persistence
- the tenth extracted slice hardens the extracted runtime for safe operation before more live-camera migration
  - `asee.video_server` defaults to no-camera mode
  - live webcam access requires an explicit `--allow-live-camera` opt-in
  - single-camera mode keeps a higher-fidelity default request (`1280x720 @ 30fps MJPG`)
  - multi-camera mode now drops to a lower-risk default request (`640x360 @ 10fps MJPG`)
  - an explicit `--capture-profile 720p` path exists for the simpler "recognition and rendering both stay 720p" choice; multi-camera keeps `10fps MJPG`
  - OWNER enrollment still runs through a `1280x720` overlay path, so the low-risk multi-camera profile intentionally trades some recognition fidelity for crash resistance
  - decoupling detection/embedding resolution from rendered stream resolution remains possible later, but it is no longer required for the first accuracy recovery step
  - capture-mode overrides are explicit via `--width`, `--height`, `--fps`, and `--fourcc`
  - multi-camera mode also caps OpenCV's internal worker pool to a single thread unless explicitly overridden
  - detector pressure can be removed with `--disable-face-detect`
  - risky repro runs can be bounded with `--auto-shutdown-sec`
  - persistent JSONL diagnostics and `faulthandler` logs are enabled for CLI launches
  - periodic memory samples track RSS/HWM, open FDs, total/native-vs-Python thread counts, GC counters, and `tracemalloc`
  - HTTP requests and camera worker lifecycle are logged for crash reconstruction
  - negotiated capture width/height/fps/FOURCC are logged after camera-open to expose driver-level fallback or mismatch
  - until model assets are migrated physically, the extracted runtime reuses `tmp/GOD_MODE/models/` as the fallback source of YuNet, SFace, and owner embeddings
  - local copies in `python/src/asee/models/` are treated as private operator state and excluded from Git history
- the eleventh extracted slice moves the temporary operator launcher into `repos/asee/tmp_main.sh`
  - `tmp_main.sh` becomes the canonical GOD MODE-style start/stop/restart/status/layout entrypoint during migration
  - it launches `python -m asee.video_server` together with `repos/asee/electron`
  - legacy wrapper flags such as `--chromium`, `--pwa-installing`, `--voice`, and `--ollama-vlm` remain accepted as compatibility no-ops
  - `stop` cleans up backend/viewer by process group, tolerates stale pid files, and always removes the launcher pid file
- OpenCV-heavy camera capture / MJPEG generation still stays in `tmp/GOD_MODE` until the runtime boundary is better isolated
- the future Electron viewer will consume `asee` backend outputs instead of owning recognition logic
- the Electron viewer keeps one polling interval alive so backend request rate scales with `ASEE_VIEWER_POLL_INTERVAL_MS` instead of React refresh count

## Consequences

- migration can begin with low-risk contract extraction and unit tests
- the Python backend remains testable without cameras, OpenCV, or a desktop session
- the default operational mode is now intentionally conservative: no-camera unless explicitly unlocked
- multi-camera live migration is intentionally conservative as well: lower requested resolution and FPS unless the operator overrides them on purpose
- multi-camera live migration now also reduces OpenCV internal parallelism by default to cut native thread pressure
- the HTTP shell can now be tested with Flask's in-process test client instead of a live threaded server
- backend-selection policy can evolve separately from the OpenCV drawing/runtime code
- overlay runtime code can now be rebuilt on top of `asee` primitives instead of `tmp/GOD_MODE` internals
- YuNet detector orchestration can now be composed in `asee` without dragging the full overlay class across at once
- `tmp/GOD_MODE` can now migrate toward a thin runtime adapter that delegates overlay behavior to `asee.overlay`
- `tmp/GOD_MODE` can also delegate state management and HTTP contract glue to `asee.server_runtime`
- `tmp/GOD_MODE/god_mode_video_server.py` now has a clear migration target in `asee.video_server`
- compatibility wrappers for `god_mode_overlay.py` and `god_mode_video_server.py` can already re-export `asee` implementations while preserving the existing tmp-facing contract
- `god_mode_enroll_owner.py` can also move behind an `asee.enroll_owner` compatibility facade
- `tmp/GOD_MODE/god_mode.sh` now has a concrete migration target in `repos/asee/tmp_main.sh`
- Electron is now the official viewer route for migrated runs, not an auxiliary read-only shell
- bounded auto-shutdown runs no longer require manual pid file cleanup before the next `start`
- crash reproduction now leaves a persistent forensic trail even when automatic tests cannot cover the failure mode
- safer staged experiments are now possible by disabling face-detect workers and arming automatic shutdown before touching real hardware
- the eventual split becomes:
  - Python backend in `repos/asee/python`
  - Electron viewer plus temporary operator launcher in `repos/asee`
