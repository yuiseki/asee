# ADR 001: Initial Python Backend Scaffold

## Status

Accepted

## Context

- `tmp/GOD_MODE` mixes camera capture, face recognition, HTTP serving, Chromium/PWA display, VOICEVOX, and shell orchestration
- the future replacement should use Electron for the visual surface, but image processing remains Python-first
- `god_mode_predictor.py` is currently dead code and should not shape the initial repository boundary

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
- OpenCV-heavy camera capture / MJPEG generation still stays in `tmp/GOD_MODE` until the runtime boundary is better isolated
- the future Electron viewer will consume `asee` backend outputs instead of owning recognition logic

## Consequences

- migration can begin with low-risk contract extraction and unit tests
- the Python backend remains testable without cameras, OpenCV, or a desktop session
- the HTTP shell can now be tested with Flask's in-process test client instead of a live threaded server
- backend-selection policy can evolve separately from the OpenCV drawing/runtime code
- overlay runtime code can now be rebuilt on top of `asee` primitives instead of `tmp/GOD_MODE` internals
- YuNet detector orchestration can now be composed in `asee` without dragging the full overlay class across at once
- `tmp/GOD_MODE` can now migrate toward a thin runtime adapter that delegates overlay behavior to `asee.overlay`
- `tmp/GOD_MODE` can also delegate state management and HTTP contract glue to `asee.server_runtime`
- `tmp/GOD_MODE/god_mode_video_server.py` now has a clear migration target in `asee.video_server`
- compatibility wrappers for `god_mode_overlay.py` and `god_mode_video_server.py` can already re-export `asee` implementations while preserving the existing tmp-facing contract
- `god_mode_enroll_owner.py` can also move behind an `asee.enroll_owner` compatibility facade
- the eventual split becomes:
  - Python backend in `repos/asee/python`
  - Electron viewer as a separate surface layer
