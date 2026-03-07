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
- OpenCV-heavy capture / overlay / MJPEG generation stays in `tmp/GOD_MODE` until the runtime boundary is better isolated
- the future Electron viewer will consume `asee` backend outputs instead of owning recognition logic

## Consequences

- migration can begin with low-risk contract extraction and unit tests
- the Python backend remains testable without cameras, OpenCV, or a desktop session
- the HTTP shell can now be tested with Flask's in-process test client instead of a live threaded server
- backend-selection policy can evolve separately from the OpenCV drawing/runtime code
- the eventual split becomes:
  - Python backend in `repos/asee/python`
  - Electron viewer as a separate surface layer
