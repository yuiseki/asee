# ADR 001: Initial Python Backend Scaffold

## Status

Accepted

## Context

- `tmp/GOD_MODE` mixes camera capture, face recognition, HTTP serving, Chromium/PWA display, VOICEVOX, and shell orchestration
- the future replacement should use Electron for the visual surface, but image processing remains Python-first
- `god_mode_predictor.py` is currently dead code and should not shape the initial repository boundary

## Decision

- `repos/asee` starts as a Python package, not an Electron app
- the first extracted slice is limited to pure backend logic:
  - camera layout helpers
  - biometric status aggregation
  - static web shell asset builders
- OpenCV-heavy capture / overlay / HTTP runtime stays in `tmp/GOD_MODE` until the contract is better isolated
- the future Electron viewer will consume `asee` backend outputs instead of owning recognition logic

## Consequences

- migration can begin with low-risk contract extraction and unit tests
- the Python backend remains testable without cameras, OpenCV, or a desktop session
- the eventual split becomes:
  - Python backend in `asee`
  - Electron viewer as a separate surface layer
