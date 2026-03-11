# asee/electron

Electron + React + TypeScript viewer for the future GOD MODE replacement.

## Commands

```bash
npm install
npm test
npm run build
npm run demo
npm run start
```

## Scope

- desktop viewer shell only
- no image processing logic
- consumes backend outputs from `../python` through a preload bridge
- currently supports the canonical `asee` / legacy-GOD-MODE-compatible HTTP contract as a viewer
- stages both MJPEG tiles and WebRTC `<video> + canvas` tiles behind the same preload snapshot contract
- launches as a frameless desktop surface so KDE title bars do not eat into the camera grid
- `scripts/run-electron-with-x11-env.mjs` now supports `--skip-build` so operator wrappers can separate build failures from runtime crashes
- `ASEE_VIEWER_DISABLE_GPU=1` or `--disable-gpu` can be used for GPU-related crash isolation

## Current Backend Contract

- `ASEE_VIEWER_BACKEND_URL`
  - default: `http://127.0.0.1:8765`
- `ASEE_VIEWER_POLL_INTERVAL_MS`
  - default: `2000`
- `ASEE_VIEWER_RESPAWN`
  - used by `../tmp_main.sh`
  - default: `1`
  - set `0` to disable viewer auto-restart during bounded tests
- consumes:
  - `/cameras`
  - `/overlay_text`
  - `/status`
  - `/biometric_status`
  - `/stream` and `/stream/<camera>`
  - `POST /offer` when `/status.transport === "webrtc"`
- `npm run demo` uses synthetic data via `ASEE_VIEWER_AUTODEMO=1`

## Transport Behavior

- `/status.transport` selects the viewer path
  - `mjpeg`
    - render `<img src="/stream/...">`
  - `webrtc`
    - open `RTCPeerConnection`
    - request staged tracks through `POST /offer`
    - paint face boxes on a sibling `<canvas>`
- snapshot polling stays active in both modes so status, caption, and biometric text remain driven by the preload bridge
