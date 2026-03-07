# asee/electron

Electron + React + TypeScript viewer for the future GOD MODE replacement.

## Commands

```bash
npm install
npm test
npm run build
npm run demo
```

## Scope

- desktop viewer shell only
- no image processing logic
- consumes backend outputs from `../python` through a preload bridge
- currently supports the existing `tmp/GOD_MODE` HTTP contract as a read-only viewer
- launches as a frameless desktop surface so KDE title bars do not eat into the camera grid

## Current Backend Contract

- `ASEE_VIEWER_BACKEND_URL`
  - default: `http://127.0.0.1:8765`
- `ASEE_VIEWER_POLL_INTERVAL_MS`
  - default: `2000`
- consumes:
  - `/cameras`
  - `/overlay_text`
  - `/status`
  - `/biometric_status`
  - `/stream` and `/stream/<camera>`
- `npm run demo` uses synthetic data via `ASEE_VIEWER_AUTODEMO=1`
