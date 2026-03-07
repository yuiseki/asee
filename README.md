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
```

### Electron viewer

```bash
cd electron
npm install
npm test
npm run build
npm run demo
```

## Migration Notes

- `tmp/GOD_MODE` remains the runtime source of truth until compatibility adapters exist
- `god_mode_predictor.py` stays excluded as dead code
- image processing remains Python-first
- desktop rendering moves toward Electron instead of Tauri/WebKitGTK
- `electron/` can already act as a read-only viewer for the current backend at `http://127.0.0.1:8765`
- `python/asee.overlay.GodModeOverlay` is now the target runtime for future `tmp/GOD_MODE` compatibility wrappers
- `python/asee.server_runtime.SeeingServerRuntime` is now the target state holder for future `god_mode_video_server.py` wrappers
