# asee

Agentic seeing split into a Python backend and an Electron viewer surface.

## Layout

- `python/`
  - extracted Python backend from `tmp/GOD_MODE`
  - camera layout, biometric status, HTTP shell contract, DNN backend policy
- `electron/`
  - Electron + React + TypeScript viewer scaffold
  - future home for the GOD MODE desktop surface
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
