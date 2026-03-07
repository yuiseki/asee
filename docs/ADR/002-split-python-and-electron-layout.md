# ADR 002: Split Python Backend and Electron Surface

## Status

Accepted

## Context

- `repos/asee` needs to host both Python-first image processing and an Electron desktop surface
- mixing Python package files and Electron build files at repo root makes the migration path harder to follow
- `repos/acaption` and `repos/asec` already proved that Electron + React + TypeScript is the right UI stack for this environment

## Decision

- keep `repos/asee/python` as the Python package root
- add `repos/asee/electron` as the Electron + React + TypeScript viewer root
- keep repository-level ADRs in `repos/asee/docs/ADR`
- reuse proven Electron launch patterns from `acaption` and `asec`, but keep `asee/electron` limited to scaffold-level responsibilities until the backend bridge is defined

## Consequences

- Python and Electron can evolve with their own toolchains without stepping on each other
- future CV/runtime extraction can continue under `python/` while UI work lands under `electron/`
- the repository structure now matches the intended long-term architecture before heavier migration work begins
