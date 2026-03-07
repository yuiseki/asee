# ADR 001: asee — Technology Selection and Design

## Status

Proposed

## Context

`asee` is the **Eyes** of the AI agent body (see [yuiclaw ADR 002](../../yuiclaw/docs/ADR/002-sensorimotor-unix-philosophy.md)).

**Single Responsibility:** Capture webcam frames, describe their visual content via a multimodal LLM (Ollama), output text descriptions to stdout / acomm.

### PoC Reference

`/home/yuiseki/Workspaces/tmp/webcam_ollama_vision/`

The PoC (`describe_webcam_with_ollama.py`, `tmux_webcam_daemon.sh`) demonstrates the core pipeline. Problems:
- Daemon management (`tmux_webcam_daemon.sh`) is embedded — belongs in `abeat`
- No structured output format
- Python-only, no type safety

`asee` extracts only the **see** portion: webcam → frame → Ollama → text.

## Decision

### Language & Runtime

**Rust** — consistent with the `a*` ecosystem. Handles subprocess (`ffmpeg`), HTTP client (Ollama), and structured JSON output.

### Multi-Device Design Principle

**「1プロセス = 1デバイス」** を基本とする。複数デバイスを監視する場合は、複数の `asee` プロセスを独立して起動し、それぞれが `acomm` に publish する。プロセス間の調整は `acomm` と `abeat` が担い、`asee` 自身は他インスタンスの存在を関知しない。

```sh
# abeat が管理する起動例
asee --camera /dev/video0 --publish &   # 部屋カメラ1
asee --camera /dev/video2 --publish &   # 部屋カメラ2
```

各インスタンスの出力 JSON には `source` フィールドを含め、`acore` 側でどのデバイスからの情報かを区別できるようにする。

### Input Sources

#### MUST: 物理カメラ (`--camera`)

**`ffmpeg`** subprocess — extract a single frame from `/dev/videoN` as JPEG:

```sh
ffmpeg -y -f v4l2 -i /dev/video0 -vframes 1 -q:v 2 /tmp/asee_frame.jpg
```

- デバイスパス指定: `--camera /dev/video0`（デフォルト: `/dev/video0`）
- Frame resolution: configurable via `--width` / `--height` (default: 1280x720)
- Rationale: `ffmpeg` is already present on the system and battle-tested in the PoC

現在接続中の物理カメラ:
- `/dev/video0` — HD Pro Webcam C920 (1台目)
- `/dev/video2` — HD Pro Webcam C920 (2台目)

#### MAY: YouTube Live カメラ (`--youtube-cdp`) ※後回し

VacuumTube の CDP エンドポイントからスクリーンショットを取得する入力ソース。
優先度 **MAY** — 物理カメラ対応が安定した後に実装する。

```sh
# 将来的な使用例（未実装）
asee --youtube-cdp http://127.0.0.1:9993 --publish &  # VacuumTube instance 1
asee --youtube-cdp http://127.0.0.1:9994 --publish &  # VacuumTube instance 2
```

実装時の方針:
- CDP `Page.captureScreenshot` コマンドで JPEG 取得
- 入力ソース種別を問わず出力 JSON フォーマットは統一する

### Visual Recognition

**Ollama HTTP API** (`localhost:11434`):
- Model: `qwen3-vl:4b` (preferred, as established in webcam_vision skill) with fallback to `qwen2-vl:4b`
- Endpoint: `POST /api/generate` with base64-encoded image
- Prompt: Japanese description prompt (configurable via `--prompt`)
- `asee` does **not** embed Ollama. It is a client.

Default prompt:
```
この画像に映っている内容を日本語で簡潔に説明してください。
```

Diff-aware mode (daemon): when `--diff` flag is set, include previous description in prompt to highlight changes only.

### Output Format

**stdout** — one JSON line per observation:

```json
{"source": "/dev/video0", "source_type": "camera", "description": "デスクに座っている人物。モニターを見ている。", "timestamp": "2026-02-27T10:15:30+09:00", "model": "qwen3-vl:4b"}
```

`source` フィールドにより、複数インスタンスの出力を `acomm` や `acore` 側でデバイスごとに識別できる。

NDJSON enables direct piping:

```sh
asee --camera /dev/video0 | amem keep --kind activity
asee --camera /dev/video0 | acore
# 複数インスタンスの出力を acomm 経由でマージ
asee --camera /dev/video0 --publish &
asee --camera /dev/video2 --publish &
```

### Stitch Mode

When `--stitch horizontal|vertical` is set, combine frames from multiple `--camera` devices into a single image before sending to Ollama. This matches the behavior in the `webcam-vision-ollama` skill.

### Watch Mode (Daemon)

`--watch` flag enables periodic capture loop:
- Interval: `--interval <SECS>` (default: 60)
- Output: per-observation JSON lines to stdout (continuous)
- Cache: frame images written to `~/.cache/yuiclaw/camera/YYYY/MM/DD/HH/MM.{png,txt}`

The PoC's `tmux_webcam_daemon.sh` is replaced by an `abeat` job definition that runs `asee --watch`.

### acomm Integration

When `--publish` flag is set, publish each observation as a `VisualContext` event:

```json
{"type": "VisualContext", "camera": 0, "description": "...", "timestamp": "..."}
```

### CLI Interface

```
asee [OPTIONS]

Input Source (MUST specify one):
  --camera <PATH>        Physical camera device path (e.g. /dev/video0)

Input Source (MAY implement later):
  --youtube-cdp <URL>    [MAY] VacuumTube CDP endpoint for YouTube Live capture

Options:
  --stitch <DIR>         Stitch multi --camera frames: horizontal | vertical
  --stitch-only          Only send stitched image (skip individual descriptions)
  --model <NAME>         Ollama model name (default: qwen3-vl:4b)
  --ollama-url <URL>     Ollama API endpoint (default: http://localhost:11434)
  --prompt <TEXT>        Custom description prompt
  --watch                Continuous capture mode
  --interval <SECS>      Capture interval in watch mode (default: 60)
  --diff                 In watch mode, only describe changes from previous frame
  --publish              Also publish to acomm as VisualContext event
  --acomm-socket <PATH>  acomm socket path (default: /tmp/acomm.sock)
  --quiet                Suppress status messages
```

### Daemon Management

`asee` is **not** a daemon manager. Long-running watch mode is invoked by:
- `abeat` job definition (automated, scheduled)
- Direct `asee --watch` invocation (manual)

## Consequences

- `asee` becomes a composable UNIX source/filter: `asee | ...`
- **「1プロセス = 1デバイス」** により、インスタンスを増やすだけで監視対象を拡張できる
- Daemon lifecycle fully delegated to `abeat`
- VOICEVOX / command parsing completely absent from `asee`
- `source` フィールドにより複数インスタンスの出力を `acore` 側で識別可能
- Diff-aware mode reduces redundant context in long-running sessions

## Implementation Plan

### MUST（優先実装）

1. Scaffold Rust project with `clap` CLI
2. Implement `--camera <PATH>` with `ffmpeg` subprocess frame capture
3. Implement Ollama HTTP client with base64 image encoding
4. Implement single-shot NDJSON output with `source` / `source_type` fields
5. Implement `--stitch` image combination (call `ffmpeg` for montage)
6. Implement `--watch` interval loop with cache writing
7. Implement `--diff` context-aware prompting
8. Implement acomm `VisualContext` publish
9. Write unit tests with mock Ollama server

### MAY（後回し）

10. Implement `--youtube-cdp <URL>` input source via CDP `Page.captureScreenshot`
