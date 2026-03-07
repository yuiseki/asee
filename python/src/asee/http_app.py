"""Flask HTTP shell extracted from the GOD MODE video server."""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from flask import Flask, Response, jsonify, request

from .biometric_status import BiometricStatusValue
from .web_shell import (
    build_icon_svg,
    build_service_worker_script,
    build_web_manifest,
)

DEFAULT_INDEX_HTML = """<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#001a12" />
    <link rel="manifest" href="/manifest.webmanifest" />
    <title>GOD MODE</title>
  </head>
  <body>
    <main id="app"></main>
    <script>
      if ("serviceWorker" in navigator) {
        window.addEventListener("load", () => {
          navigator.serviceWorker.register("/service-worker.js").catch(() => {});
        });
      }
    </script>
  </body>
</html>
"""

type BiometricStatusPayload = Mapping[str, BiometricStatusValue]
type StreamFactory = Callable[[int | None], Iterable[bytes] | None]


@dataclass(slots=True)
class OverlayTextState:
    """Mutable caption/prediction state for the viewer shell."""

    caption: str = ""
    prediction: str = ""


class SeeingHttpRuntime(Protocol):
    """Minimal runtime surface needed by the extracted Flask app."""

    title: str
    overlay_state: OverlayTextState
    is_running: bool
    camera_ids: Sequence[int]

    def update_overlay_text(self, *, caption: str = "", prediction: str = "") -> None: ...
    def get_biometric_status(self) -> BiometricStatusPayload: ...
    def get_snapshot_jpeg(self) -> bytes | None: ...
    def iter_mjpeg(self, device: int | None = None) -> Iterable[bytes] | None: ...


@dataclass(slots=True)
class InMemoryHttpRuntime:
    """Small in-memory runtime used by tests and compatibility shims."""

    title: str = "GOD MODE"
    overlay_state: OverlayTextState = field(default_factory=OverlayTextState)
    is_running: bool = False
    camera_ids: tuple[int, ...] = ()
    biometric_status: dict[str, BiometricStatusValue] = field(
        default_factory=lambda: {
            "running": False,
            "ownerEmbeddingLoaded": False,
            "ownerPresent": False,
            "ownerCount": 0,
            "subjectCount": 0,
            "peopleCount": 0,
            "ownerSeenAgoMs": None,
            "updatedAt": 0.0,
        }
    )
    snapshot_jpeg: bytes | None = None
    stream_factory: StreamFactory | None = None

    def update_overlay_text(self, *, caption: str = "", prediction: str = "") -> None:
        self.overlay_state = OverlayTextState(caption=caption, prediction=prediction)

    def get_biometric_status(self) -> BiometricStatusPayload:
        return dict(self.biometric_status)

    def get_snapshot_jpeg(self) -> bytes | None:
        return self.snapshot_jpeg

    def iter_mjpeg(self, device: int | None = None) -> Iterable[bytes] | None:
        if self.stream_factory is None:
            return None
        return self.stream_factory(device)


def render_index_html(title: str, template: str = DEFAULT_INDEX_HTML) -> str:
    """Inject the runtime title into the static shell HTML."""
    escaped_title = html.escape(title, quote=False)
    if "<title>GOD MODE</title>" in template:
        return template.replace("<title>GOD MODE</title>", f"<title>{escaped_title}</title>")
    return template.replace("__TITLE__", escaped_title)


def create_http_app(
    runtime: SeeingHttpRuntime,
    *,
    index_html: str = DEFAULT_INDEX_HTML,
) -> Flask:
    """Build the extracted Flask shell around an injectable runtime."""
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return render_index_html(runtime.title, template=index_html)

    @app.route("/manifest.webmanifest")
    def manifest() -> Response:
        return Response(
            json.dumps(build_web_manifest(runtime.title), ensure_ascii=False),
            mimetype="application/manifest+json",
        )

    @app.route("/service-worker.js")
    def service_worker() -> Response:
        return Response(
            build_service_worker_script(),
            mimetype="application/javascript",
        )

    @app.route("/icon.svg")
    def icon() -> Response:
        return Response(
            build_icon_svg(runtime.title),
            mimetype="image/svg+xml",
        )

    @app.route("/update", methods=["POST"])
    def update() -> Response:
        data = request.get_json(silent=True) or {}
        caption = str(data.get("caption", ""))
        prediction = str(data.get("prediction", ""))
        runtime.update_overlay_text(caption=caption, prediction=prediction)
        return jsonify({"status": "ok"})

    @app.route("/cameras")
    def cameras() -> Response:
        return jsonify({"cameras": list(runtime.camera_ids)})

    @app.route("/stream")
    def stream() -> Response | tuple[str, int]:
        stream_iter = runtime.iter_mjpeg()
        if stream_iter is None:
            return ("Streaming not implemented", 501)
        return Response(
            stream_iter,
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/stream/<int:device>")
    def stream_device(device: int) -> Response | tuple[str, int]:
        stream_iter = runtime.iter_mjpeg(device)
        if stream_iter is None:
            return ("Streaming not implemented", 501)
        return Response(
            stream_iter,
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/snapshot")
    def snapshot() -> Response | tuple[str, int]:
        jpeg = runtime.get_snapshot_jpeg()
        if jpeg is None:
            return ("No frame", 503)
        return Response(jpeg, mimetype="image/jpeg")

    @app.route("/overlay_text")
    def overlay_text() -> Response:
        return jsonify(
            {
                "caption": runtime.overlay_state.caption,
                "prediction": runtime.overlay_state.prediction,
            }
        )

    @app.route("/status")
    def status() -> Response:
        return jsonify({"running": runtime.is_running})

    @app.route("/biometric_status")
    def biometric_status() -> Response:
        return jsonify(dict(runtime.get_biometric_status()))

    return app
