"""Unit tests for the extracted GOD MODE HTTP contract."""

from __future__ import annotations

from asee.http_app import InMemoryHttpRuntime, OverlayTextState, create_http_app


def make_client(runtime: InMemoryHttpRuntime):
    app = create_http_app(runtime)
    return app.test_client()


def test_root_returns_html_shell_with_title_and_pwa_assets():
    runtime = InMemoryHttpRuntime(title="GOD MODE TEST")

    response = make_client(runtime).get("/")

    assert response.status_code == 200
    assert "text/html" in response.content_type
    body = response.get_data(as_text=True)
    assert "<title>GOD MODE TEST</title>" in body
    assert 'rel="manifest"' in body
    assert "serviceWorker.register" in body


def test_manifest_endpoint_uses_runtime_title():
    runtime = InMemoryHttpRuntime(title="GOD MODE TEST")

    response = make_client(runtime).get("/manifest.webmanifest")

    assert response.status_code == 200
    assert "application/manifest+json" in response.content_type
    assert response.get_json() == {
        "name": "GOD MODE TEST",
        "short_name": "GOD MODE",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#000000",
        "theme_color": "#001a12",
        "description": "Local GOD MODE monitoring overlay.",
        "icons": [
            {
                "src": "/icon.svg",
                "sizes": "512x512",
                "type": "image/svg+xml",
                "purpose": "any maskable",
            }
        ],
    }


def test_update_endpoint_updates_overlay_text():
    runtime = InMemoryHttpRuntime()

    response = make_client(runtime).post(
        "/update",
        json={"caption": "観測テスト", "prediction": "予測テスト"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert runtime.overlay_state == OverlayTextState(
        caption="観測テスト",
        prediction="予測テスト",
    )


def test_overlay_text_endpoint_reflects_current_overlay_state():
    runtime = InMemoryHttpRuntime(
        overlay_state=OverlayTextState(caption="現在の字幕", prediction="現在の推測"),
    )

    response = make_client(runtime).get("/overlay_text")

    assert response.status_code == 200
    assert response.get_json() == {
        "caption": "現在の字幕",
        "prediction": "現在の推測",
    }


def test_status_endpoint_reports_running_flag():
    runtime = InMemoryHttpRuntime(is_running=True)

    response = make_client(runtime).get("/status")

    assert response.status_code == 200
    assert response.get_json() == {"running": True}


def test_biometric_status_endpoint_passes_payload_through():
    runtime = InMemoryHttpRuntime(
        biometric_status={
            "running": True,
            "ownerEmbeddingLoaded": True,
            "ownerPresent": True,
            "ownerCount": 1,
            "subjectCount": 0,
            "peopleCount": 1,
            "ownerSeenAgoMs": 42,
            "updatedAt": 1234.5,
        }
    )

    response = make_client(runtime).get("/biometric_status")

    assert response.status_code == 200
    assert response.get_json() == runtime.biometric_status


def test_cameras_endpoint_reports_camera_ids():
    runtime = InMemoryHttpRuntime(camera_ids=(0, 2, 4, 6))

    response = make_client(runtime).get("/cameras")

    assert response.status_code == 200
    assert response.get_json() == {"cameras": [0, 2, 4, 6]}


def test_snapshot_returns_503_without_frame():
    runtime = InMemoryHttpRuntime(snapshot_jpeg=None)

    response = make_client(runtime).get("/snapshot")

    assert response.status_code == 503
    assert response.get_data(as_text=True) == "No frame"


def test_snapshot_returns_jpeg_bytes_when_available():
    runtime = InMemoryHttpRuntime(snapshot_jpeg=b"\xff\xd8\xff")

    response = make_client(runtime).get("/snapshot")

    assert response.status_code == 200
    assert response.content_type == "image/jpeg"
    assert response.data == b"\xff\xd8\xff"


def test_stream_returns_not_implemented_without_generator():
    runtime = InMemoryHttpRuntime()

    response = make_client(runtime).get("/stream")

    assert response.status_code == 501
    assert response.get_data(as_text=True) == "Streaming not implemented"
