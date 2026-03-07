from asee.web_shell import (
    build_icon_svg,
    build_service_worker_script,
    build_web_manifest,
)


def test_build_web_manifest_uses_title_and_standalone_defaults() -> None:
    manifest = build_web_manifest("GOD MODE 18771")

    assert manifest["name"] == "GOD MODE 18771"
    assert manifest["short_name"] == "GOD MODE"
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "/"
    assert manifest["icons"]


def test_build_service_worker_script_skips_stream_and_snapshot_endpoints() -> None:
    script = build_service_worker_script()

    assert "/stream" in script
    assert "/snapshot" in script
    assert "serviceWorker" not in script


def test_build_icon_svg_escapes_title() -> None:
    icon = build_icon_svg("GOD MODE <OWNER>")

    assert "&lt;OWNER&gt;" in icon
    assert "<OWNER>" not in icon
    assert "GM" in icon
