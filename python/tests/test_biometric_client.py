from __future__ import annotations

import json

from asee import (
    RemoteBiometricStatusClient,
    fetch_remote_biometric_status,
    owner_face_absent_for_lock_from_status,
    owner_face_recent_for_unlock_from_status,
    resolve_remote_biometric_status_client,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def read(self) -> bytes:
        return self._payload


def test_fetch_status_returns_decoded_json_payload() -> None:
    client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": True, "ownerSeenAgoMs": 100}).encode("utf-8")
        ),
    )

    status = client.fetch_status()

    assert status == {"ownerPresent": True, "ownerSeenAgoMs": 100}


def test_fetch_status_returns_none_and_logs_on_failure() -> None:
    logs: list[str] = []
    client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        logger=logs.append,
        opener=lambda request, timeout: (_ for _ in ()).throw(RuntimeError("down")),  # noqa: ARG005
    )

    status = client.fetch_status()

    assert status is None
    assert logs == ["god mode biometric status fetch failed: down"]


def test_owner_face_absent_for_lock_requires_threshold() -> None:
    client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": False, "ownerSeenAgoMs": 130_000}).encode("utf-8")
        ),
    )

    assert client.owner_face_absent_for_lock(absent_lock_sec=120) is True
    assert client.owner_face_absent_for_lock(absent_lock_sec=180) is False


def test_owner_face_absent_for_lock_is_false_when_owner_present() -> None:
    client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": True, "ownerSeenAgoMs": 999_999}).encode("utf-8")
        ),
    )

    assert client.owner_face_absent_for_lock(absent_lock_sec=120) is False


def test_owner_face_absent_for_lock_is_false_during_status_warmup() -> None:
    client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": False}).encode("utf-8")
        ),
    )

    assert client.owner_face_absent_for_lock(absent_lock_sec=120) is False


def test_owner_face_recent_for_unlock_accepts_present_or_fresh_owner() -> None:
    present_client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": True, "ownerSeenAgoMs": 9_999}).encode("utf-8")
        ),
    )
    fresh_client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": False, "ownerSeenAgoMs": 1_500}).encode("utf-8")
        ),
    )

    assert present_client.owner_face_recent_for_unlock(fresh_ms=2_000) is True
    assert fresh_client.owner_face_recent_for_unlock(fresh_ms=2_000) is True


def test_owner_face_recent_for_unlock_rejects_stale_or_missing_owner() -> None:
    stale_client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": False, "ownerSeenAgoMs": 3_000}).encode("utf-8")
        ),
    )
    missing_age_client = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": False}).encode("utf-8")
        ),
    )

    assert stale_client.owner_face_recent_for_unlock(fresh_ms=2_000) is False
    assert missing_age_client.owner_face_recent_for_unlock(fresh_ms=2_000) is False


def test_resolve_remote_biometric_status_client_reuses_existing_client() -> None:
    current = object()

    client = resolve_remote_biometric_status_client(
        current_client=current,
        status_url="http://127.0.0.1:8765/biometric_status",
    )

    assert client is current


def test_resolve_remote_biometric_status_client_builds_client_when_url_is_present() -> None:
    created: list[dict[str, object]] = []
    logs: list[str] = []

    def factory(**kwargs: object) -> object:
        created.append(kwargs)
        return {"client": True}

    client = resolve_remote_biometric_status_client(
        current_client=None,
        status_url=" http://127.0.0.1:8765/biometric_status ",
        timeout_sec=2.0,
        logger=logs.append,
        client_factory=factory,
    )

    assert client == {"client": True}
    assert created == [
        {
            "status_url": "http://127.0.0.1:8765/biometric_status",
            "timeout_sec": 2.0,
            "logger": logs.append,
        }
    ]


def test_resolve_remote_biometric_status_client_returns_none_for_blank_url() -> None:
    client = resolve_remote_biometric_status_client(
        current_client=None,
        status_url="   ",
    )

    assert client is None


def test_fetch_remote_biometric_status_reuses_client_and_returns_status() -> None:
    current = RemoteBiometricStatusClient(
        status_url="http://127.0.0.1:8765/biometric_status",
        opener=lambda request, timeout: _FakeResponse(  # noqa: ARG005
            json.dumps({"ownerPresent": True}).encode("utf-8")
        ),
    )

    client, status = fetch_remote_biometric_status(
        current_client=current,
        status_url="http://127.0.0.1:8765/biometric_status",
    )

    assert client is current
    assert status == {"ownerPresent": True}


def test_fetch_remote_biometric_status_returns_none_when_no_url_is_available() -> None:
    client, status = fetch_remote_biometric_status(
        current_client=None,
        status_url=" ",
    )

    assert client is None
    assert status is None


def test_owner_face_absent_for_lock_from_status_applies_threshold() -> None:
    assert owner_face_absent_for_lock_from_status(
        {"ownerPresent": False, "ownerSeenAgoMs": 130_000},
        absent_lock_sec=120,
    ) is True
    assert owner_face_absent_for_lock_from_status(
        {"ownerPresent": False, "ownerSeenAgoMs": 30_000},
        absent_lock_sec=120,
    ) is False


def test_owner_face_absent_for_lock_from_status_is_false_when_age_is_missing() -> None:
    assert owner_face_absent_for_lock_from_status(
        {"ownerPresent": False},
        absent_lock_sec=120,
    ) is False


def test_owner_face_recent_for_unlock_from_status_accepts_present_or_fresh_owner() -> None:
    assert owner_face_recent_for_unlock_from_status(
        {"ownerPresent": True, "ownerSeenAgoMs": 9_999},
        fresh_ms=2_000,
    ) is True
    assert owner_face_recent_for_unlock_from_status(
        {"ownerPresent": False, "ownerSeenAgoMs": 1_500},
        fresh_ms=2_000,
    ) is True
