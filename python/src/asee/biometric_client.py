"""Client helpers for consuming asee biometric status endpoints."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RemoteBiometricStatusClient:
    status_url: str
    timeout_sec: float = 1.5
    logger: Callable[[str], None] | None = None
    opener: Callable[..., Any] = urllib.request.urlopen

    def fetch_status(self) -> dict[str, Any] | None:
        url = self.status_url.strip()
        if not url:
            return None
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with self.opener(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            if self.logger is not None:
                self.logger(f"god mode biometric status fetch failed: {exc}")
            return None
        return data if isinstance(data, dict) else None

    def owner_face_absent_for_lock(self, *, absent_lock_sec: int) -> bool:
        status = self.fetch_status()
        return owner_face_absent_for_lock_from_status(status, absent_lock_sec=absent_lock_sec)

    def owner_face_recent_for_unlock(self, *, fresh_ms: int) -> bool:
        status = self.fetch_status()
        return owner_face_recent_for_unlock_from_status(status, fresh_ms=fresh_ms)


def owner_face_absent_for_lock_from_status(
    status: dict[str, Any] | None,
    *,
    absent_lock_sec: int,
) -> bool:
    if not isinstance(status, dict):
        return False
    if bool(status.get("ownerPresent")):
        return False
    age_ms = status.get("ownerSeenAgoMs")
    if age_ms is None:
        return True
    try:
        threshold_ms = max(0, int(absent_lock_sec) * 1000)
        return int(age_ms) >= threshold_ms
    except Exception:
        return False


def owner_face_recent_for_unlock_from_status(
    status: dict[str, Any] | None,
    *,
    fresh_ms: int,
) -> bool:
    if not isinstance(status, dict):
        return False
    if bool(status.get("ownerPresent")):
        return True
    age_ms = status.get("ownerSeenAgoMs")
    if age_ms is None:
        return False
    try:
        threshold_ms = max(0, int(fresh_ms))
        return int(age_ms) <= threshold_ms
    except Exception:
        return False


def resolve_remote_biometric_status_client(
    *,
    current_client: Any | None,
    status_url: str,
    logger: Callable[[str], None] | None = None,
    timeout_sec: float = 1.5,
    client_factory: Callable[..., Any] = RemoteBiometricStatusClient,
) -> Any | None:
    if current_client is not None:
        return current_client
    resolved_url = str(status_url or "").strip()
    if not resolved_url:
        return None
    return client_factory(
        status_url=resolved_url,
        timeout_sec=timeout_sec,
        logger=logger,
    )


def fetch_remote_biometric_status(
    *,
    current_client: Any | None,
    status_url: str,
    logger: Callable[[str], None] | None = None,
    timeout_sec: float = 1.5,
    client_factory: Callable[..., Any] = RemoteBiometricStatusClient,
) -> tuple[Any | None, dict[str, Any] | None]:
    client = resolve_remote_biometric_status_client(
        current_client=current_client,
        status_url=status_url,
        logger=logger,
        timeout_sec=timeout_sec,
        client_factory=client_factory,
    )
    if client is None:
        return None, None
    status = client.fetch_status()
    return client, status if isinstance(status, dict) else None
