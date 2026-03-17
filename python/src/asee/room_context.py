"""Best-effort room context providers for face-capture metadata."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

type CommandResult = subprocess.CompletedProcess[str] | Any
type CommandRunner = Callable[[list[str]], CommandResult]
type RoomContextPayload = dict[str, Any]


def _default_command_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=5.0,
    )


class SwitchBotRoomContextProvider:
    """Fetch and cache room context from SwitchBot device status commands."""

    def __init__(
        self,
        *,
        motion_sensor_name: str | None,
        meter_name: str | None,
        ttl_sec: float = 5.0,
        command_runner: CommandRunner | None = None,
        monotonic: Callable[[], float] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._motion_sensor_name = _normalize_device_name(motion_sensor_name)
        self._meter_name = _normalize_device_name(meter_name)
        self._ttl_sec = max(0.0, float(ttl_sec))
        self._command_runner = command_runner or _default_command_runner
        self._monotonic = monotonic or time.monotonic
        self._now_provider = now_provider or datetime.now
        self._cached_payload: RoomContextPayload | None = None
        self._cached_at: float | None = None

    def __call__(self) -> RoomContextPayload | None:
        if self._motion_sensor_name is None and self._meter_name is None:
            return None

        now_monotonic = self._monotonic()
        if self._cached_payload is not None and self._cached_at is not None:
            if now_monotonic - self._cached_at < self._ttl_sec:
                return self._cached_payload

        payload = self._collect_payload()
        if payload is not None:
            self._cached_payload = payload
            self._cached_at = now_monotonic
            return payload
        return self._cached_payload

    def _collect_payload(self) -> RoomContextPayload | None:
        payload: RoomContextPayload = {
            "source": "switchbot",
            "observedAt": self._now_provider().isoformat(timespec="seconds"),
        }
        found = False

        motion_status = self._fetch_status(self._motion_sensor_name)
        if motion_status is not None:
            payload["motionSensor"] = motion_status
            found = True

        meter_status = self._fetch_status(self._meter_name)
        if meter_status is not None:
            payload["meter"] = meter_status
            found = True

        return payload if found else None

    def _fetch_status(self, device_name: str | None) -> dict[str, Any] | None:
        if device_name is None:
            return None
        command = ["switchbot", "status", "--name", device_name, "--json"]
        try:
            result = self._command_runner(command)
        except Exception as error:
            logger.warning("SwitchBot room context command failed: %s", error)
            return None

        returncode = getattr(result, "returncode", 1)
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
        if returncode != 0:
            logger.warning(
                "SwitchBot room context command failed: cmd=%s returncode=%s stderr=%s",
                command,
                returncode,
                str(stderr).strip(),
            )
            return None

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            logger.warning("SwitchBot room context JSON decode failed: %s", error)
            return None

        return _normalize_status_payload(payload)


def _normalize_device_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = name.strip()
    return normalized or None


def _normalize_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    normalized: dict[str, Any] = {
        "deviceId": payload.get("device_id"),
        "deviceName": payload.get("device_name"),
        "deviceType": payload.get("device_type"),
    }
    if isinstance(status, dict):
        normalized.update(status)
    return {
        key: value
        for key, value in normalized.items()
        if value is not None
    }
