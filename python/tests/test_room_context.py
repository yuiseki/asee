from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from asee.room_context import SwitchBotRoomContextProvider


def _completed(payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=json.dumps(payload, ensure_ascii=False), stderr="")


def test_switchbot_room_context_provider_collects_motion_and_meter_status() -> None:
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> SimpleNamespace:
        calls.append(list(cmd))
        if "リビングルームの人感センサー" in cmd:
            return _completed(
                {
                    "device_id": "motion-1",
                    "device_name": "リビングルームの人感センサー",
                    "device_type": "Motion Sensor",
                    "status": {
                        "moveDetected": False,
                        "brightness": "dim",
                        "battery": 100,
                    },
                }
            )
        return _completed(
            {
                "device_id": "meter-1",
                "device_name": "リビング温湿度計",
                "device_type": "Meter",
                "status": {
                    "temperature": 24.7,
                    "humidity": 25,
                    "battery": 100,
                },
            }
        )

    provider = SwitchBotRoomContextProvider(
        motion_sensor_name="リビングルームの人感センサー",
        meter_name="リビング温湿度計",
        ttl_sec=5.0,
        command_runner=runner,
        monotonic=lambda: 100.0,
        now_provider=lambda: datetime(2026, 3, 17, 6, 0, 0),
    )

    payload = provider()

    assert payload == {
        "source": "switchbot",
        "observedAt": "2026-03-17T06:00:00",
        "motionSensor": {
            "deviceId": "motion-1",
            "deviceName": "リビングルームの人感センサー",
            "deviceType": "Motion Sensor",
            "moveDetected": False,
            "brightness": "dim",
            "battery": 100,
        },
        "meter": {
            "deviceId": "meter-1",
            "deviceName": "リビング温湿度計",
            "deviceType": "Meter",
            "temperature": 24.7,
            "humidity": 25,
            "battery": 100,
        },
    }
    assert len(calls) == 2


def test_switchbot_room_context_provider_uses_cache_within_ttl() -> None:
    calls: list[list[str]] = []
    current_time = {"value": 10.0}

    def runner(cmd: list[str]) -> SimpleNamespace:
        calls.append(list(cmd))
        return _completed(
            {
                "device_id": "motion-1",
                "device_name": "リビングルームの人感センサー",
                "device_type": "Motion Sensor",
                "status": {"moveDetected": True, "brightness": "bright"},
            }
        )

    provider = SwitchBotRoomContextProvider(
        motion_sensor_name="リビングルームの人感センサー",
        meter_name=None,
        ttl_sec=5.0,
        command_runner=runner,
        monotonic=lambda: current_time["value"],
        now_provider=lambda: datetime(2026, 3, 17, 6, 1, 0),
    )

    first = provider()
    current_time["value"] = 12.0
    second = provider()

    assert first == second
    assert len(calls) == 1
