"""Agentic seeing backend primitives."""

from .biometric_status import BiometricStatusTracker
from .camera_layout import extend_with_optional_camera, parse_v4l2_devices
from .http_app import InMemoryHttpRuntime, OverlayTextState, create_http_app
from .web_shell import (
    build_icon_svg,
    build_service_worker_script,
    build_web_manifest,
)

__all__ = [
    "BiometricStatusTracker",
    "create_http_app",
    "build_icon_svg",
    "build_service_worker_script",
    "build_web_manifest",
    "extend_with_optional_camera",
    "InMemoryHttpRuntime",
    "OverlayTextState",
    "parse_v4l2_devices",
]
