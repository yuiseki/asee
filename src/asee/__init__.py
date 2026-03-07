"""Agentic seeing backend primitives."""

from .biometric_status import BiometricStatusTracker
from .camera_layout import extend_with_optional_camera, parse_v4l2_devices
from .web_shell import (
    build_icon_svg,
    build_service_worker_script,
    build_web_manifest,
)

__all__ = [
    "BiometricStatusTracker",
    "build_icon_svg",
    "build_service_worker_script",
    "build_web_manifest",
    "extend_with_optional_camera",
    "parse_v4l2_devices",
]
