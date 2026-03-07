"""Agentic seeing backend primitives."""

from .biometric_status import BiometricStatusTracker
from .camera_layout import extend_with_optional_camera, parse_v4l2_devices
from .dnn_policy import (
    IMPORTANT_OPENCL_WARNING_NOTE,
    emit_opencl_nonfatal_warning_note,
    should_use_opencl_dnn,
)
from .http_app import InMemoryHttpRuntime, OverlayTextState, create_http_app
from .web_shell import (
    build_icon_svg,
    build_service_worker_script,
    build_web_manifest,
)

__all__ = [
    "BiometricStatusTracker",
    "IMPORTANT_OPENCL_WARNING_NOTE",
    "create_http_app",
    "build_icon_svg",
    "build_service_worker_script",
    "build_web_manifest",
    "emit_opencl_nonfatal_warning_note",
    "extend_with_optional_camera",
    "InMemoryHttpRuntime",
    "OverlayTextState",
    "parse_v4l2_devices",
    "should_use_opencl_dnn",
]
