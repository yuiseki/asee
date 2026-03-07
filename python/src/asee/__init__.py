"""Agentic seeing backend primitives."""

from .biometric_status import BiometricStatusTracker
from .camera_layout import (
    build_camera_csv,
    detect_v4l2_devices,
    extend_with_optional_camera,
    parse_camera_csv,
    parse_v4l2_devices,
)
from .capture_writer import FaceCaptureWriter
from .detection_runtime import YunetDetectionPipeline, to_square
from .dnn_policy import (
    IMPORTANT_OPENCL_WARNING_NOTE,
    emit_opencl_nonfatal_warning_note,
    should_use_opencl_dnn,
)
from .http_app import InMemoryHttpRuntime, OverlayTextState, create_http_app
from .overlay import GodModeOverlay
from .owner_policy import OWNER_COSINE_THRESHOLD, OWNER_TOPK, keep_largest_owner
from .server_runtime import SeeingServerRuntime
from .tracking import FaceBox, FaceTracker
from .video_server import (
    GodModeVideoServer,
    build_arg_parser,
    build_server_from_args,
    encode_frame_to_jpeg,
    main,
    resolve_camera_args,
)
from .web_shell import (
    build_icon_svg,
    build_service_worker_script,
    build_web_manifest,
)

__all__ = [
    "BiometricStatusTracker",
    "FaceBox",
    "FaceCaptureWriter",
    "FaceTracker",
    "IMPORTANT_OPENCL_WARNING_NOTE",
    "GodModeVideoServer",
    "build_arg_parser",
    "build_server_from_args",
    "build_camera_csv",
    "detect_v4l2_devices",
    "keep_largest_owner",
    "main",
    "OWNER_COSINE_THRESHOLD",
    "OWNER_TOPK",
    "create_http_app",
    "build_icon_svg",
    "build_service_worker_script",
    "build_web_manifest",
    "YunetDetectionPipeline",
    "emit_opencl_nonfatal_warning_note",
    "extend_with_optional_camera",
    "GodModeOverlay",
    "InMemoryHttpRuntime",
    "OverlayTextState",
    "parse_camera_csv",
    "parse_v4l2_devices",
    "SeeingServerRuntime",
    "should_use_opencl_dnn",
    "to_square",
    "encode_frame_to_jpeg",
    "resolve_camera_args",
]
