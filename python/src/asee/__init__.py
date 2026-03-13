"""Agentic seeing backend primitives."""

from .biometric_client import (
    RemoteBiometricStatusClient,
    fetch_remote_biometric_status,
    owner_face_absent_for_lock_from_status,
    owner_face_recent_for_unlock_from_status,
    resolve_remote_biometric_status_client,
)
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
from .diagnostics import (
    JsonlDiagnosticsLogger,
    MemoryMonitor,
    NullDiagnosticsLogger,
    ProcessMetrics,
    build_default_diagnostics_log_path,
    read_process_metrics,
)
from .dnn_policy import (
    IMPORTANT_OPENCL_WARNING_NOTE,
    emit_opencl_nonfatal_warning_note,
    should_use_opencl_dnn,
)
from .enroll_owner import (
    DEFAULT_OWNER_EMBED_PATH,
    EnrollmentError,
    fetch_frame_from_server,
    run_enrollment,
)
from .http_app import InMemoryHttpRuntime, OverlayTextState, create_http_app
from .model_assets import (
    candidate_model_asset_paths,
    resolve_model_asset_path,
)
from .overlay import GodModeOverlay
from .owner_policy import OWNER_COSINE_THRESHOLD, OWNER_TOPK, keep_largest_owner
from .server_runtime import SeeingServerRuntime
from .tracking import FaceBox, FaceTracker
from .video_server import (
    CaptureSettings,
    GodModeVideoServer,
    LiveCameraDisabledError,
    build_arg_parser,
    build_server_from_args,
    decode_fourcc_value,
    encode_frame_to_jpeg,
    main,
    resolve_camera_args,
    resolve_capture_settings,
    resolve_opencv_threads,
)
from .web_shell import (
    build_icon_svg,
    build_service_worker_script,
    build_web_manifest,
)

__all__ = [
    "BiometricStatusTracker",
    "RemoteBiometricStatusClient",
    "fetch_remote_biometric_status",
    "owner_face_absent_for_lock_from_status",
    "owner_face_recent_for_unlock_from_status",
    "resolve_remote_biometric_status_client",
    "CaptureSettings",
    "FaceBox",
    "FaceCaptureWriter",
    "FaceTracker",
    "IMPORTANT_OPENCL_WARNING_NOTE",
    "GodModeVideoServer",
    "JsonlDiagnosticsLogger",
    "LiveCameraDisabledError",
    "MemoryMonitor",
    "NullDiagnosticsLogger",
    "ProcessMetrics",
    "build_arg_parser",
    "build_server_from_args",
    "build_camera_csv",
    "build_default_diagnostics_log_path",
    "candidate_model_asset_paths",
    "create_http_app",
    "detect_v4l2_devices",
    "decode_fourcc_value",
    "DEFAULT_OWNER_EMBED_PATH",
    "EnrollmentError",
    "build_icon_svg",
    "build_service_worker_script",
    "build_web_manifest",
    "fetch_frame_from_server",
    "YunetDetectionPipeline",
    "emit_opencl_nonfatal_warning_note",
    "extend_with_optional_camera",
    "GodModeOverlay",
    "InMemoryHttpRuntime",
    "OverlayTextState",
    "keep_largest_owner",
    "main",
    "OWNER_COSINE_THRESHOLD",
    "OWNER_TOPK",
    "parse_camera_csv",
    "parse_v4l2_devices",
    "read_process_metrics",
    "resolve_model_asset_path",
    "SeeingServerRuntime",
    "should_use_opencl_dnn",
    "to_square",
    "encode_frame_to_jpeg",
    "resolve_camera_args",
    "resolve_capture_settings",
    "resolve_opencv_threads",
    "run_enrollment",
]
