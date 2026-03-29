"""GOD MODE-compatible video server rebuilt on top of extracted asee modules."""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import ipaddress
import logging
import os
import platform
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

import cv2
import numpy as np
import numpy.typing as npt
import werkzeug.serving

try:
    from turbojpeg import TurboJPEG as _TurboJPEG  # type: ignore[import-untyped]

    _TURBOJPEG_AVAILABLE = True
except ImportError:
    _TurboJPEG = None
    _TURBOJPEG_AVAILABLE = False

from .detection_runtime import to_square
from .diagnostics import (
    DiagnosticsLogger,
    JsonlDiagnosticsLogger,
    MemoryMonitor,
    NullDiagnosticsLogger,
    build_default_diagnostics_log_path,
)
from .full_frame_capture import FullFrameCaptureWriter
from .http_app import create_http_app
from .model_assets import resolve_model_asset_path
from .overlay import GodModeOverlay
from .overlay_broadcaster import OverlayBroadcaster
from .room_context import SwitchBotRoomContextProvider
from .server_runtime import SeeingServerRuntime
from .tracking import FaceBox, FaceTracker
from .webrtc_signaling import create_webrtc_app

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
OWNER_EMBED_PATH = resolve_model_asset_path("owner_embedding.npy")
OWNER_EMBED_PATHS = {
    "facenet-pytorch": resolve_model_asset_path("owner_embedding_facenet_pytorch.npy"),
    "opencv-sface": resolve_model_asset_path("owner_embedding_opencv_sface.npy"),
}

type FrameArray = npt.NDArray[np.uint8]
type CameraCaptureSource = int | str


class LiveCameraDisabledError(RuntimeError):
    """Raised when the caller attempts live capture without an explicit opt-in."""


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    """Resolved capture parameters used for camera-open attempts."""

    width: int
    height: int
    fps: float
    fourcc: str


@dataclass(slots=True)
class CameraRuntimeStats:
    """Capture/detection counters recorded in diagnostic summaries."""

    frame_count: int = 0
    read_failures: int = 0
    consecutive_read_failures: int = 0
    detection_iterations: int = 0
    last_frame_at: float | None = None
    last_failure_at: float | None = None
    last_detection_at: float | None = None
    last_capture_log_at: float = 0.0
    last_detection_log_at: float = 0.0


@dataclass(slots=True)
class MjpegChunkCacheEntry:
    """Cached MJPEG chunk for a specific stream revision."""

    revision: int = -1
    chunk: bytes | None = None


jpeg_encoder = _TurboJPEG() if _TURBOJPEG_AVAILABLE else None


def encode_frame_to_jpeg(frame: FrameArray, quality: int = 70) -> bytes:
    """Encode a frame into a JPEG payload using TurboJPEG with OpenCV fallback."""
    if jpeg_encoder is not None:
        try:
            return cast(bytes, jpeg_encoder.encode(frame, quality=quality))
        except Exception as e:
            logger.error("TurboJPEG encode failed: %s, falling back to OpenCV", e)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buffer = cv2.imencode(".jpg", frame, encode_param)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buffer.tobytes()


def decode_fourcc_value(value: float) -> str:
    """Decode an OpenCV numeric FOURCC value into a 4-character code."""
    int_value = int(value)
    chars = [chr((int_value >> (8 * shift)) & 0xFF) for shift in range(4)]
    return "".join(chars)


def resolve_capture_settings(
    *,
    camera_ids: Sequence[int],
    capture_profile: str = "auto",
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    fourcc: str | None = None,
) -> CaptureSettings:
    """Resolve safe capture settings based on camera count and explicit overrides."""
    normalized_profile = capture_profile.strip().lower()
    if normalized_profile not in {"auto", "720p"}:
        raise ValueError("Capture profile must be one of: auto, 720p")

    if normalized_profile == "720p":
        base = CaptureSettings(
            width=1280,
            height=720,
            fps=30.0,
            fourcc="MJPG",
        )
    else:
        base = (
            CaptureSettings(width=1280, height=720, fps=30.0, fourcc="MJPG")
            if len(camera_ids) > 1
            else CaptureSettings(width=1280, height=720, fps=30.0, fourcc="MJPG")
        )
    resolved = CaptureSettings(
        width=base.width if width is None else int(width),
        height=base.height if height is None else int(height),
        fps=base.fps if fps is None else float(fps),
        fourcc=base.fourcc if fourcc is None else str(fourcc).upper(),
    )
    if resolved.width <= 0 or resolved.height <= 0:
        raise ValueError("Capture width and height must be positive integers")
    if resolved.fps <= 0:
        raise ValueError("Capture fps must be positive")
    if len(resolved.fourcc) != 4:
        raise ValueError("Capture FOURCC must be a 4-character code")
    return resolved


def resolve_opencv_threads(
    *,
    camera_ids: Sequence[int],
    opencv_threads: int | None = None,
) -> int | None:
    """Resolve a safer OpenCV thread limit for the current camera topology."""
    if opencv_threads is not None:
        if int(opencv_threads) <= 0:
            raise ValueError("OpenCV thread limit must be positive")
        return int(opencv_threads)
    if len(camera_ids) > 1:
        return 1
    return None


def resolve_camera_args(*, device: int, cameras_csv: str) -> tuple[int | None, list[int] | None]:
    """Resolve the CLI camera arguments into server constructor values."""
    if cameras_csv.strip():
        camera_list = [int(chunk.strip()) for chunk in cameras_csv.split(",") if chunk.strip()]
        return camera_list[0], camera_list
    if device < 0:
        return None, None
    return device, None


def resolve_camera_source_args(
    *,
    camera_sources_csv: str,
) -> tuple[int | None, list[int] | None, dict[int, CameraCaptureSource] | None]:
    """Resolve logical camera ids mapped to explicit capture sources.

    Format:
      0@0,2@2,4@rtsp://atomcam-hoge.local:8554/video0_unicast
    """
    if not camera_sources_csv.strip():
        return None, None, None
    camera_source_map: dict[int, CameraCaptureSource] = {}
    for raw_entry in camera_sources_csv.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "@" not in entry:
            raise ValueError(
                "camera source entries must use '<camera_id>@<source>' format"
            )
        camera_id_text, source_text = entry.split("@", 1)
        camera_id = int(camera_id_text.strip())
        source_value = source_text.strip()
        if not source_value:
            raise ValueError("camera source must not be empty")
        source: CameraCaptureSource
        if source_value.lstrip("-").isdigit():
            source = int(source_value)
        else:
            source = source_value
        camera_source_map[camera_id] = source
    if not camera_source_map:
        return None, None, None
    camera_list = list(camera_source_map.keys())
    return camera_list[0], camera_list, camera_source_map


def is_network_capture_source(source: str) -> bool:
    """Return True when the source string is a network media URL."""
    scheme = urlsplit(source).scheme.lower()
    return scheme in {"rtsp", "rtsps", "http", "https", "tcp", "udp", "rtmp"}


def resolve_hostname_ipv4(hostname: str, *, timeout_sec: float = 1.0) -> str | None:
    """Resolve a hostname to IPv4 via getent without relying on Python mDNS support."""
    try:
        completed = subprocess.run(
            ["getent", "hosts", hostname],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        candidate = parts[0].strip()
        try:
            parsed = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if parsed.version == 4:
            return candidate
    return None


def _build_url_with_host(parts: SplitResult, host: str) -> str:
    """Replace the hostname in a parsed URL while preserving auth and port."""
    username = parts.username or ""
    password = parts.password or ""
    auth = ""
    if username:
        auth = username
        if password:
            auth += f":{password}"
        auth += "@"
    port = f":{parts.port}" if parts.port is not None else ""
    netloc = f"{auth}{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def normalize_network_capture_source(source: str) -> str:
    """Resolve network capture hostnames to IPv4 when possible."""
    if not is_network_capture_source(source):
        return source
    parts = urlsplit(source)
    hostname = parts.hostname
    if hostname is None:
        return source
    try:
        ipaddress.ip_address(hostname)
        return source
    except ValueError:
        pass
    resolved = resolve_hostname_ipv4(hostname)
    if resolved is None:
        return source
    return _build_url_with_host(parts, resolved)


def discover_stable_camera_source(device_index: int) -> str | None:
    """Return a stable V4L symlink for the current numeric device, if available."""
    device_path = Path(f"/dev/video{device_index}")
    if not device_path.exists():
        return None

    candidates = [
        Path("/dev/v4l/by-id"),
        Path("/dev/v4l/by-path"),
    ]
    for base in candidates:
        if not base.exists():
            continue
        try:
            entries = sorted(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.resolve() != device_path.resolve():
                    continue
            except OSError:
                continue
            if entry.name.endswith("video-index0"):
                return str(entry)
        for entry in entries:
            try:
                if entry.resolve() == device_path.resolve():
                    return str(entry)
            except OSError:
                continue
    return None


def discover_available_stable_camera_sources() -> list[str]:
    """List unique stable V4L symlinks, preferring by-id over by-path."""
    discovered: list[str] = []
    seen_targets: set[Path] = set()
    for base in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
        if not base.exists():
            continue
        try:
            entries = sorted(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.name.endswith("video-index0"):
                continue
            try:
                target = entry.resolve()
            except OSError:
                continue
            if target in seen_targets:
                continue
            seen_targets.add(target)
            discovered.append(str(entry))
    return discovered


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the extracted video server."""
    parser = argparse.ArgumentParser(description="GOD MODE Video Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="単一カメラのデバイス番号。既定値 -1 は no-camera 安全モード",
    )
    parser.add_argument(
        "--cameras",
        default="",
        help="カンマ区切りのデバイス番号リスト (例: 0,2). 指定すると複数カメラを同時配信",
    )
    parser.add_argument(
        "--camera-sources",
        default="",
        help=(
            "論理 camera_id と capture source の対応。"
            "例: 0@0,2@2,4@rtsp://cam-a:8554/video0_unicast,"
            "6@rtsp://cam-b:8554/video0_unicast"
        ),
    )
    parser.add_argument("--cam-interval", type=int, default=60)
    parser.add_argument("--title", default="GOD MODE", help="ウィンドウタイトル")
    parser.add_argument(
        "--face-capture-dir",
        default="/tmp/god-mode-face-segments",
        help="OWNER 顔クロップの保存先ディレクトリ（空文字で無効化）",
    )
    parser.add_argument(
        "--face-capture-min-interval",
        type=float,
        default=1.0,
        help="顔クロップ保存の最小間隔（秒）。デフォルト: 1.0",
    )
    parser.add_argument(
        "--subject-capture-dir",
        default="/tmp/god-mode-subject-segments",
        help="SUBJECT 顔クロップの保存先ディレクトリ（閾値チューニング用 true negative データ）",
    )
    parser.add_argument(
        "--full-frame-capture-dir",
        default="/home/yuiseki/Workspaces/private/datasets/webcams",
        help="定期 full-frame サンプリングの保存先ディレクトリ（空文字で無効化）",
    )
    parser.add_argument(
        "--full-frame-morning-interval-sec",
        type=float,
        default=300.0,
        help="朝帯 (05:00-09:59) の full-frame 保存間隔秒数。既定 300 秒",
    )
    parser.add_argument(
        "--full-frame-day-interval-sec",
        type=float,
        default=900.0,
        help="昼帯 (10:00-16:59) の full-frame 保存間隔秒数。既定 900 秒",
    )
    parser.add_argument(
        "--full-frame-evening-interval-sec",
        type=float,
        default=600.0,
        help="晩帯 (17:00-23:59) の full-frame 保存間隔秒数。既定 600 秒",
    )
    parser.add_argument(
        "--full-frame-overnight-interval-sec",
        type=float,
        default=0.0,
        help="深夜帯 (00:00-04:59) の full-frame 保存間隔秒数。既定 0 で無効",
    )
    parser.add_argument(
        "--motion-sensor-name",
        default="リビングルームの人感センサー",
        help="face crop sidecar に記録する SwitchBot Motion Sensor の名前。空文字で無効化",
    )
    parser.add_argument(
        "--meter-name",
        default="リビング温湿度計",
        help="face crop sidecar に記録する SwitchBot Meter の名前。空文字で無効化",
    )
    parser.add_argument(
        "--room-context-ttl-sec",
        type=float,
        default=60.0,
        help="SwitchBot room context の成功時キャッシュ TTL（秒）。失敗時は追加 backoff が入る",
    )
    parser.add_argument(
        "--allow-live-camera",
        action="store_true",
        help="危険な実機 Web カメラ入力を明示的に許可する",
    )
    parser.add_argument(
        "--diagnostic-log-path",
        default=None,
        help="再発解析用 JSONL 診断ログの保存先。未指定時は persistent path を自動採番",
    )
    parser.add_argument(
        "--memory-log-interval-sec",
        type=float,
        default=15.0,
        help="プロセスメモリ統計の採取間隔（秒）。0 以下で無効化",
    )
    parser.add_argument(
        "--auto-shutdown-sec",
        type=float,
        default=0.0,
        help="安全のため指定秒数で自動停止する。0 以下で無効化",
    )
    parser.add_argument(
        "--capture-profile",
        choices=("auto", "720p"),
        default="auto",
        help=(
            "既定キャプチャ profile。"
            "auto は現在の camera topology 既定、720p は 1280x720 を要求する"
        ),
    )
    parser.add_argument("--width", type=int, default=None, help="キャプチャ要求幅")
    parser.add_argument("--height", type=int, default=None, help="キャプチャ要求高さ")
    parser.add_argument("--fps", type=float, default=None, help="キャプチャ要求 FPS")
    parser.add_argument(
        "--fourcc",
        default=None,
        help="キャプチャ要求 FOURCC (例: MJPG, YUYV, H264)",
    )
    parser.add_argument(
        "--disable-face-detect",
        action="store_true",
        help="安全な切り分けのため顔検出 worker を起動しない",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=None,
        help="OpenCV の内部 thread 数。複数カメラ時の既定は 1",
    )
    parser.add_argument(
        "--detection-backend",
        choices=["opencv", "onnxruntime", "insightface"],
        default="insightface",
        help=(
            "顔検出バックエンド: insightface (既定, CUDA), "
            "onnxruntime (YuNet CUDA), または opencv (CPU)"
        ),
    )
    parser.add_argument(
        "--insightface-det-size",
        type=int,
        default=320,
        help="InsightFace detector の det_size。既定は 320",
    )
    parser.add_argument(
        "--recognition-backend",
        choices=["facenet-pytorch", "opencv-sface"],
        default="facenet-pytorch",
        help="顔識別バックエンド: facenet-pytorch (既定, CUDA) または opencv-sface",
    )
    parser.add_argument(
        "--transport",
        choices=("mjpeg", "webrtc"),
        default="webrtc",
        help="配信 transport。既定は WebRTC、mjpeg は compatibility fallback",
    )
    return parser


def build_server_from_args(args: argparse.Namespace) -> GodModeVideoServer:
    """Build a server instance from parsed CLI arguments."""
    device_index, camera_list, camera_source_map = resolve_camera_source_args(
        camera_sources_csv=str(getattr(args, "camera_sources", "")),
    )
    if camera_source_map is None:
        device_index, camera_list = resolve_camera_args(
            device=int(args.device),
            cameras_csv=str(args.cameras),
        )
    allow_live_camera = bool(args.allow_live_camera)
    if (camera_list or device_index is not None) and not allow_live_camera:
        raise LiveCameraDisabledError(
            "Live camera access is disabled by default. Pass --allow-live-camera to opt in."
        )
    diagnostic_log_path = (
        Path(str(args.diagnostic_log_path))
        if args.diagnostic_log_path is not None
        else build_default_diagnostics_log_path()
    )
    diagnostics_logger = JsonlDiagnosticsLogger(diagnostic_log_path)
    motion_sensor_name = str(args.motion_sensor_name).strip() or None
    meter_name = str(args.meter_name).strip() or None
    room_context_provider = None
    if motion_sensor_name is not None or meter_name is not None:
        room_context_provider = SwitchBotRoomContextProvider(
            motion_sensor_name=motion_sensor_name,
            meter_name=meter_name,
            ttl_sec=float(args.room_context_ttl_sec),
        )
    return GodModeVideoServer(
        port=int(args.port),
        device_index=device_index,
        camera_list=camera_list,
        camera_source_map=camera_source_map,
        cam_interval=int(args.cam_interval),
        title=str(args.title),
        face_capture_dir=str(args.face_capture_dir) or None,
        face_capture_min_interval=float(args.face_capture_min_interval),
        subject_capture_dir=str(args.subject_capture_dir) or None,
        full_frame_capture_dir=str(args.full_frame_capture_dir) or None,
        full_frame_morning_interval_sec=float(args.full_frame_morning_interval_sec),
        full_frame_day_interval_sec=float(args.full_frame_day_interval_sec),
        full_frame_evening_interval_sec=float(args.full_frame_evening_interval_sec),
        full_frame_overnight_interval_sec=float(args.full_frame_overnight_interval_sec),
        allow_live_camera=allow_live_camera,
        diagnostics_logger=diagnostics_logger,
        memory_log_interval_sec=float(args.memory_log_interval_sec),
        auto_shutdown_sec=float(args.auto_shutdown_sec),
        capture_profile=str(args.capture_profile),
        room_context_provider=room_context_provider,
        width=None if args.width is None else int(args.width),
        height=None if args.height is None else int(args.height),
        fps=None if args.fps is None else float(args.fps),
        fourcc=None if args.fourcc is None else str(args.fourcc),
        opencv_threads=None if args.opencv_threads is None else int(args.opencv_threads),
        enable_face_detection=not bool(args.disable_face_detect),
        detection_backend=str(args.detection_backend),
        insightface_det_size=int(getattr(args, "insightface_det_size", 320)),
        recognition_backend=str(args.recognition_backend),
        owner_embedding_path=resolve_default_owner_embedding_path(str(args.recognition_backend)),
        transport=str(getattr(args, "transport", "webrtc")),
    )


def resolve_default_owner_embedding_path(recognition_backend: str) -> Path:
    """Resolve the default owner embedding asset for the selected recognition backend."""
    try:
        return OWNER_EMBED_PATHS[recognition_backend]
    except KeyError as exc:
        raise ValueError(f"unsupported recognition backend: {recognition_backend}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the extracted video server."""
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    server = build_server_from_args(args)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
    return 0


class GodModeVideoServer:
    """Webcam capture, overlay drawing, and MJPEG serving."""

    def __init__(
        self,
        *,
        port: int = 8765,
        device_index: int | None = 0,
        camera_list: list[int] | None = None,
        camera_source_map: dict[int, CameraCaptureSource] | None = None,
        cam_interval: int = 60,
        title: str = "GOD MODE",
        face_capture_dir: str | None = None,
        face_capture_min_interval: float = 1.0,
        subject_capture_dir: str | None = None,
        full_frame_capture_dir: str | None = None,
        full_frame_morning_interval_sec: float = 300.0,
        full_frame_day_interval_sec: float = 900.0,
        full_frame_evening_interval_sec: float = 600.0,
        full_frame_overnight_interval_sec: float = 0.0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        fourcc: str | None = None,
        opencv_threads: int | None = None,
        owner_embedding_path: str | Path | None = None,
        allow_live_camera: bool = False,
        diagnostics_logger: DiagnosticsLogger | None = None,
        memory_log_interval_sec: float = 30.0,
        auto_shutdown_sec: float = 0.0,
        enable_face_detection: bool = True,
        capture_profile: str = "auto",
        detection_backend: str = "insightface",
        insightface_det_size: int = 320,
        recognition_backend: str = "facenet-pytorch",
        transport: str = "webrtc",
        room_context_provider: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        requested_camera_list = (
            list(camera_source_map.keys())
            if camera_source_map
            else (camera_list or ([device_index] if device_index is not None else []))
        )
        if requested_camera_list and not allow_live_camera:
            raise LiveCameraDisabledError(
                "Live camera access is disabled by default. Pass --allow-live-camera to opt in."
            )
        self.capture_settings = resolve_capture_settings(
            camera_ids=requested_camera_list,
            capture_profile=capture_profile,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
        )
        self._opencv_threads = resolve_opencv_threads(
            camera_ids=requested_camera_list,
            opencv_threads=opencv_threads,
        )
        self.port = port
        self.device_index = device_index
        self.title = title
        self.width = self.capture_settings.width
        self.height = self.capture_settings.height
        self._camera_list = requested_camera_list
        self._cam_interval = cam_interval
        self._cam_index = 0
        self._allow_live_camera = allow_live_camera
        self._auto_shutdown_sec = auto_shutdown_sec
        self._capture_profile = capture_profile
        self._capture_fps = self.capture_settings.fps
        self._capture_period = 1.0 / self.capture_settings.fps
        self._capture_fourcc = self.capture_settings.fourcc
        self._enable_face_detection = enable_face_detection
        self._transport = transport
        self._diagnostics = diagnostics_logger or NullDiagnosticsLogger()
        self._memory_monitor = MemoryMonitor(
            self._diagnostics,
            interval_sec=memory_log_interval_sec,
        )
        self.overlay = GodModeOverlay(
            width=self.width,
            height=self.height,
            face_capture_dir=face_capture_dir,
            face_capture_min_interval=face_capture_min_interval,
            subject_capture_dir=subject_capture_dir,
            detection_backend=detection_backend,
            insightface_det_size=insightface_det_size,
            recognition_backend=recognition_backend,
            room_context_provider=room_context_provider,
        )
        self._room_context_provider = room_context_provider
        self._full_frame_capture_writer = (
            FullFrameCaptureWriter(
                full_frame_capture_dir,
                morning_interval_sec=full_frame_morning_interval_sec,
                day_interval_sec=full_frame_day_interval_sec,
                evening_interval_sec=full_frame_evening_interval_sec,
                overnight_interval_sec=full_frame_overnight_interval_sec,
            )
            if full_frame_capture_dir is not None
            else None
        )
        self.runtime = SeeingServerRuntime(
            title=title,
            overlay=self.overlay,
            camera_ids=tuple(self._camera_list),
            jpeg_encoder=encode_frame_to_jpeg,
            jpeg_quality=80,
            stream_factory=self.iter_mjpeg,
        )
        self.runtime.transport = transport
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.is_running = False
        self._frame_count = 0
        self._face_boxes: list[FaceBox] = []
        self._face_lock = threading.Lock()
        self._switch_event = threading.Event()
        self._next_device: int | None = None
        self._multi_frames: dict[int, FrameArray | None] = {
            device: None for device in self._camera_list
        }
        self._multi_locks: dict[int, threading.Lock] = {
            device: threading.Lock() for device in self._camera_list
        }
        self._face_boxes_per_cam: dict[int, list[FaceBox]] = {
            device: [] for device in self._camera_list
        }
        self._face_locks_per_cam: dict[int, threading.Lock] = {
            device: threading.Lock() for device in self._camera_list
        }
        self._detect_lock = threading.Lock()
        self._trackers_per_cam: dict[int, FaceTracker] = {
            device: FaceTracker(alpha=0.4, max_lost_frames=2, min_hits=3)
            for device in self._camera_list
        }
        self._worker_threads: list[threading.Thread] = []
        self._camera_stats: dict[int, CameraRuntimeStats] = {
            device: CameraRuntimeStats() for device in self._camera_list
        }
        self._single_camera_stats = CameraRuntimeStats()
        self._explicit_camera_sources = dict(camera_source_map or {})
        self._camera_sources = self._bootstrap_camera_sources(
            self._camera_list,
            explicit_sources=self._explicit_camera_sources,
        )
        if self.device_index is not None:
            self._single_camera_source: CameraCaptureSource = (
                discover_stable_camera_source(self.device_index) or self.device_index
            )
        else:
            self._single_camera_source = -1
        self._capture_reopen_failure_threshold = 10
        self._capture_stale_frame_threshold = 45
        self._stream_state_lock = threading.Lock()
        self._stream_state_changed = threading.Condition(self._stream_state_lock)
        self._stream_revisions: dict[int | None, int] = {None: 0}
        self._mjpeg_chunk_cache: dict[int | None, MjpegChunkCacheEntry] = {
            None: MjpegChunkCacheEntry()
        }
        for device in self._camera_list:
            self._stream_revisions[device] = 0
            self._mjpeg_chunk_cache[device] = MjpegChunkCacheEntry()
        self._app = create_http_app(self.runtime, request_logger=self._record_http_request)
        self._webrtc_broadcaster = OverlayBroadcaster()
        self._http_server: werkzeug.serving.BaseWSGIServer | None = None
        resolved_owner_embedding_path = (
            Path(owner_embedding_path)
            if owner_embedding_path is not None
            else resolve_default_owner_embedding_path(recognition_backend)
        )
        self._load_owner_embedding(
            resolved_owner_embedding_path,
            recognition_backend=recognition_backend,
        )

    def _bootstrap_camera_sources(
        self,
        camera_ids: Sequence[int],
        *,
        explicit_sources: dict[int, CameraCaptureSource] | None = None,
    ) -> dict[int, CameraCaptureSource]:
        assigned: dict[int, CameraCaptureSource] = {}
        used_sources: set[str] = set()
        for device in camera_ids:
            explicit = None if explicit_sources is None else explicit_sources.get(device)
            if explicit is not None:
                assigned[device] = explicit
                if isinstance(explicit, str) and Path(explicit).exists():
                    used_sources.add(explicit)
                continue
            discovered = discover_stable_camera_source(device)
            if discovered is not None:
                assigned[device] = discovered
                used_sources.add(discovered)
            else:
                assigned[device] = device

        leftovers = [
            source
            for source in discover_available_stable_camera_sources()
            if source not in used_sources
        ]
        unresolved = [
            device
            for device, source in assigned.items()
            if isinstance(source, int) and not Path(f"/dev/video{source}").exists()
        ]
        for device, source in zip(unresolved, leftovers, strict=False):
            assigned[device] = source
            logger.info(
                "Camera %s remapped to stable source %s after device re-enumeration",
                device,
                source,
            )
        return assigned

    @property
    def current_frame(self) -> FrameArray | None:
        return self.runtime.current_frame

    def update_frame(self, frame: FrameArray, *, camera_id: int | None = None) -> None:
        with self._frame_lock:
            resolved_camera_id = camera_id
            if resolved_camera_id is None and len(self._camera_list) == 1:
                resolved_camera_id = self._camera_list[0]
            self.runtime.update_frame(frame, camera_id=resolved_camera_id)
            if resolved_camera_id is not None:
                self._multi_frames[resolved_camera_id] = frame
        self._invalidate_mjpeg_stream(resolved_camera_id)

    def update_overlay_text(self, *, caption: str = "", prediction: str = "") -> None:
        self.runtime.update_overlay_text(caption=caption, prediction=prediction)
        self._invalidate_all_mjpeg_streams()

    def get_biometric_status(self) -> dict[str, bool | int | float | None]:
        self.runtime.set_running(self.is_running)
        return self.runtime.get_biometric_status()

    def _load_owner_embedding(
        self,
        path: str | Path,
        *,
        recognition_backend: str,
    ) -> None:
        resolved = Path(path)
        candidates = [resolved]
        if recognition_backend == "opencv-sface" and resolved != OWNER_EMBED_PATH:
            candidates.append(OWNER_EMBED_PATH)

        for candidate in candidates:
            if candidate.exists():
                self.runtime.load_owner_embedding(candidate)
                logger.info("Owner embedding loaded: %s", candidate)
                self._diagnostics.log_event(
                    "owner_embedding_loaded",
                    path=str(candidate),
                    recognition_backend=recognition_backend,
                )
                return

        logger.info("No owner embedding found. All faces labeled SUBJECT.")
        self._diagnostics.log_event(
            "owner_embedding_missing",
            path=str(resolved),
            recognition_backend=recognition_backend,
        )

    def _record_owner_presence(
        self,
        faces: list[FaceBox],
        *,
        camera_id: int | None = None,
    ) -> None:
        resolved_camera_id = camera_id
        if resolved_camera_id is None and len(self._camera_list) == 1:
            resolved_camera_id = self._camera_list[0]

        if resolved_camera_id is None:
            with self._face_lock:
                self._face_boxes = list(faces)
        else:
            face_lock = self._face_locks_per_cam.get(resolved_camera_id)
            if face_lock is not None:
                with face_lock:
                    self._face_boxes_per_cam[resolved_camera_id] = list(faces)

        self.runtime.record_faces(faces, camera_id=resolved_camera_id)
        self._invalidate_mjpeg_stream(resolved_camera_id)

    def start(self) -> None:
        self._stop_event.clear()
        self._worker_threads = []
        self._configure_fault_handler()
        self._diagnostics.log_event(
            "server_starting",
            pid=os.getpid(),
            port=self.port,
            title=self.title,
            device_index=self.device_index,
            camera_list=self._camera_list,
            width=self.width,
            height=self.height,
            allow_live_camera=self._allow_live_camera,
            capture_profile=self._capture_profile,
            requested_fps=self._capture_fps,
            requested_fourcc=self._capture_fourcc,
            opencv_threads=self._opencv_threads,
            face_detection_enabled=self._enable_face_detection,
            transport=self._transport,
            opencv_version=cv2.__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
        )
        self._apply_opencv_thread_limit()
        self._memory_monitor.start()
        if self._auto_shutdown_sec > 0:
            self._start_worker(
                name="auto_shutdown",
                target=self._auto_shutdown_loop,
            )

        if self._camera_list:
            for device in self._camera_list:
                self._start_worker(
                    name=f"capture_{device}",
                    target=self._capture_loop_device,
                    args=(device,),
                )
            self._start_face_detection_workers()
        elif self.device_index is not None:
            self._start_worker(
                name="capture",
                target=self._capture_loop,
            )
            self._start_face_detection_workers()
        else:
            self.update_frame(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            self._diagnostics.log_event("camera_capture_disabled")

        if self._transport == "mjpeg":
            self._http_server = werkzeug.serving.make_server(
                "0.0.0.0",
                self.port,
                self._app,
                threaded=True,
            )
            self._http_server.socket.settimeout(1.0)

        self.is_running = True
        self.runtime.set_running(True)
        self._diagnostics.log_event("server_started", port=self.port, transport=self._transport)

        try:
            if self._transport == "webrtc":
                self._serve_webrtc()
            else:
                self._serve_http()
        finally:
            self._stop_event.set()
            for thread in self._worker_threads:
                thread.join(timeout=1.0)
                self._diagnostics.log_event(
                    "worker_joined",
                    worker=thread.name,
                    alive=thread.is_alive(),
                )
            self._memory_monitor.stop()
            self.is_running = False
            self.runtime.set_running(False)
            self._diagnostics.log_event(
                "server_stopped",
                frame_count=self._frame_count,
                camera_stats=self._camera_stats_payload(),
            )
            self._diagnostics.close()

    def stop(self) -> None:
        self._diagnostics.log_event("server_stop_requested")
        self._stop_event.set()
        with self._stream_state_changed:
            self._stream_state_changed.notify_all()

    def _serve_http(self) -> None:
        server = self._http_server
        if server is None:
            raise RuntimeError("HTTP server is not initialized")
        try:
            while not self._stop_event.is_set():
                try:
                    server.handle_request()
                except OSError as error:
                    logger.warning("HTTP server request loop error: %s", error)
                    self._diagnostics.log_event("http_server_oserror", error=repr(error))
        finally:
            server.server_close()
            self._http_server = None

    def _create_webrtc_app(self) -> Any:
        return create_webrtc_app(
            runtime=self.runtime,
            broadcaster=self._webrtc_broadcaster,
            fps=max(1, int(round(self._capture_fps))),
        )

    def _serve_webrtc(self) -> None:
        asyncio.run(self._serve_webrtc_async())

    async def _serve_webrtc_async(self) -> None:
        from aiohttp import web

        app = self._create_webrtc_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)
        finally:
            await runner.cleanup()

    def switch_camera(self, device_index: int) -> None:
        """Request a camera switch for the single-camera capture loop."""
        self._next_device = device_index
        self._switch_event.set()
        logger.info("Camera switch requested -> device %s", device_index)

    def _normalize_frame(self, frame: FrameArray) -> FrameArray:
        height, width = frame.shape[:2]
        if width == self.width and height == self.height:
            return frame
        return cast(
            FrameArray,
            cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR),
        )

    def _resolve_camera_source(self, device_index: int) -> CameraCaptureSource:
        stored = self._camera_sources.get(device_index)
        if stored is not None:
            if isinstance(stored, str):
                if is_network_capture_source(stored):
                    resolved = normalize_network_capture_source(stored)
                    self._camera_sources[device_index] = resolved
                    return resolved
                if Path(stored).exists():
                    return stored
            if isinstance(stored, int):
                discovered = discover_stable_camera_source(stored)
                if discovered is not None:
                    self._camera_sources[device_index] = discovered
                    return discovered
                return stored
        discovered = discover_stable_camera_source(device_index)
        if discovered is not None:
            self._camera_sources[device_index] = discovered
            return discovered
        self._camera_sources[device_index] = device_index
        return device_index

    def _camera_frame_signature(self, frame: FrameArray) -> bytes:
        sample = frame[::64, ::64]
        return sample.tobytes()

    def _open_camera(self, device_index: int) -> Any:
        source = self._resolve_camera_source(device_index)
        self._diagnostics.log_event(
            "camera_open_attempt",
            camera_id=device_index,
            camera_source=source,
            requested_width=self.width,
            requested_height=self.height,
            requested_fps=self._capture_fps,
            requested_fourcc=self._capture_fourcc,
        )
        if isinstance(source, str) and is_network_capture_source(source):
            cap = cv2.VideoCapture(source, getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY))
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source)
            fourcc = cv2.VideoWriter_fourcc(*self._capture_fourcc)  # type: ignore[attr-defined]
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self._capture_fps)
        if not cap.isOpened():
            logger.error("Cannot open camera device %s (source=%s)", device_index, source)
            self._diagnostics.log_event(
                "camera_open_failed",
                camera_id=device_index,
                camera_source=source,
            )
            cap.release()
            return None
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        actual_fourcc = decode_fourcc_value(float(cap.get(cv2.CAP_PROP_FOURCC)))
        logger.info(
            "Camera %s opened: requested=%sx%s@%sfps/%s actual=%sx%s@%sfps/%s",
            device_index,
            self.width,
            self.height,
            self._capture_fps,
            self._capture_fourcc,
            actual_width,
            actual_height,
            actual_fps,
            actual_fourcc,
        )
        self._diagnostics.log_event(
            "camera_opened",
            camera_id=device_index,
            camera_source=source,
            actual_width=actual_width,
            actual_height=actual_height,
            actual_fps=actual_fps,
            actual_fourcc=actual_fourcc,
        )
        return cap

    def _reopen_camera(self, device_index: int, current_cap: Any, *, reason: str) -> Any:
        self._diagnostics.log_event(
            "camera_reopen_requested",
            camera_id=device_index,
            reason=reason,
        )
        logger.warning("Reopening camera %s after %s", device_index, reason)
        reopened = self._open_camera(device_index)
        if reopened is None:
            return current_cap
        try:
            current_cap.release()
        except Exception:
            logger.warning("Failed to release camera %s during reopen", device_index)
        return reopened

    def _apply_opencv_thread_limit(self) -> None:
        if self._opencv_threads is None:
            self._diagnostics.log_event(
                "opencv_thread_limit_skipped",
                current_threads=cv2.getNumThreads(),
            )
            return
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(self._opencv_threads)
        self._diagnostics.log_event(
            "opencv_thread_limit_applied",
            requested_threads=self._opencv_threads,
            previous_threads=previous_threads,
            current_threads=cv2.getNumThreads(),
        )

    def _capture_loop(self) -> None:
        current_device = self.device_index
        if current_device is None:
            return
        cap = self._open_camera(current_device)
        if cap is None:
            self.update_frame(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            return
        last_signature: bytes | None = None
        consecutive_stale_frames = 0

        while not self._stop_event.is_set():
            if self._switch_event.is_set():
                self._switch_event.clear()
                new_device = self._next_device
                if new_device is not None and new_device != current_device:
                    cap.release()
                    logger.info("Camera %s released", current_device)
                    new_cap = self._open_camera(new_device)
                    if new_cap is not None:
                        cap = new_cap
                        current_device = new_device
                        self.device_index = new_device
                    else:
                        logger.warning("Fallback: keep using camera %s", current_device)
                        cap = self._open_camera(current_device) or cap

            ok, frame = cap.read()
            if not ok:
                self._record_capture_failure(current_device)
                if (
                    self._single_camera_stats.consecutive_read_failures
                    >= self._capture_reopen_failure_threshold
                ):
                    cap = self._reopen_camera(
                        current_device,
                        cap,
                        reason="repeated_read_failures",
                    )
                    self._single_camera_stats.consecutive_read_failures = 0
                time.sleep(0.05)
                continue
            normalized = self._normalize_frame(frame)
            signature = self._camera_frame_signature(normalized)
            if signature == last_signature:
                consecutive_stale_frames += 1
            else:
                consecutive_stale_frames = 0
            last_signature = signature
            if consecutive_stale_frames >= self._capture_stale_frame_threshold:
                cap = self._reopen_camera(current_device, cap, reason="stale_frame")
                consecutive_stale_frames = 0
                last_signature = None
                continue
            self.update_frame(normalized)
            self._record_capture_success(current_device, normalized)
            self._maybe_save_full_frame_sample(current_device, normalized)

        cap.release()
        logger.info("Camera released")
        self._diagnostics.log_event("camera_released", camera_id=current_device)

    def _capture_loop_device(self, device: int) -> None:
        cap = self._open_camera(device)
        if cap is None:
            fallback = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.update_frame(fallback, camera_id=device)
            return
        last_signature: bytes | None = None
        consecutive_stale_frames = 0

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                self._record_capture_failure(device)
                if (
                    self._camera_stats[device].consecutive_read_failures
                    >= self._capture_reopen_failure_threshold
                ):
                    cap = self._reopen_camera(
                        device,
                        cap,
                        reason="repeated_read_failures",
                    )
                    self._camera_stats[device].consecutive_read_failures = 0
                time.sleep(0.05)
                continue
            normalized = self._normalize_frame(frame)
            signature = self._camera_frame_signature(normalized)
            if signature == last_signature:
                consecutive_stale_frames += 1
            else:
                consecutive_stale_frames = 0
            last_signature = signature
            if consecutive_stale_frames >= self._capture_stale_frame_threshold:
                cap = self._reopen_camera(device, cap, reason="stale_frame")
                consecutive_stale_frames = 0
                last_signature = None
                continue
            self.update_frame(normalized, camera_id=device)
            self._record_capture_success(device, normalized)
            self._maybe_save_full_frame_sample(device, normalized)

        cap.release()
        logger.info("Camera %s released", device)
        self._diagnostics.log_event("camera_released", camera_id=device)

    def iter_mjpeg(self, device: int | None = None) -> Any:
        if device is None:
            if self._camera_list:
                return self._generate_mjpeg_device(self._camera_list[0])
            return self._generate_mjpeg()
        return self._generate_mjpeg_device(device)

    def _generate_mjpeg(self) -> Any:
        yield from self._generate_mjpeg_stream(camera_id=None)

    def _generate_mjpeg_device(self, device: int) -> Any:
        yield from self._generate_mjpeg_stream(camera_id=device)

    def _generate_mjpeg_stream(self, *, camera_id: int | None) -> Any:
        last_revision = -1
        while not self._stop_event.is_set():
            revision = self._wait_for_mjpeg_stream_revision(
                camera_id=camera_id,
                after_revision=last_revision,
            )
            if revision == last_revision:
                continue
            last_revision = revision
            chunk = self._get_or_build_mjpeg_chunk(camera_id=camera_id)
            if chunk is None:
                continue
            yield chunk

    def _invalidate_mjpeg_stream(self, camera_id: int | None) -> None:
        key = camera_id
        with self._stream_state_changed:
            if key not in self._stream_revisions:
                self._stream_revisions[key] = 0
                self._mjpeg_chunk_cache[key] = MjpegChunkCacheEntry()
            self._stream_revisions[key] += 1
            self._mjpeg_chunk_cache[key] = MjpegChunkCacheEntry()
            self._stream_state_changed.notify_all()

    def _invalidate_all_mjpeg_streams(self) -> None:
        with self._stream_state_changed:
            for key in tuple(self._stream_revisions):
                self._stream_revisions[key] += 1
                self._mjpeg_chunk_cache[key] = MjpegChunkCacheEntry()
            self._stream_state_changed.notify_all()

    def _wait_for_mjpeg_stream_revision(
        self,
        *,
        camera_id: int | None,
        after_revision: int,
        timeout: float = 0.1,
    ) -> int:
        key = camera_id
        with self._stream_state_changed:
            if key not in self._stream_revisions:
                self._stream_revisions[key] = 0
                self._mjpeg_chunk_cache[key] = MjpegChunkCacheEntry()
            self._stream_state_changed.wait_for(
                lambda: self._stop_event.is_set()
                or self._stream_revisions[key] != after_revision,
                timeout=timeout,
            )
            return self._stream_revisions[key]

    def _get_or_build_mjpeg_chunk(self, *, camera_id: int | None) -> bytes | None:
        key = camera_id
        while not self._stop_event.is_set():
            with self._stream_state_changed:
                if key not in self._stream_revisions:
                    self._stream_revisions[key] = 0
                    self._mjpeg_chunk_cache[key] = MjpegChunkCacheEntry()
                revision = self._stream_revisions[key]
                cached = self._mjpeg_chunk_cache[key]
                if cached.revision == revision and cached.chunk is not None:
                    return cached.chunk

            frame = self.current_frame if camera_id is None else self.runtime.get_frame(camera_id)
            if frame is None:
                return None

            face_boxes = self._current_face_boxes(camera_id=camera_id)
            self._frame_count += 1
            if camera_id is None:
                processed = self.overlay.draw(
                    frame.copy(),
                    frame_count=self._frame_count,
                    face_boxes=face_boxes,
                )
            else:
                processed = self.overlay.draw(
                    frame.copy(),
                    frame_count=self._frame_count,
                    face_boxes=face_boxes,
                    smooth=False,
                )
            try:
                jpeg = encode_frame_to_jpeg(processed, quality=70)
            except RuntimeError:
                return None

            chunk = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            with self._stream_state_changed:
                current_revision = self._stream_revisions.get(key, revision)
                if current_revision != revision:
                    continue
                self._mjpeg_chunk_cache[key] = MjpegChunkCacheEntry(
                    revision=revision,
                    chunk=chunk,
                )
                return chunk
        return None

    def _current_face_boxes(self, *, camera_id: int | None) -> list[FaceBox]:
        if camera_id is None:
            with self._face_lock:
                return list(self._face_boxes)
        face_lock = self._face_locks_per_cam.get(camera_id)
        if face_lock is None:
            return []
        with face_lock:
            return list(self._face_boxes_per_cam.get(camera_id, []))

    def _configure_fault_handler(self) -> None:
        stream = self._diagnostics.open_fault_handler_stream()
        if stream is None:
            return
        faulthandler.enable(file=stream, all_threads=True)
        self._diagnostics.log_event(
            "faulthandler_enabled",
            fault_log_path=str(getattr(stream, "name", "<unknown>")),
        )

    def _start_worker(
        self,
        *,
        name: str,
        target: Callable[..., None],
        args: tuple[object, ...] = (),
    ) -> None:
        thread = threading.Thread(
            target=self._run_worker,
            args=(name, target, args),
            daemon=True,
            name=name,
        )
        self._worker_threads.append(thread)
        thread.start()

    def _run_worker(
        self,
        worker_name: str,
        target: Callable[..., None],
        args: tuple[object, ...],
    ) -> None:
        self._diagnostics.log_event("worker_started", worker=worker_name)
        try:
            target(*args)
        except Exception as error:
            logger.exception("Worker %s crashed", worker_name)
            self._diagnostics.log_event(
                "worker_crashed",
                worker=worker_name,
                error=repr(error),
            )
            raise
        finally:
            self._diagnostics.log_event("worker_stopped", worker=worker_name)

    def _auto_shutdown_loop(self) -> None:
        self._diagnostics.log_event(
            "auto_shutdown_armed",
            auto_shutdown_sec=self._auto_shutdown_sec,
        )
        if self._stop_event.wait(self._auto_shutdown_sec):
            return
        logger.warning("Auto shutdown triggered after %.3f seconds", self._auto_shutdown_sec)
        self._diagnostics.log_event(
            "auto_shutdown_triggered",
            auto_shutdown_sec=self._auto_shutdown_sec,
        )
        self.stop()

    def _face_detect_worker_centralized(self) -> None:
        """Centralized worker that batches frames from all cameras for GPU inference."""
        logger.info("Starting centralized face detection worker (Batch mode)")
        while not self._stop_event.is_set():
            active_cameras = self._camera_list
            if not active_cameras:
                time.sleep(0.1)
                continue

            frames_to_process = []
            camera_ids = []
            
            for device in active_cameras:
                frame = self.runtime.get_frame(device)
                if frame is not None:
                    frames_to_process.append(frame)
                    camera_ids.append(device)
            
            if not frames_to_process:
                time.sleep(0.01)
                continue

            started_at = time.monotonic()
            
            # Batch Inference on GPU
            try:
                # Use detect_batch if available, else sequential
                detector = self.overlay._detector
                if hasattr(detector, "detect_batch"):
                    _, batch_results = detector.detect_batch(frames_to_process)
                else:
                    batch_results = [self.overlay.detect_faces(f) for f in frames_to_process]

                # Prepare for batch recognition (all faces from all cameras)
                recognition_requests: list[tuple[FrameArray, FaceBox, int, int]] = []

                for i, detections in enumerate(batch_results):
                    device = camera_ids[i]
                    if detections is None:
                        continue

                    source_frame = frames_to_process[i]
                    frame_h, frame_w = source_frame.shape[:2]

                    for row in detections:
                        # Create FaceBox from detector output
                        rx, ry, rw, rh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                        sq_x, sq_y, sq_w, sq_h = to_square(
                            rx,
                            ry,
                            rw,
                            rh,
                            frame_w=frame_w,
                            frame_h=frame_h,
                        )
                        face_box = FaceBox(
                            x=sq_x,
                            y=sq_y,
                            w=sq_w,
                            h=sq_h,
                            confidence=float(row[14]),
                            raw_detection=row,
                        )

                        # (frame, face_box, camera_id, camera_index_in_batch)
                        recognition_requests.append((source_frame, face_box, device, i))

                # Batch Recognize on GPU
                if recognition_requests:
                    batch_inputs = [(r[0], r[1]) for r in recognition_requests]
                    embeddings = self.overlay.extract_embeddings_batch(batch_inputs)

                    # Group results by camera
                    faces_per_cam: dict[int, list[FaceBox]] = {cid: [] for cid in camera_ids}
                    captured_faces_per_cam: dict[int, list[tuple[FrameArray, FaceBox]]] = {
                        cid: [] for cid in camera_ids
                    }

                    for req, embedding in zip(recognition_requests, embeddings, strict=False):
                        source_frame, face_box, device, _ = req
                        if embedding is not None:
                            # Classify owner status using extracted embedding
                            face_box.label, face_box.confidence = (
                                self.overlay._classify_label_with_embedding(
                                    embedding,
                                    face_box,
                                )
                            )
                        captured_faces_per_cam[device].append((source_frame, face_box))
                        faces_per_cam[device].append(face_box)

                    # Update trackers and records
                    for device, faces in faces_per_cam.items():
                        owner_count = sum(1 for face in faces if face.label == "OWNER")
                        subject_count = sum(1 for face in faces if face.label == "SUBJECT")
                        people_count = len(faces)
                        for source_frame, face_box in captured_faces_per_cam[device]:
                            self.overlay._save_face_capture(
                                source_frame,
                                face_box,
                                label=face_box.label,
                                score=face_box.confidence,
                                metadata={
                                    "cameraId": int(device),
                                    "frameCounts": {
                                        "ownerCount": owner_count,
                                        "subjectCount": subject_count,
                                        "peopleCount": people_count,
                                    },
                                },
                            )
                        tracker = self._trackers_per_cam.get(device)
                        if tracker is not None:
                            faces = tracker.update(faces)
                        
                        self._record_owner_presence(faces, camera_id=device)
                        self._record_detection(device, faces, started_at=started_at)
                    
            except Exception as e:
                logger.error("Centralized inference error: %s", e)
                time.sleep(0.1)
            
            # Target 60 FPS for the loop
            elapsed = time.monotonic() - started_at
            target_period = 1.0 / 60.0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

    def _start_face_detection_workers(self) -> None:
        if not self._enable_face_detection:
            self._diagnostics.log_event("face_detection_disabled")
            return
        
        self._start_worker(
            name="face_detect_centralized",
            target=self._face_detect_worker_centralized,
        )

    def _record_http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        self._diagnostics.log_event(
            "http_request",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration_ms, 3),
        )

    def _record_capture_success(self, camera_id: int, frame: FrameArray) -> None:
        stats = self._get_camera_stats(camera_id)
        now = time.monotonic()
        stats.frame_count += 1
        stats.consecutive_read_failures = 0
        stats.last_frame_at = now
        if now - stats.last_capture_log_at >= 5.0:
            stats.last_capture_log_at = now
            self._diagnostics.log_event(
                "camera_capture_heartbeat",
                camera_id=camera_id,
                frame_count=stats.frame_count,
                read_failures=stats.read_failures,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
            )

    def _record_capture_failure(self, camera_id: int) -> None:
        stats = self._get_camera_stats(camera_id)
        stats.read_failures += 1
        stats.consecutive_read_failures += 1
        stats.last_failure_at = time.monotonic()
        if stats.consecutive_read_failures == 1 or stats.consecutive_read_failures % 30 == 0:
            self._diagnostics.log_event(
                "camera_read_failure",
                camera_id=camera_id,
                read_failures=stats.read_failures,
                consecutive_read_failures=stats.consecutive_read_failures,
            )

    def _maybe_save_full_frame_sample(self, camera_id: int, frame: FrameArray) -> None:
        writer = self._full_frame_capture_writer
        if writer is None:
            return

        faces = self.runtime.get_faces(camera_id)
        camera_owner_count = sum(1 for face in faces if getattr(face, "label", "") == "OWNER")
        camera_subject_count = sum(1 for face in faces if getattr(face, "label", "") != "OWNER")
        biometric_status = self.runtime.get_biometric_status()
        global_owner_count = biometric_status.get("ownerCount")
        global_subject_count = biometric_status.get("subjectCount")
        global_people_count = biometric_status.get("peopleCount")
        metadata: dict[str, Any] = {
            "cameraSource": str(self._camera_sources.get(camera_id, camera_id)),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "requestedWidth": int(self.width),
            "requestedHeight": int(self.height),
            "requestedFps": float(self._capture_fps),
            "requestedFourcc": self._capture_fourcc,
            "presence": {
                "cameraPeopleCount": len(faces),
                "cameraOwnerCount": camera_owner_count,
                "cameraSubjectCount": camera_subject_count,
                "globalOwnerPresent": bool(biometric_status["ownerPresent"]),
                "globalOwnerCount": int(global_owner_count or 0),
                "globalSubjectCount": int(global_subject_count or 0),
                "globalPeopleCount": int(global_people_count or 0),
                "globalOwnerSeenAgoMs": biometric_status["ownerSeenAgoMs"],
            },
        }
        if self._room_context_provider is not None:
            try:
                room_context = self._room_context_provider()
            except Exception as error:
                logger.warning("full-frame room context provider failed: %s", error)
            else:
                if room_context is not None:
                    metadata["roomContext"] = room_context
        writer.save(frame, camera_id=camera_id, metadata=metadata)

    def _record_detection(
        self,
        camera_id: int | None,
        faces: list[FaceBox],
        *,
        started_at: float,
    ) -> None:
        stats = self._get_camera_stats(camera_id)
        now = time.monotonic()
        stats.detection_iterations += 1
        stats.last_detection_at = now
        if now - stats.last_detection_log_at >= 5.0:
            stats.last_detection_log_at = now
            self._diagnostics.log_event(
                "face_detection_heartbeat",
                camera_id=camera_id,
                detection_iterations=stats.detection_iterations,
                face_count=len(faces),
                duration_ms=round((now - started_at) * 1000.0, 3),
            )

    def _get_camera_stats(self, camera_id: int | None) -> CameraRuntimeStats:
        if camera_id is None:
            return self._single_camera_stats
        return self._camera_stats.setdefault(camera_id, CameraRuntimeStats())

    def _camera_stats_payload(self) -> list[dict[str, int | float | None]]:
        payload: list[dict[str, int | float | None]] = []
        if not self._camera_list:
            payload.append(self._camera_stats_record(None, self._single_camera_stats))
            return payload
        for camera_id in self._camera_list:
            payload.append(self._camera_stats_record(camera_id, self._get_camera_stats(camera_id)))
        return payload

    def _camera_stats_record(
        self,
        camera_id: int | None,
        stats: CameraRuntimeStats,
    ) -> dict[str, int | float | None]:
        return {
            "camera_id": camera_id,
            "frame_count": stats.frame_count,
            "read_failures": stats.read_failures,
            "consecutive_read_failures": stats.consecutive_read_failures,
            "detection_iterations": stats.detection_iterations,
            "last_frame_at": stats.last_frame_at,
            "last_failure_at": stats.last_failure_at,
            "last_detection_at": stats.last_detection_at,
        }


if __name__ == "__main__":
    raise SystemExit(main())
