"""GOD MODE-compatible video server rebuilt on top of extracted asee modules."""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import platform
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import werkzeug.serving
from turbojpeg import TurboJPEG

from .diagnostics import (
    DiagnosticsLogger,
    JsonlDiagnosticsLogger,
    MemoryMonitor,
    NullDiagnosticsLogger,
    build_default_diagnostics_log_path,
)
from .http_app import create_http_app
from .model_assets import resolve_model_asset_path
from .overlay import GodModeOverlay
from .server_runtime import SeeingServerRuntime
from .tracking import FaceBox, FaceTracker

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
OWNER_EMBED_PATH = resolve_model_asset_path("owner_embedding.npy")

type FrameArray = npt.NDArray[np.uint8]


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


jpeg_encoder = TurboJPEG()


def encode_frame_to_jpeg(frame: FrameArray, quality: int = 70) -> bytes:
    """Encode a frame into a JPEG payload using TurboJPEG."""
    try:
        # TurboJPEG is significantly faster than cv2.imencode
        return cast(bytes, jpeg_encoder.encode(frame, quality=quality))
    except Exception as e:
        logger.error("TurboJPEG encode failed: %s, falling back to OpenCV", e)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, buffer = cv2.imencode(".jpg", frame, encode_param)
        if not ok:
            raise RuntimeError("JPEG encode failed") from e
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
            fps=10.0 if len(camera_ids) > 1 else 30.0,
            fourcc="MJPG",
        )
    else:
        base = (
            CaptureSettings(width=640, height=360, fps=10.0, fourcc="MJPG")
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
        help="既定キャプチャ profile。auto は低リスク既定、720p は 1280x720 を要求する",
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
        choices=["opencv", "onnxruntime"],
        default="opencv",
        help="顔検出バックエンド: opencv (既定) または onnxruntime (CUDA GPU 推論)",
    )
    return parser


def build_server_from_args(args: argparse.Namespace) -> GodModeVideoServer:
    """Build a server instance from parsed CLI arguments."""
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
    return GodModeVideoServer(
        port=int(args.port),
        device_index=device_index,
        camera_list=camera_list,
        cam_interval=int(args.cam_interval),
        title=str(args.title),
        face_capture_dir=str(args.face_capture_dir) or None,
        face_capture_min_interval=float(args.face_capture_min_interval),
        subject_capture_dir=str(args.subject_capture_dir) or None,
        allow_live_camera=allow_live_camera,
        diagnostics_logger=diagnostics_logger,
        memory_log_interval_sec=float(args.memory_log_interval_sec),
        auto_shutdown_sec=float(args.auto_shutdown_sec),
        capture_profile=str(args.capture_profile),
        width=None if args.width is None else int(args.width),
        height=None if args.height is None else int(args.height),
        fps=None if args.fps is None else float(args.fps),
        fourcc=None if args.fourcc is None else str(args.fourcc),
        opencv_threads=None if args.opencv_threads is None else int(args.opencv_threads),
        enable_face_detection=not bool(args.disable_face_detect),
        detection_backend=str(args.detection_backend),
    )


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
        cam_interval: int = 60,
        title: str = "GOD MODE",
        face_capture_dir: str | None = None,
        face_capture_min_interval: float = 1.0,
        subject_capture_dir: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        fourcc: str | None = None,
        opencv_threads: int | None = None,
        owner_embedding_path: str | Path = OWNER_EMBED_PATH,
        allow_live_camera: bool = False,
        diagnostics_logger: DiagnosticsLogger | None = None,
        memory_log_interval_sec: float = 30.0,
        auto_shutdown_sec: float = 0.0,
        enable_face_detection: bool = True,
        capture_profile: str = "auto",
        detection_backend: str = "opencv",
    ) -> None:
        requested_camera_list = camera_list or ([device_index] if device_index is not None else [])
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
        )
        self.runtime = SeeingServerRuntime(
            title=title,
            overlay=self.overlay,
            camera_ids=tuple(self._camera_list),
            jpeg_encoder=encode_frame_to_jpeg,
            jpeg_quality=80,
            stream_factory=self.iter_mjpeg,
        )
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
        self._app = create_http_app(self.runtime, request_logger=self._record_http_request)
        self._load_owner_embedding(owner_embedding_path)

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

    def update_overlay_text(self, *, caption: str = "", prediction: str = "") -> None:
        self.runtime.update_overlay_text(caption=caption, prediction=prediction)

    def get_biometric_status(self) -> dict[str, bool | int | float | None]:
        self.runtime.set_running(self.is_running)
        return self.runtime.get_biometric_status()

    def _load_owner_embedding(self, path: str | Path) -> None:
        resolved = Path(path)
        if resolved.exists():
            self.runtime.load_owner_embedding(resolved)
            logger.info("Owner embedding loaded: %s", resolved)
            self._diagnostics.log_event("owner_embedding_loaded", path=str(resolved))
        else:
            logger.info("No owner embedding found. All faces labeled SUBJECT.")
            self._diagnostics.log_event("owner_embedding_missing", path=str(resolved))

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

        server = werkzeug.serving.make_server(
            "0.0.0.0",
            self.port,
            self._app,
            threaded=True,
        )
        server.socket.settimeout(1.0)
        self.is_running = True
        self.runtime.set_running(True)
        self._diagnostics.log_event("server_started", port=self.port)

        try:
            while not self._stop_event.is_set():
                try:
                    server.handle_request()
                except OSError as error:
                    logger.warning("HTTP server request loop error: %s", error)
                    self._diagnostics.log_event("http_server_oserror", error=repr(error))
        finally:
            server.server_close()
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

    def _open_camera(self, device_index: int) -> Any:
        self._diagnostics.log_event(
            "camera_open_attempt",
            camera_id=device_index,
            requested_width=self.width,
            requested_height=self.height,
            requested_fps=self._capture_fps,
            requested_fourcc=self._capture_fourcc,
        )
        cap = cv2.VideoCapture(device_index)
        fourcc = cv2.VideoWriter_fourcc(*self._capture_fourcc)  # type: ignore[attr-defined]
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self._capture_fps)
        if not cap.isOpened():
            logger.error("Cannot open camera device %s", device_index)
            self._diagnostics.log_event("camera_open_failed", camera_id=device_index)
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
            actual_width=actual_width,
            actual_height=actual_height,
            actual_fps=actual_fps,
            actual_fourcc=actual_fourcc,
        )
        return cap

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
                time.sleep(0.05)
                continue
            normalized = self._normalize_frame(frame)
            self.update_frame(normalized)
            self._record_capture_success(current_device, normalized)
            time.sleep(self._capture_period)

        cap.release()
        logger.info("Camera released")
        self._diagnostics.log_event("camera_released", camera_id=current_device)

    def _capture_loop_device(self, device: int) -> None:
        cap = self._open_camera(device)
        if cap is None:
            fallback = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.update_frame(fallback, camera_id=device)
            return

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                self._record_capture_failure(device)
                time.sleep(0.05)
                continue
            normalized = self._normalize_frame(frame)
            self.update_frame(normalized, camera_id=device)
            self._record_capture_success(device, normalized)
            time.sleep(self._capture_period)

        cap.release()
        logger.info("Camera %s released", device)
        self._diagnostics.log_event("camera_released", camera_id=device)

    def _face_detect_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self.current_frame
            if frame is not None:
                started_at = time.monotonic()
                with self._detect_lock:
                    faces = self.overlay.detect_faces(frame)
                self._record_owner_presence(faces)
                self._record_detection(None, faces, started_at=started_at)
            time.sleep(0.2)

    def _face_detect_loop_device(self, device: int) -> None:
        while not self._stop_event.is_set():
            frame = self.runtime.get_frame(device)
            if frame is not None:
                started_at = time.monotonic()
                with self._detect_lock:
                    faces = self.overlay.detect_faces(frame)
                tracker = self._trackers_per_cam.get(device)
                if tracker is not None:
                    faces = tracker.update(faces)
                self._record_owner_presence(faces, camera_id=device)
                self._record_detection(device, faces, started_at=started_at)
            time.sleep(0.2)

    def iter_mjpeg(self, device: int | None = None) -> Any:
        if device is None:
            if self._camera_list:
                return self._generate_mjpeg_device(self._camera_list[0])
            return self._generate_mjpeg()
        return self._generate_mjpeg_device(device)

    def _generate_mjpeg(self) -> Any:
        while not self._stop_event.is_set():
            frame = self.current_frame
            if frame is None:
                time.sleep(0.05)
                continue

            with self._face_lock:
                face_boxes = list(self._face_boxes)

            self._frame_count += 1
            processed = self.overlay.draw(
                frame.copy(),
                frame_count=self._frame_count,
                face_boxes=face_boxes,
            )
            try:
                jpeg = encode_frame_to_jpeg(processed, quality=70)
            except RuntimeError:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1 / 15)

    def _generate_mjpeg_device(self, device: int) -> Any:
        while not self._stop_event.is_set():
            frame = self.runtime.get_frame(device)
            if frame is None:
                time.sleep(0.05)
                continue

            face_lock = self._face_locks_per_cam.get(device)
            if face_lock is None:
                face_boxes: list[FaceBox] = []
            else:
                with face_lock:
                    face_boxes = list(self._face_boxes_per_cam.get(device, []))

            self._frame_count += 1
            processed = self.overlay.draw(
                frame.copy(),
                frame_count=self._frame_count,
                face_boxes=face_boxes,
                smooth=False,
            )
            try:
                jpeg = encode_frame_to_jpeg(processed, quality=70)
            except RuntimeError:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1 / 15)

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

    def _start_face_detection_workers(self) -> None:
        if not self._enable_face_detection:
            self._diagnostics.log_event("face_detection_disabled")
            return
        if self._camera_list:
            for device in self._camera_list:
                self._start_worker(
                    name=f"face_detect_{device}",
                    target=self._face_detect_loop_device,
                    args=(device,),
                )
            return
        self._start_worker(
            name="face_detect",
            target=self._face_detect_loop,
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
