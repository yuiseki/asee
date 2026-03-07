"""GOD MODE-compatible video server rebuilt on top of extracted asee modules."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import werkzeug.serving

from .http_app import create_http_app
from .overlay import GodModeOverlay
from .server_runtime import SeeingServerRuntime
from .tracking import FaceBox, FaceTracker

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
OWNER_EMBED_PATH = MODELS_DIR / "owner_embedding.npy"

type FrameArray = npt.NDArray[np.uint8]


def encode_frame_to_jpeg(frame: FrameArray, quality: int = 70) -> bytes:
    """Encode a frame into a JPEG payload."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buffer = cv2.imencode(".jpg", frame, encode_param)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buffer.tobytes()


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
        width: int = 1280,
        height: int = 720,
        owner_embedding_path: str | Path = OWNER_EMBED_PATH,
    ) -> None:
        self.port = port
        self.device_index = device_index
        self.title = title
        self.width = width
        self.height = height
        self._camera_list = camera_list or ([device_index] if device_index is not None else [])
        self._cam_interval = cam_interval
        self._cam_index = 0
        self.overlay = GodModeOverlay(
            width=self.width,
            height=self.height,
            face_capture_dir=face_capture_dir,
            face_capture_min_interval=face_capture_min_interval,
            subject_capture_dir=subject_capture_dir,
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
        self._app = create_http_app(self.runtime)
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
        else:
            logger.info("No owner embedding found. All faces labeled SUBJECT.")

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

        if self._camera_list:
            for device in self._camera_list:
                capture_thread = threading.Thread(
                    target=self._capture_loop_device,
                    args=(device,),
                    daemon=True,
                    name=f"capture_{device}",
                )
                capture_thread.start()
            for device in self._camera_list:
                face_thread = threading.Thread(
                    target=self._face_detect_loop_device,
                    args=(device,),
                    daemon=True,
                    name=f"face_detect_{device}",
                )
                face_thread.start()
        elif self.device_index is not None:
            capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="capture",
            )
            capture_thread.start()
            face_thread = threading.Thread(
                target=self._face_detect_loop,
                daemon=True,
                name="face_detect",
            )
            face_thread.start()
        else:
            self.update_frame(np.zeros((self.height, self.width, 3), dtype=np.uint8))

        server = werkzeug.serving.make_server(
            "0.0.0.0",
            self.port,
            self._app,
            threaded=True,
        )
        server.socket.settimeout(1.0)
        self.is_running = True
        self.runtime.set_running(True)

        while not self._stop_event.is_set():
            try:
                server.handle_request()
            except OSError:
                pass

        server.server_close()
        self.is_running = False
        self.runtime.set_running(False)

    def stop(self) -> None:
        self._stop_event.set()

    def _normalize_frame(self, frame: FrameArray) -> FrameArray:
        height, width = frame.shape[:2]
        if width == self.width and height == self.height:
            return frame
        return cast(
            FrameArray,
            cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR),
        )

    def _open_camera(self, device_index: int) -> Any:
        cap = cv2.VideoCapture(device_index)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            logger.error("Cannot open camera device %s", device_index)
            cap.release()
            return None
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera %s opened: requested=%sx%s actual=%sx%s",
            device_index,
            self.width,
            self.height,
            actual_width,
            actual_height,
        )
        return cap

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
                time.sleep(0.05)
                continue
            self.update_frame(frame)
            time.sleep(1 / 30)

        cap.release()
        logger.info("Camera released")

    def _capture_loop_device(self, device: int) -> None:
        cap = self._open_camera(device)
        if cap is None:
            fallback = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.update_frame(fallback, camera_id=device)
            return

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            normalized = self._normalize_frame(frame)
            self.update_frame(normalized, camera_id=device)
            time.sleep(1 / 30)

        cap.release()
        logger.info("Camera %s released", device)

    def _face_detect_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self.current_frame
            if frame is not None:
                with self._detect_lock:
                    faces = self.overlay.detect_faces(frame)
                self._record_owner_presence(faces)
            time.sleep(0.2)

    def _face_detect_loop_device(self, device: int) -> None:
        while not self._stop_event.is_set():
            frame = self.runtime.get_frame(device)
            if frame is not None:
                with self._detect_lock:
                    faces = self.overlay.detect_faces(frame)
                tracker = self._trackers_per_cam.get(device)
                if tracker is not None:
                    faces = tracker.update(faces)
                self._record_owner_presence(faces, camera_id=device)
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
