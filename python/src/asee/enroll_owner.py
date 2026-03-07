"""OWNER enrollment flow owned by asee."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from time import sleep as default_sleep
from typing import Protocol, cast
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np
import numpy.typing as npt

from .model_assets import resolve_model_asset_path
from .overlay import GodModeOverlay

logger = logging.getLogger(__name__)

DEFAULT_OWNER_EMBED_PATH = resolve_model_asset_path("owner_embedding.npy")

type FrameArray = npt.NDArray[np.uint8]
type EmbeddingArray = npt.NDArray[np.float32]
type SleepFn = Callable[[float], None]
type SaveEmbeddingsFn = Callable[[Path, EmbeddingArray], None]


class EnrollmentError(RuntimeError):
    """Raised when OWNER enrollment cannot complete safely."""


class FaceLike(Protocol):
    """Minimal face box interface needed by enrollment."""

    w: int
    h: int
    confidence: float


class OverlayLike(Protocol):
    """Minimal overlay surface needed for enrollment."""

    _detector: object | None
    _recognizer: object | None

    def detect_faces(self, frame: FrameArray) -> list[FaceLike]: ...
    def extract_embedding(
        self,
        frame: FrameArray,
        face_box: FaceLike,
    ) -> EmbeddingArray | None: ...


class VideoCaptureLike(Protocol):
    """Minimal OpenCV VideoCapture-compatible surface."""

    def isOpened(self) -> bool: ...  # noqa: N802
    def read(self) -> tuple[bool, FrameArray | None]: ...
    def release(self) -> None: ...
    def set(self, prop_id: int, value: float) -> bool: ...


def fetch_frame_from_server(server_url: str) -> FrameArray | None:
    """Fetch one snapshot frame from a running asee-compatible server."""
    if not server_url:
        return None
    try:
        with urlopen(f"{server_url}/snapshot", timeout=5) as response:
            status = getattr(response, "status", None)
            if status != 200:
                return None
            payload = response.read()
    except (OSError, URLError):
        return None

    encoded = np.frombuffer(payload, np.uint8)
    return cast(FrameArray | None, cv2.imdecode(encoded, cv2.IMREAD_COLOR))


def save_owner_embeddings(path: Path, embeddings: EmbeddingArray) -> None:
    """Persist the collected OWNER embeddings."""
    np.save(path, embeddings)


def open_video_capture(device_index: int) -> VideoCaptureLike:
    """OpenCV-compatible capture factory used by the enrollment flow."""
    return cast(VideoCaptureLike, cv2.VideoCapture(device_index))


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for OWNER enrollment."""
    parser = argparse.ArgumentParser(description="Enroll OWNER face for GOD MODE")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--server", default="http://localhost:8765")
    parser.add_argument("--save-path", default=str(DEFAULT_OWNER_EMBED_PATH))
    return parser


def _prepare_capture(
    *,
    device_index: int,
    server_url: str,
    fetch_frame: Callable[[str], FrameArray | None],
    video_capture_factory: Callable[[int], VideoCaptureLike],
) -> tuple[bool, VideoCaptureLike | None]:
    if server_url:
        frame = fetch_frame(server_url)
        if frame is not None:
            logger.info("Using server snapshot: %s/snapshot", server_url)
            return True, None
        logger.warning("Server not reachable (%s), falling back to direct camera.", server_url)

    capture = video_capture_factory(device_index)
    if hasattr(capture, "set"):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not capture.isOpened():
        raise EnrollmentError(f"Cannot open camera {device_index}")
    logger.info("Camera %s opened directly.", device_index)
    return False, capture


def run_enrollment(
    *,
    device_index: int = 0,
    n_samples: int = 30,
    server_url: str = "",
    save_path: str | Path = DEFAULT_OWNER_EMBED_PATH,
    overlay: OverlayLike | None = None,
    fetch_frame: Callable[[str], FrameArray | None] = fetch_frame_from_server,
    save_embeddings: SaveEmbeddingsFn = save_owner_embeddings,
    sleep: SleepFn = default_sleep,
    video_capture_factory: Callable[[int], VideoCaptureLike] = open_video_capture,
) -> EmbeddingArray:
    """Collect OWNER face embeddings and persist them."""
    active_overlay: OverlayLike
    if overlay is not None:
        active_overlay = overlay
    else:
        active_overlay = cast(OverlayLike, GodModeOverlay(width=1280, height=720))
    if active_overlay._detector is None:
        raise EnrollmentError("YuNet model not loaded. Cannot enroll.")
    if active_overlay._recognizer is None:
        raise EnrollmentError("SFace model not loaded. Cannot enroll.")

    use_server, capture = _prepare_capture(
        device_index=device_index,
        server_url=server_url,
        fetch_frame=fetch_frame,
        video_capture_factory=video_capture_factory,
    )

    def read_frame() -> FrameArray | None:
        if use_server:
            return fetch_frame(server_url)
        assert capture is not None
        ok, frame = capture.read()
        return frame if ok else None

    embeddings: list[EmbeddingArray] = []
    attempts = 0
    max_attempts = n_samples * 20

    try:
        while len(embeddings) < n_samples and attempts < max_attempts:
            attempts += 1
            frame = read_frame()
            if frame is None:
                sleep(0.1)
                continue

            faces = active_overlay.detect_faces(frame)
            if not faces:
                sleep(0.1)
                continue

            face_box = max(faces, key=lambda face: face.w * face.h)
            if face_box.w < 80 or face_box.h < 80:
                sleep(0.1)
                continue
            if face_box.confidence < 0.75:
                sleep(0.1)
                continue

            embedding = active_overlay.extract_embedding(frame, face_box)
            if embedding is None:
                continue

            embeddings.append(embedding)
            sleep(0.4)
    finally:
        if capture is not None:
            capture.release()

    if len(embeddings) < n_samples // 2:
        raise EnrollmentError(f"Not enough samples ({len(embeddings)}). Enrollment failed.")

    stacked = cast(EmbeddingArray, np.stack(embeddings, axis=0))
    save_embeddings(Path(save_path), stacked)
    return stacked


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for OWNER enrollment."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    try:
        run_enrollment(
            device_index=int(args.device),
            n_samples=int(args.samples),
            server_url=str(args.server),
            save_path=Path(str(args.save_path)),
        )
    except EnrollmentError as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
