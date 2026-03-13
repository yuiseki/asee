"""GOD MODE-style overlay runtime rebuilt on top of extracted asee primitives."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from .capture_writer import FaceCaptureWriter
from .detection_runtime import YunetDetectionPipeline, to_square
from .dnn_policy import (
    emit_opencl_nonfatal_warning_note,
    should_use_opencl_dnn,
)
from .model_assets import resolve_model_asset_path
from .owner_policy import OWNER_COSINE_THRESHOLD, OWNER_TOPK, keep_largest_owner
from .tracking import FaceBox, FaceTracker

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_YUNET_PATH = str(resolve_model_asset_path("face_detection_yunet_2023mar.onnx"))
DEFAULT_SFACE_PATH = str(resolve_model_asset_path("face_recognition_sface_2021dec.onnx"))

COLOR_CYAN = (255, 200, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (40, 40, 255)
COLOR_GREEN = (40, 210, 40)
COLOR_MAGENTA = (200, 40, 200)

type FrameArray = npt.NDArray[np.uint8]
type EmbeddingArray = npt.NDArray[np.float32]


def _select_dnn_backend() -> tuple[int, int]:
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("DNN backend selected: CUDA")
            return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    except (cv2.error, AttributeError):
        pass

    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                disable_value = str(
                    os.environ.get("GOD_MODE_DISABLE_OPENCL_DNN", "")
                ).strip()
                disable_requested = disable_value.lower() in {"1", "true", "yes", "on"}
                allow_unsafe_value = str(
                    os.environ.get("GOD_MODE_ALLOW_UNSAFE_OPENCL_DNN", "")
                ).strip()
                allow_unsafe = allow_unsafe_value.lower() in {"1", "true", "yes", "on"}
                device_name = ""
                try:
                    device_name = cv2.ocl.Device.getDefault().name()
                except (cv2.error, AttributeError):
                    device_name = ""
                if not should_use_opencl_dnn(
                    device_name,
                    allow_unsafe=allow_unsafe,
                    disable_requested=disable_requested,
                ):
                    logger.warning(
                        (
                            "OpenCL runtime is available (%s), "
                            "but DNN OpenCL is disabled; falling back to CPU"
                        ),
                        device_name or "unknown OpenCL device",
                    )
                    return cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU
                if "nvidia" in device_name.lower():
                    emit_opencl_nonfatal_warning_note(device_name)
                logger.info("DNN backend selected: OpenCL")
                return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL
    except (cv2.error, AttributeError):
        pass

    logger.info("DNN backend selected: CPU")
    return cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU


class GodModeOverlay:
    """Draw the HUD and run face detection / classification."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        yunet_path: str = DEFAULT_YUNET_PATH,
        sface_path: str = DEFAULT_SFACE_PATH,
        face_capture_dir: str | None = None,
        face_capture_min_interval: float = 1.0,
        subject_capture_dir: str | None = None,
        detection_backend: str = "opencv",
    ) -> None:
        self.width = width
        self.height = height
        self.caption = ""
        self.prediction = ""
        self._owner_embeddings: EmbeddingArray | None = None

        # GPU recognizer needs a CPU counterpart for alignCrop (OpenCV implementation)
        self._cpu_recognizer = self._load_sface(sface_path)

        if detection_backend == "onnxruntime":
            self._detector = self._load_yunet_onnxruntime(yunet_path, width, height)
            self._recognizer = self._load_sface_onnxruntime(sface_path)
        else:
            self._detector = self._load_yunet(yunet_path, width, height)
            self._recognizer = self._cpu_recognizer
        self._haar = self._load_haar()
        self._tracker = FaceTracker(alpha=0.4, max_lost_frames=2, min_hits=3)
        self._yunet_pipeline: YunetDetectionPipeline | None = None

        self._face_capture_writer = (
            FaceCaptureWriter(face_capture_dir, min_interval_sec=face_capture_min_interval)
            if face_capture_dir is not None
            else None
        )
        self._subject_capture_writer = (
            FaceCaptureWriter(subject_capture_dir, min_interval_sec=face_capture_min_interval)
            if subject_capture_dir is not None
            else None
        )
        self._grid_overlay = self._create_grid_overlay(width, height)

    def _create_grid_overlay(self, width: int, height: int) -> FrameArray:
        """Create a static grid overlay image."""
        grid = np.zeros((height, width, 3), dtype=np.uint8)
        step = 80
        for x in range(0, width, step):
            cv2.line(grid, (x, 0), (x, height), COLOR_CYAN, 1)
        for y in range(0, height, step):
            cv2.line(grid, (0, y), (width, y), COLOR_CYAN, 1)
        return grid

    def set_caption(self, text: str) -> None:
        self.caption = text

    def set_prediction(self, text: str) -> None:
        self.prediction = text

    def set_owner_embedding(self, embedding: EmbeddingArray) -> None:
        if embedding.ndim == 1:
            self._owner_embeddings = cast(EmbeddingArray, embedding.reshape(1, -1))
        else:
            self._owner_embeddings = embedding.copy()

    def detect_faces(self, frame: FrameArray) -> list[FaceBox]:
        if self._detector is not None:
            faces = self._detect_yunet(frame)
        else:
            faces = self._detect_haar(frame)
        return keep_largest_owner(faces)

    def smooth_faces(self, face_boxes: list[FaceBox]) -> list[FaceBox]:
        return self._tracker.update(face_boxes)

    def extract_embedding(
        self,
        frame: FrameArray,
        face_box: FaceBox,
    ) -> EmbeddingArray | None:
        """Sequential single-face embedding extraction."""
        results = self.extract_embeddings_batch([(frame, face_box)])
        return results[0] if results else None

    def extract_embeddings_batch(
        self,
        requests: list[tuple[FrameArray, FaceBox]],
    ) -> list[EmbeddingArray | None]:
        """Extract multiple face embeddings efficiently."""
        if not requests or self._recognizer is None:
            return [None] * len(requests)

        aligned_crops: list[FrameArray] = []
        valid_indices: list[int] = []

        # 1. Perform alignment on CPU (OpenCV alignCrop is sequential)
        for i, (frame, face_box) in enumerate(requests):
            aligned = None
            if face_box.raw_detection is not None and self._cpu_recognizer is not None:
                try:
                    aligned = self._cpu_recognizer.alignCrop(frame, face_box.raw_detection)
                except Exception as error:
                    logger.debug("alignCrop failed: %s", error)

            if aligned is None:
                # Manual crop fallback
                frame_h, frame_w = frame.shape[:2]
                x1, y1 = max(0, face_box.x), max(0, face_box.y)
                x2, y2 = min(x1 + face_box.w, frame_w), min(y1 + face_box.h, frame_h)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        aligned = cast(
                            FrameArray,
                            cv2.resize(crop, (112, 112)),
                        )

            if aligned is not None:
                aligned_crops.append(aligned)
                valid_indices.append(i)

        # 2. Perform feature extraction on GPU in batch (or high-speed sequential)
        results: list[EmbeddingArray | None] = [None] * len(requests)
        if aligned_crops:
            try:
                # Use batch interface if available
                if hasattr(self._recognizer, "feature_batch"):
                    embeddings = self._recognizer.feature_batch(aligned_crops)
                else:
                    embeddings = [self._recognizer.feature(ac) for ac in aligned_crops]

                for idx, emb in zip(valid_indices, embeddings, strict=False):
                    results[idx] = emb
            except Exception as error:
                logger.error("Batch feature extraction failed: %s", error)

        return results

    def draw(
        self,
        frame: FrameArray,
        *,
        frame_count: int = 0,
        face_boxes: list[FaceBox] | None = None,
        smooth: bool = True,
    ) -> np.ndarray:
        del frame_count
        # Draw on a copy to keep the original frame clean
        output = cast(
            FrameArray,
            cv2.addWeighted(frame, 0.96, self._grid_overlay, 0.04, 0),
        )
        visible_faces = self.smooth_faces(face_boxes or []) if smooth else (face_boxes or [])

        for face_box in visible_faces:
            self._draw_face_box(output, face_box)
        return output

    def _load_yunet(self, path: str, width: int, height: int) -> Any:
        if not os.path.exists(path):
            logger.warning("YuNet model not found: %s, falling back to Haar", path)
            return None
        backend_id, target_id = _select_dnn_backend()
        for next_backend_id, next_target_id in [
            (backend_id, target_id),
            (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU),
        ]:
            try:
                return cv2.FaceDetectorYN.create(
                    path,
                    "",
                    (width, height),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=10,
                    backend_id=next_backend_id,
                    target_id=next_target_id,
                )
            except Exception as error:
                if next_backend_id == cv2.dnn.DNN_BACKEND_DEFAULT:
                    logger.warning("YuNet load failed: %s", error)
                    return None
                logger.warning("YuNet load failed with backend=%s: %s", next_backend_id, error)
        return None

    def _load_yunet_onnxruntime(self, path: str, width: int, height: int) -> Any:
        """Load YuNet via onnxruntime with CUDA, falling back to OpenCV on error."""
        try:
            from .gpu_yunet import GpuYuNetDetector

            det = GpuYuNetDetector(
                model_path=path,
                input_size=(width, height),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=10,
            )
            logger.info("YuNet loaded via onnxruntime (%s)", det.active_provider)
            return det
        except Exception as error:
            logger.warning(
                "onnxruntime YuNet load failed, falling back to OpenCV: %s", error
            )
            return self._load_yunet(path, width, height)

    def _load_sface_onnxruntime(self, path: str) -> Any:
        """Load SFace via onnxruntime with CUDA, falling back to OpenCV on error."""
        try:
            from .gpu_sface import GpuSFaceRecognizer

            rec = GpuSFaceRecognizer(model_path=path)
            logger.info("SFace loaded via onnxruntime (%s)", rec.active_provider)
            return rec
        except Exception as error:
            logger.warning(
                "onnxruntime SFace load failed, falling back to OpenCV: %s", error
            )
            return self._load_sface(path)

    def _load_sface(self, path: str) -> Any:
        if not os.path.exists(path):
            logger.warning("SFace model not found: %s", path)
            return None
        backend_id, target_id = _select_dnn_backend()
        for next_backend_id, next_target_id in [
            (backend_id, target_id),
            (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU),
        ]:
            try:
                return cv2.FaceRecognizerSF.create(
                    path,
                    "",
                    backend_id=next_backend_id,
                    target_id=next_target_id,
                )
            except Exception as error:
                if next_backend_id == cv2.dnn.DNN_BACKEND_DEFAULT:
                    logger.warning("SFace load failed: %s", error)
                    return None
                logger.warning("SFace load failed with backend=%s: %s", next_backend_id, error)
        return None

    @staticmethod
    def _load_haar() -> Any:
        cv2_data = getattr(cv2, "data", None)
        haarcascades = "" if cv2_data is None else str(getattr(cv2_data, "haarcascades", ""))
        path = haarcascades + "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(path)
        return None if classifier.empty() else classifier

    def _detect_yunet(self, frame: FrameArray) -> list[FaceBox]:
        if self._detector is None:
            return []
        pipeline = self._yunet_pipeline
        if pipeline is None:
            pipeline = YunetDetectionPipeline(
                detector=self._detector,
                classify_label=lambda current_frame, face_box: self._classify_label(
                    cast(FrameArray, current_frame),
                    face_box,
                ),
            )
            self._yunet_pipeline = pipeline
        return pipeline.detect_faces(frame)

    def _detect_haar(self, frame: FrameArray) -> list[FaceBox]:
        if self._haar is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(50, 50),
        )
        if len(detections) == 0:
            return []
        frame_h, frame_w = frame.shape[:2]
        result: list[FaceBox] = []
        for x, y, width, height in detections:
            square_x, square_y, square_w, square_h = to_square(
                int(x),
                int(y),
                int(width),
                int(height),
                frame_w=frame_w,
                frame_h=frame_h,
            )
            result.append(FaceBox(x=square_x, y=square_y, w=square_w, h=square_h))
        return result

    def _classify_label(self, frame: FrameArray, face_box: FaceBox) -> tuple[str, float]:
        embedding = self.extract_embedding(frame, face_box)
        if embedding is None:
            label = "SUBJECT"
            score = face_box.confidence
            self._save_face_capture(frame, face_box, label=label, score=score)
            return label, score
        label, score = self._classify_label_with_embedding(embedding, face_box)
        self._save_face_capture(frame, face_box, label=label, score=score)
        return label, score

    def _classify_label_with_embedding(
        self, embedding: EmbeddingArray, face_box: FaceBox
    ) -> tuple[str, float]:
        if self._recognizer is None or self._owner_embeddings is None:
            return "SUBJECT", face_box.confidence

        try:
            scores = sorted(
                [
                    self._recognizer.match(
                        reference.reshape(1, -1),
                        embedding,
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )
                    for reference in self._owner_embeddings
                ],
                reverse=True,
            )
            score = float(np.mean(scores[:OWNER_TOPK]))
            if score >= OWNER_COSINE_THRESHOLD:
                # Note: face_capture_writer needs the aligned frame, 
                # which is not available here without re-aligning.
                # For now we focus on recognition speed.
                return "OWNER", score
        except Exception as error:
            logger.debug("Label classification failed: %s", error)
            return "SUBJECT", face_box.confidence

        return "SUBJECT", face_box.confidence

    def _save_face_capture(
        self,
        frame: FrameArray,
        face_box: FaceBox,
        *,
        label: str,
        score: float,
    ) -> None:
        writer = None
        if label == "OWNER":
            writer = self._face_capture_writer
        elif label == "SUBJECT":
            writer = self._subject_capture_writer
        if writer is None:
            return

        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, face_box.x)
        y1 = max(0, face_box.y)
        x2 = min(x1 + face_box.w, frame_w)
        y2 = min(y1 + face_box.h, frame_h)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        if crop.size <= 0:
            return
        writer.save(crop, score)

    def _draw_grid(self, image: FrameArray, width: int, height: int) -> None:
        step = 80
        grid = image.copy()
        for x in range(0, width, step):
            cv2.line(grid, (x, 0), (x, height), COLOR_CYAN, 1)
        for y in range(0, height, step):
            cv2.line(grid, (0, y), (width, y), COLOR_CYAN, 1)
        cv2.addWeighted(grid, 0.04, image, 0.96, 0, image)

    @staticmethod
    def _put_text_outlined(
        image: FrameArray,
        text: str,
        position: tuple[int, int],
        *,
        font: int = cv2.FONT_HERSHEY_PLAIN,
        scale: float = 1.2,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
    ) -> None:
        outline_thickness = thickness + 2
        cv2.putText(
            image,
            text,
            position,
            font,
            scale,
            (0, 0, 0),
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

    def _draw_face_box(self, image: FrameArray, face_box: FaceBox) -> None:
        is_owner = face_box.label == "OWNER"
        box_color = COLOR_MAGENTA if is_owner else COLOR_WHITE
        corner_color = COLOR_MAGENTA if is_owner else COLOR_CYAN
        label_color = COLOR_MAGENTA if is_owner else COLOR_CYAN

        x, y, width, height = face_box.x, face_box.y, face_box.w, face_box.h
        self._draw_dashed_rect(image, x, y, x + width, y + height, box_color, dash=10)

        arm = max(16, width // 6)
        corner_thickness = 5 if is_owner else 3
        top_left, top_right, bottom_left, bottom_right = face_box.corners()

        for (point_x, point_y), (delta_x, delta_y) in [
            (top_left, (arm, 0)),
            (top_left, (0, arm)),
            (top_right, (-arm, 0)),
            (top_right, (0, arm)),
            (bottom_left, (arm, 0)),
            (bottom_left, (0, -arm)),
            (bottom_right, (-arm, 0)),
            (bottom_right, (0, -arm)),
        ]:
            cv2.line(
                image,
                (point_x, point_y),
                (point_x + delta_x, point_y + delta_y),
                corner_color,
                corner_thickness,
            )

        guide_length = max(8, width // 12)
        guide_thickness = max(1, corner_thickness - 2)
        center_x = x + width // 2
        center_y = y + height // 2
        cv2.line(image, (center_x, y), (center_x, y + guide_length), corner_color, guide_thickness)
        cv2.line(
            image,
            (center_x, y + height),
            (center_x, y + height - guide_length),
            corner_color,
            guide_thickness,
        )
        cv2.line(image, (x, center_y), (x + guide_length, center_y), corner_color, guide_thickness)
        cv2.line(
            image,
            (x + width, center_y),
            (x + width - guide_length, center_y),
            corner_color,
            guide_thickness,
        )

        label_x = x
        label_y = max(y - 8, 12)
        self._put_text_outlined(
            image,
            face_box.label,
            (label_x, label_y),
            scale=2.6,
            color=label_color,
            thickness=2,
        )
        if face_box.confidence < 1.0:
            (label_width, _), _ = cv2.getTextSize(
                face_box.label + " ",
                cv2.FONT_HERSHEY_PLAIN,
                2.6,
                2,
            )
            self._put_text_outlined(
                image,
                f"{face_box.confidence:.0%}",
                (label_x + label_width, label_y),
                scale=2.6,
                color=COLOR_WHITE,
                thickness=2,
            )

    @staticmethod
    def _draw_dashed_rect(
        image: FrameArray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        *,
        dash: int = 10,
        gap: int = 6,
        thickness: int = 1,
    ) -> None:
        segments = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for (start_x, start_y), (end_x, end_y) in segments:
            length = int(np.hypot(end_x - start_x, end_y - start_y))
            if length == 0:
                continue
            delta_x = (end_x - start_x) / length
            delta_y = (end_y - start_y) / length
            position = 0
            while position < length:
                end = min(position + dash, length)
                point_1 = (int(start_x + delta_x * position), int(start_y + delta_y * position))
                point_2 = (int(start_x + delta_x * end), int(start_y + delta_y * end))
                cv2.line(image, point_1, point_2, color, thickness)
                position += dash + gap
