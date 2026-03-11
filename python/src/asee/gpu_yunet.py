"""GPU-accelerated YuNet face detector using onnxruntime.

Drop-in replacement for ``cv2.FaceDetectorYN`` that routes inference through
``onnxruntime`` with ``CUDAExecutionProvider`` when available, falling back to
``CPUExecutionProvider`` automatically.

Output format matches ``cv2.FaceDetectorYN.detect()`` exactly:
    (n_detections, np.ndarray of shape [n, 15])
    Each row: [x, y, w, h, kps_x1, kps_y1, ..., kps_x5, kps_y5, confidence]
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional

if TYPE_CHECKING:
    import onnxruntime as ort  # type: ignore[import-untyped]

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

logger = logging.getLogger(__name__)

type FrameArray = npt.NDArray[np.uint8]
type DetectionArray = npt.NDArray[np.float32]

# Anchor stride levels that YuNet uses.
_STRIDES = [8, 16, 32]


class GpuYuNetDetector:
    """onnxruntime-based face detector with the same API as cv2.FaceDetectorYN.

    Parameters
    ----------
    model_path:
        Path to the YuNet ONNX model file.
    input_size:
        (width, height) of the model input.  Use ``setInputSize`` to change
        after construction.
    score_threshold:
        Minimum detection confidence (default 0.6).
    nms_threshold:
        Non-maximum suppression overlap threshold (default 0.3).
    top_k:
        Maximum candidates kept before NMS (default 10).
    device_id:
        CUDA device index (default 0).
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 10,
        device_id: int = 0,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for GpuYuNetDetector. "
                "Install via: pip install onnxruntime-gpu"
            )

        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._input_w, self._input_h = input_size

        providers: list[str | tuple[str, dict[str, object]]] = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]

        options = ort.SessionOptions()
        options.log_severity_level = 3  # suppress verbose ort logs

        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=providers
        )

        # Inspect actual model input shape to ensure correct resizing
        input_meta = self._session.get_inputs()[0]
        shape = input_meta.shape # e.g. [1, 3, 640, 640]
        # YuNet models often have fixed 640x640 or dynamic shapes.
        # If static, use the model's dimensions. If dynamic, use requested input_size.
        self._target_h = shape[2] if isinstance(shape[2], int) else self._input_h
        self._target_w = shape[3] if isinstance(shape[3], int) else self._input_w
        logger.info(
            "GpuYuNetDetector: target inference size set to %dx%d",
            self._target_w,
            self._target_h,
        )

    # ------------------------------------------------------------------
    # Public interface – matches cv2.FaceDetectorYN
    # ------------------------------------------------------------------

    @property
    def active_provider(self) -> str:
        """Return the first active execution provider."""
        providers = self._session.get_providers()
        return str(providers[0]) if providers else "Unknown"

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
        """Update (width, height) that the model will be run at."""
        self._input_w, self._input_h = size

    def set_input_size(self, size: tuple[int, int]) -> None:
        """Snake-case alias used by ``detection_runtime.set_detector_input_size``."""
        self.setInputSize(size)

    def detect(
        self, frame: FrameArray
    ) -> tuple[int, DetectionArray | None]:
        """Run detection on a single *frame*."""
        n, results = self.detect_batch([frame])
        return n, results[0] if results else None

    def detect_batch(
        self, frames: list[FrameArray]
    ) -> tuple[int, list[DetectionArray | None]]:
        """Run sequential detection on multiple *frames* using GPU with IO Binding."""
        if not frames:
            return 0, []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        final_results: list[DetectionArray | None] = []
        total_detections = 0

        with torch.no_grad():
            for frame in frames:
                # 1. Preprocess on GPU
                # YuNet expects raw [0, 255] float32 values – same as
                # cv2.FaceDetectorYN which calls blobFromImage(scalefactor=1.0).
                t = torch.from_numpy(frame).to(device).float()
                t = t.permute(2, 0, 1).unsqueeze(0).contiguous()

                # Resize on GPU to actual model target size
                if t.shape[2] != self._target_h or t.shape[3] != self._target_w:
                    t = functional.interpolate(
                        t,
                        size=(self._target_h, self._target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    t = t.contiguous()  # Ensure memory is contiguous after resize.

                # 2. IO Binding for this single frame
                io_binding = self._session.io_binding()
                io_binding.bind_input(
                    name="input",
                    device_type=device.type,
                    device_id=device.index or 0,
                    element_type=np.float32,
                    shape=t.shape,
                    buffer_ptr=t.data_ptr(),
                )
                for output in self._session.get_outputs():
                    io_binding.bind_output(output.name)

                # 3. Run
                self._session.run_with_iobinding(io_binding)
                outputs = io_binding.copy_outputs_to_cpu()

                # 4. Postprocess
                detections = self._postprocess(outputs)
                if detections is not None:
                    total_detections += len(detections)
                final_results.append(detections)

            return total_detections, final_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: FrameArray) -> npt.NDArray[np.float32]:
        """Legacy helper, logic moved into detect() for IO binding."""
        return np.zeros((1, 3, self._input_h, self._input_w), dtype=np.float32)

    def _postprocess(
        self, outputs: list[npt.NDArray[np.float32]]
    ) -> DetectionArray | None:
        """Decode YuNet anchor-based outputs into cv2.FaceDetectorYN-style rows.

        Output tensor layout (12 tensors):
            [0-2]  cls_8 / cls_16 / cls_32   – shape [1, n, 1], sigmoid applied
            [3-5]  obj_8 / obj_16 / obj_32   – shape [1, n, 1], sigmoid applied
            [6-8]  bbox_8 / bbox_16 / bbox_32 – shape [1, n, 4], raw
            [9-11] kps_8 / kps_16 / kps_32   – shape [1, n, 10], raw

        Box decoding (NanoDet / CenterPoint style: [dx, dy, log_w, log_h]):
            anchor_cx = x_idx * stride           (no 0.5 offset)
            anchor_cy = y_idx * stride
            center_x  = anchor_cx + bbox[:, 0] * stride
            center_y  = anchor_cy + bbox[:, 1] * stride
            w         = exp(bbox[:, 2]) * stride
            h         = exp(bbox[:, 3]) * stride
            x1        = center_x - w / 2
            y1        = center_y - h / 2

        Keypoint decoding:
            kps_x = anchor_cx + kps[:, 2*i]   * stride
            kps_y = anchor_cy + kps[:, 2*i+1] * stride

        Score: geometric mean sqrt(cls * obj).

        Note: empirically verified against cv2.FaceDetectorYN on face_detection_yunet_2023mar.onnx.
        """
        all_boxes: list[npt.NDArray[np.float32]] = []
        all_scores: list[npt.NDArray[np.float32]] = []
        all_kps: list[npt.NDArray[np.float32]] = []

        # Scale factors to map model-output coordinates back to capture space.
        # When input_size equals target_size the factors are 1.0 (no change).
        scale_x = self._input_w / self._target_w
        scale_y = self._input_h / self._target_h

        for i, stride in enumerate(_STRIDES):
            # Use actual model target size for feature-map grid, NOT the capture size.
            # The model always outputs a fixed grid matching _target_w/_target_h.
            cols = math.ceil(self._target_w / stride)
            n_rows = math.ceil(self._target_h / stride)
            n = n_rows * cols

            cls = outputs[i][0].reshape(n)       # [n]
            obj = outputs[i + 3][0].reshape(n)   # [n]
            bbox = outputs[i + 6][0]              # [n, 4]
            kps_raw = outputs[i + 9][0]           # [n, 10]

            score = np.sqrt(np.clip(cls * obj, 0.0, None))

            mask = score > self._score_threshold
            if not np.any(mask):
                continue

            # Anchor positions: grid corners (no 0.5 offset), rows vary slowest
            y_idx = np.repeat(np.arange(n_rows), cols).astype(np.float32)
            x_idx = np.tile(np.arange(cols), n_rows).astype(np.float32)
            cx = x_idx * stride   # anchor top-left x in model space
            cy = y_idx * stride   # anchor top-left y in model space

            # Decode bounding boxes: [dx, dy, log_w, log_h] format
            # center predicted = anchor + delta * stride; size = exp(log_size) * stride
            xc = (cx + bbox[:, 0] * stride) * scale_x
            yc = (cy + bbox[:, 1] * stride) * scale_y
            w_box = np.exp(bbox[:, 2]) * stride * scale_x
            h_box = np.exp(bbox[:, 3]) * stride * scale_y
            x1 = xc - w_box / 2.0
            y1 = yc - h_box / 2.0
            w_box = np.clip(w_box, 0.0, None)
            h_box = np.clip(h_box, 0.0, None)

            # Decode keypoints: linear delta from anchor
            kps_dec = np.empty_like(kps_raw)
            kps_dec[:, 0::2] = (cx[:, np.newaxis] + kps_raw[:, 0::2] * stride) * scale_x
            kps_dec[:, 1::2] = (cy[:, np.newaxis] + kps_raw[:, 1::2] * stride) * scale_y

            boxes_xywh = np.stack(
                [x1[mask], y1[mask], w_box[mask], h_box[mask]], axis=1
            )
            all_boxes.append(boxes_xywh)
            all_scores.append(score[mask])
            all_kps.append(kps_dec[mask])

        if not all_boxes:
            return None

        all_boxes_np = np.concatenate(all_boxes, axis=0)
        all_scores_np = np.concatenate(all_scores, axis=0)
        all_kps_np = np.concatenate(all_kps, axis=0)

        # Pre-NMS top-k selection
        if len(all_scores_np) > self._top_k:
            top_k_idx = np.argsort(all_scores_np)[::-1][: self._top_k]
            all_boxes_np = all_boxes_np[top_k_idx]
            all_scores_np = all_scores_np[top_k_idx]
            all_kps_np = all_kps_np[top_k_idx]

        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            all_boxes_np.tolist(),
            all_scores_np.tolist(),
            self._score_threshold,
            self._nms_threshold,
        )
        if len(indices) == 0:
            return None

        # Build cv2.FaceDetectorYN-compatible rows:
        # [x, y, w, h, kps_x1, kps_y1, ..., kps_x5, kps_y5, score]  ← 15 elements
        rows: list[list[float]] = []
        for idx in np.asarray(indices).ravel():
            x, y, w, h = all_boxes_np[idx]
            kps_flat = all_kps_np[idx].tolist()  # 10 elements
            score_val = float(all_scores_np[idx])
            rows.append([float(x), float(y), float(w), float(h)] + kps_flat + [score_val])

        return np.array(rows, dtype=np.float32)
