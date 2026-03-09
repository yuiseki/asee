"""GPU-accelerated YuNet face detector using onnxruntime.

Drop-in replacement for ``cv2.FaceDetectorYN`` that routes inference through
``onnxruntime`` with ``CUDAExecutionProvider`` when available, falling back to
``CPUExecutionProvider`` automatically.

Output format matches ``cv2.FaceDetectorYN.detect()`` exactly:
    (n_detections, np.ndarray of shape [n, 15])
    Each row: [x, y, w, h, kps_x1, kps_y1, ..., kps_x5, kps_y5, confidence]
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import onnxruntime as ort

try:
    import onnxruntime as ort  # type: ignore

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

FrameArray: TypeAlias = npt.NDArray[np.uint8]
DetectionArray: TypeAlias = npt.NDArray[np.float32]

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

        providers: list[str | tuple[str, dict]] = [
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

    # ------------------------------------------------------------------
    # Public interface – matches cv2.FaceDetectorYN
    # ------------------------------------------------------------------

    @property
    def active_provider(self) -> str:
        """Return the first active execution provider."""
        providers = self._session.get_providers()
        return providers[0] if providers else "Unknown"

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
        """Update (width, height) that the model will be run at."""
        self._input_w, self._input_h = size

    def set_input_size(self, size: tuple[int, int]) -> None:
        """Snake-case alias used by ``detection_runtime.set_detector_input_size``."""
        self.setInputSize(size)

    def detect(
        self, frame: FrameArray
    ) -> tuple[int, DetectionArray | None]:
        """Run detection on *frame* and return (n, detections).

        Compatible with ``cv2.FaceDetectorYN.detect()``.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            # 1. Preprocess on GPU
            # Upload to GPU
            t = torch.from_numpy(frame).to(device).float()
            # HWC -> BCHW
            t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
            
            # Resize on GPU
            if t.shape[2] != self._input_h or t.shape[3] != self._input_w:
                t = F.interpolate(t, size=(self._input_h, self._input_w), mode='bilinear', align_corners=False)
            
            # 2. IO Binding for zero-copy input
            io_binding = self._session.io_binding()
            io_binding.bind_input(
                name="input",
                device_type=device.type,
                device_id=device.index or 0,
                element_type=np.float32,
                shape=t.shape,
                buffer_ptr=t.data_ptr(),
            )
            
            # Bind outputs to CPU for existing postprocess (to keep logic simple for now)
            for output in self._session.get_outputs():
                io_binding.bind_output(output.name)
            
            # 3. Run
            self._session.run_with_iobinding(io_binding)
            outputs = io_binding.copy_outputs_to_cpu()
            
            # 4. Postprocess
            detections = self._postprocess(outputs)
            if detections is None or len(detections) == 0:
                return 0, None
            return len(detections), detections

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

        Box decoding (FCOS / NanoDet distance-based):
            x1 = anchor_cx - bbox[:, 0] * stride
            y1 = anchor_cy - bbox[:, 1] * stride
            x2 = anchor_cx + bbox[:, 2] * stride
            y2 = anchor_cy + bbox[:, 3] * stride

        Keypoint decoding:
            kps_x = anchor_cx + kps[:, 2*i]   * stride
            kps_y = anchor_cy + kps[:, 2*i+1] * stride

        Score: geometric mean sqrt(cls * obj).
        """
        all_boxes: list[npt.NDArray[np.float32]] = []
        all_scores: list[npt.NDArray[np.float32]] = []
        all_kps: list[npt.NDArray[np.float32]] = []

        for i, stride in enumerate(_STRIDES):
            cols = math.ceil(self._input_w / stride)
            n_rows = math.ceil(self._input_h / stride)
            n = n_rows * cols

            cls = outputs[i][0].reshape(n)       # [n]
            obj = outputs[i + 3][0].reshape(n)   # [n]
            bbox = outputs[i + 6][0]              # [n, 4]
            kps_raw = outputs[i + 9][0]           # [n, 10]

            score = np.sqrt(np.clip(cls * obj, 0.0, None))

            mask = score > self._score_threshold
            if not np.any(mask):
                continue

            # Generate anchor centers (columns vary fastest)
            y_idx = np.repeat(np.arange(n_rows), cols).astype(np.float32)
            x_idx = np.tile(np.arange(cols), n_rows).astype(np.float32)
            cx = (x_idx + 0.5) * stride
            cy = (y_idx + 0.5) * stride

            # Decode bounding boxes
            x1 = cx - bbox[:, 0] * stride
            y1 = cy - bbox[:, 1] * stride
            x2 = cx + bbox[:, 2] * stride
            y2 = cy + bbox[:, 3] * stride
            w_box = np.clip(x2 - x1, 0.0, None)
            h_box = np.clip(y2 - y1, 0.0, None)

            # Decode keypoints
            kps_dec = np.empty_like(kps_raw)
            kps_dec[:, 0::2] = cx[:, np.newaxis] + kps_raw[:, 0::2] * stride
            kps_dec[:, 1::2] = cy[:, np.newaxis] + kps_raw[:, 1::2] * stride

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
