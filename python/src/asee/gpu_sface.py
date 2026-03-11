"""GPU-accelerated SFace face recognizer using onnxruntime.

Provides a high-performance alternative to ``cv2.FaceRecognizerSF`` using 
``onnxruntime-gpu`` and ``torch`` for GPU-based preprocessing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
type EmbeddingArray = npt.NDArray[np.float32]


class GpuSFaceRecognizer:
    """SFace recognizer optimized for NVIDIA GPUs."""

    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for GpuSFaceRecognizer. "
                "Install via: pip install onnxruntime-gpu"
            )

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
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=providers
        )

        # SFace fixed input size is 112x112
        self._target_h = 112
        self._target_w = 112
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    @property
    def active_provider(self) -> str:
        """Return the actual execution provider being used."""
        return str(self._session.get_providers()[0])

    def feature(self, face_aligned: FrameArray) -> EmbeddingArray:
        """Extract face embedding from a single ALIGNED face crop."""
        results = self.feature_batch([face_aligned])
        return results[0]

    def feature_batch(self, faces_aligned: list[FrameArray]) -> list[EmbeddingArray]:
        """Extract face embeddings from multiple ALIGNED face crops using GPU batching."""
        if not faces_aligned:
            return []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            # 1. Preprocess all faces on GPU
            tensors = []
            for face in faces_aligned:
                t = torch.from_numpy(face).to(device).float()
                t = t.permute(2, 0, 1).unsqueeze(0).contiguous()

                # SFace expects 112x112
                if t.shape[2] != self._target_h or t.shape[3] != self._target_w:
                    t = functional.interpolate(
                        t,
                        size=(self._target_h, self._target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    t = t.contiguous()
                tensors.append(t)

            # For SFace 2021dec, batch size is often fixed to 1. We therefore
            # keep preprocessing on GPU but execute each face crop sequentially.
            final_embeddings: list[EmbeddingArray] = []

            for t_batch in tensors:
                io_binding = self._session.io_binding()
                io_binding.bind_input(
                    name=self._input_name,
                    device_type=device.type,
                    device_id=device.index or 0,
                    element_type=np.float32,
                    shape=t_batch.shape,
                    buffer_ptr=t_batch.data_ptr(),
                )
                io_binding.bind_output(self._output_name)

                # 3. Run
                self._session.run_with_iobinding(io_binding)
                outputs = io_binding.copy_outputs_to_cpu()

                # 4. Normalize
                embedding = outputs[0].flatten()
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    embedding = embedding / norm
                final_embeddings.append(embedding.astype(np.float32))

            return final_embeddings

    def match(
        self,
        face1_embedding: EmbeddingArray,
        face2_embedding: EmbeddingArray,
        dis_type: int = 0,
    ) -> float:
        """Match two face embeddings using cosine similarity.

        Args:
            face1_embedding: First embedding vector.
            face2_embedding: Second embedding vector.
            dis_type: Distance type (currently only FR_COSINE supported).

        Returns:
            Cosine similarity score (higher is more similar).
        """
        # SFace default is cosine similarity (cv2.FaceRecognizerSF_FR_COSINE = 0)
        f1 = face1_embedding.flatten()
        f2 = face2_embedding.flatten()

        # Cosine similarity: (A . B) / (||A|| * ||B||)
        # Since our feature() already returns normalized vectors, this is just dot product.
        dot = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        return float(dot / (norm1 * norm2))

    def alignCrop(  # noqa: N802
        self,
        frame: FrameArray,
        raw_detection: npt.NDArray[Any],
    ) -> FrameArray:
        """Align and crop face using raw YuNet detection (landmark based).

        Note: Currently uses OpenCV backend for alignment as it's geometry-heavy
        and not easily ported to pure onnxruntime/torch without significant effort.
        """
        # Fallback to a temporary OpenCV instance for geometric alignment
        # This keeps the interface compatible while still offloading feature extraction.
        # Ideally we'd reuse an existing cv2 recognizer instance.
        # For now, we'll implement it manually or borrow it.
        # But to be safe and compatible with GodModeOverlay, we'll add a placeholder
        # and let GodModeOverlay handle the delegation if needed.
        raise NotImplementedError("alignCrop not yet implemented in GpuSFaceRecognizer")
