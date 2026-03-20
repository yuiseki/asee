"""PyTorch FaceNet recognizer backend for ASEE owner classification."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from facenet_pytorch import (  # type: ignore[import-untyped]
    InceptionResnetV1,
    fixed_image_standardization,
)

type FrameArray = npt.NDArray[np.uint8]
type EmbeddingArray = npt.NDArray[np.float32]


class FaceNetPytorchRecognizer:
    """FaceNet embedding backend using PyTorch/CUDA when available."""

    def __init__(self, *, device: str = "cuda") -> None:
        resolved = device
        if resolved == "cuda" and not torch.cuda.is_available():
            resolved = "cpu"
        self._device = torch.device(resolved)
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)

    @property
    def active_provider(self) -> str:
        return str(self._device)

    def feature(self, face_aligned: FrameArray) -> EmbeddingArray:
        embeddings = self.feature_batch([face_aligned])
        return embeddings[0]

    def feature_batch(self, faces_aligned: list[FrameArray]) -> list[EmbeddingArray]:
        if not faces_aligned:
            return []

        tensors = []
        for face in faces_aligned:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).to(dtype=torch.float32)
            tensor = fixed_image_standardization(tensor)
            tensors.append(tensor.unsqueeze(0))

        batch = torch.cat(tensors, dim=0).to(self._device)
        with torch.inference_mode():
            matrix = self._model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
        return [matrix[index] for index in range(matrix.shape[0])]

    def match(
        self,
        reference: EmbeddingArray,
        embedding: EmbeddingArray,
        metric: int = 0,
    ) -> float:
        del metric
        ref = reference.reshape(-1).astype(np.float32, copy=False)
        emb = embedding.reshape(-1).astype(np.float32, copy=False)
        ref_norm = float(np.linalg.norm(ref))
        emb_norm = float(np.linalg.norm(emb))
        if ref_norm <= 1e-6 or emb_norm <= 1e-6:
            return 0.0
        return float(np.dot(ref, emb) / (ref_norm * emb_norm))

    def alignCrop(  # noqa: N802
        self,
        frame: FrameArray,
        raw_detection: npt.NDArray[Any],
    ) -> FrameArray:
        raise NotImplementedError("FaceNetPytorchRecognizer does not implement alignCrop")
