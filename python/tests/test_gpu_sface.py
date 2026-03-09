"""Unit tests for GPU-accelerated SFace recognizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asee.gpu_sface import GpuSFaceRecognizer

# Mock SFace model path
SFACE_MODEL_PATH = "src/asee/models/face_recognition_sface_2021dec.onnx"

def _make_mock_session():
    mock = MagicMock()
    # SFace returns [1, 128] feature vector
    output_meta = MagicMock()
    output_meta.name = "output"
    mock.get_outputs.return_value = [output_meta]
    
    input_meta = MagicMock()
    input_meta.name = "input"
    mock.get_inputs.return_value = [input_meta]
    
    mock.get_providers.return_value = ["CUDAExecutionProvider"]
    
    # Mock IO Binding
    mock_binding = MagicMock()
    mock_binding.copy_outputs_to_cpu.return_value = [np.random.randn(1, 128).astype(np.float32)]
    mock.io_binding.return_value = mock_binding
    
    return mock

@pytest.fixture
def mock_ort_session():
    with patch("onnxruntime.InferenceSession", return_value=_make_mock_session()):
        yield

def test_sface_initialization(mock_ort_session):
    recognizer = GpuSFaceRecognizer(SFACE_MODEL_PATH)
    assert recognizer is not None
    assert recognizer.active_provider == "CUDAExecutionProvider"

def test_sface_feature_shape(mock_ort_session):
    recognizer = GpuSFaceRecognizer(SFACE_MODEL_PATH)
    dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
    
    feature = recognizer.feature(dummy_face)
    assert isinstance(feature, np.ndarray)
    # Output is flattened by _postprocess logic in our implementation
    assert feature.shape == (128,)

def test_sface_feature_normalization(mock_ort_session):
    recognizer = GpuSFaceRecognizer(SFACE_MODEL_PATH)
    dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
    
    feature = recognizer.feature(dummy_face)
    # Check L2 norm is approx 1.0
    norm = np.linalg.norm(feature)
    assert pytest.approx(norm, rel=1e-5) == 1.0
