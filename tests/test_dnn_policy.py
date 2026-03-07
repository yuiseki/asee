"""Unit tests for the extracted GOD MODE DNN backend policy."""

from __future__ import annotations

import pytest

from asee.dnn_policy import (
    IMPORTANT_OPENCL_WARNING_NOTE,
    emit_opencl_nonfatal_warning_note,
    should_use_opencl_dnn,
)


def test_should_use_opencl_dnn_allows_non_nvidia_devices():
    assert should_use_opencl_dnn("Intel(R) UHD Graphics") is True
    assert should_use_opencl_dnn("Portable Computing Language") is True


def test_should_use_opencl_dnn_allows_nvidia_by_default():
    assert should_use_opencl_dnn("NVIDIA GeForce RTX 3060") is True


def test_should_use_opencl_dnn_can_disable_opencl_explicitly():
    assert should_use_opencl_dnn("NVIDIA GeForce RTX 3060", disable_requested=True) is False


def test_should_use_opencl_dnn_disable_request_wins_over_legacy_allow_unsafe():
    assert (
        should_use_opencl_dnn(
            "NVIDIA GeForce RTX 3060",
            allow_unsafe=True,
            disable_requested=True,
        )
        is False
    )


def test_emit_opencl_nonfatal_warning_note_writes_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    emit_opencl_nonfatal_warning_note("NVIDIA GeForce RTX 3060")

    captured = capsys.readouterr()
    assert IMPORTANT_OPENCL_WARNING_NOTE in captured.err
    assert "NVIDIA GeForce RTX 3060" in captured.err
