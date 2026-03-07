"""Pure DNN backend policy helpers extracted from GOD MODE overlay code."""

from __future__ import annotations

import sys

IMPORTANT_OPENCL_WARNING_NOTE = "!!!IMPORTANT NOTE: warnings of -cl-no-subgroup-ifp is not fatal!!!"


def should_use_opencl_dnn(
    device_name: str,
    allow_unsafe: bool = False,
    disable_requested: bool = False,
) -> bool:
    """Return whether the OpenCL DNN backend should be used on this device."""
    lowered = str(device_name or "").strip().lower()
    if not lowered:
        return True
    if disable_requested:
        return False
    # Legacy flag retained for compatibility with older startup flows.
    _ = allow_unsafe
    return True


def emit_opencl_nonfatal_warning_note(device_name: str = "") -> None:
    """Write a clear note when the noisy ocl4dnn warning is expected."""
    note = IMPORTANT_OPENCL_WARNING_NOTE
    if device_name:
        note = f"{note} device={device_name}"
    print(note, file=sys.stderr, flush=True)
