#!/usr/bin/env python3
"""Vulkan output readback with a narrowly scoped IREE host-import fallback.

IREE 3.13.0 may select NumPy host-buffer import on Vulkan devices whose
minImportedHostPointerAlignment the NumPy allocation does not satisfy.
Prefer IREE's native transfer path and use HAL-owned staging only when
that exact defect occurs. Later transfers amortize staging allocations
and synchronization objects. Keep all driver-specific handling here.
"""

from __future__ import annotations

import numpy as np


class VulkanReadback:
    """Synchronous float32 readback with reusable HAL staging resources.

    Calls are sequential. Returned staging arrays own their storage, so
    a later GPU transfer cannot overwrite an earlier result.
    """

    def __init__(self, device, runtime, *, force_staging=False):
        self.device = device
        self.runtime = runtime
        self._staging = {}
        self._semaphore = None
        self._signal_value = 0
        self.strategy = "staging" if force_staging else "runtime"

    def read(self, value) -> np.ndarray:
        if self.strategy == "runtime":
            try:
                result = np.asarray(value.to_host())
            except Exception as exc:
                # HAL status errors can be raised as backend-specific
                # exception types; only the diagnosed alignment failure
                # authorizes the staging fallback.
                if "minImportedHostPointerAlignment" not in str(exc):
                    raise
                self.strategy = "staging"
            else:
                return self._check_result(result)
        return self._read_staging(value)

    @staticmethod
    def _check_result(result):
        if result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("Nonfinite or non-float32 Vulkan result")
        return result

    def _read_staging(self, value):
        view = value._buffer_view
        source = view.get_buffer()
        if np.dtype(value.dtype) != np.dtype("float32"):
            raise ValueError(f"Expected float32 GPU output, got {value.dtype}")

        size = source.byte_length()
        staging = self._staging.get(size)
        if staging is None:
            memory_type = int(
                self.runtime.MemoryType.HOST_LOCAL
                | self.runtime.MemoryType.DEVICE_VISIBLE
            )
            usage = int(
                self.runtime.BufferUsage.TRANSFER_TARGET
                | self.runtime.BufferUsage.MAPPING_SCOPED
            )
            staging = self.device.allocator.allocate_buffer(
                memory_type=memory_type,
                allowed_usage=usage,
                allocation_size=size,
            )
            self._staging[size] = staging

        if self._semaphore is None:
            self._semaphore = self.device.create_semaphore(0)
        next_value = self._signal_value + 1
        wait = self.runtime.HalFence.create_at(
            self._semaphore, self._signal_value
        )
        signal = self.runtime.HalFence.create_at(self._semaphore, next_value)
        self.device.queue_copy(
            source, staging,
            wait_semaphores=wait,
            signal_semaphores=signal,
        )
        signal.wait()
        self._signal_value = next_value

        mapped = staging.map()
        # MappedMemory.asarray requires an actual NumPy dtype descriptor;
        # np.float32 is a scalar class and fails in the native IREE binding.
        result = np.array(
            mapped.asarray(view.shape, np.dtype("float32")), copy=True
        )
        return self._check_result(result)


def staged_readback(value, device, runtime) -> np.ndarray:
    """One-shot explicit staging readback (also useful for regression tests)."""
    return VulkanReadback(device, runtime, force_staging=True).read(value)
