#!/usr/bin/env python3
"""Explicit IREE HAL staging readback for device-local Vulkan tensors.

Use the same public HAL staging-copy sequence as IREE 3.13.0's
DeviceArray._copy_to_host fallback without host-pointer import. The
DeviceArray buffer-view accessor remains private in IREE 3.13.0; keep
this access isolated here until the runtime's host import path is fixed.
"""

from __future__ import annotations

import numpy as np


def staged_readback(value, device, runtime) -> np.ndarray:
    """Synchronously copy a device array through HAL-managed host memory.

    A copied NumPy array owns its data and does not depend on the
    lifetime of the temporary Vulkan buffer or mapped memory.
    """
    view = value._buffer_view
    source = view.get_buffer()
    if np.dtype(value.dtype) != np.dtype("float32"):
        raise ValueError(f"Expected float32 GPU output, got {value.dtype}")
    memory_type = int(runtime.MemoryType.HOST_LOCAL | runtime.MemoryType.DEVICE_VISIBLE)
    usage = int(runtime.BufferUsage.TRANSFER_TARGET | runtime.BufferUsage.MAPPING_SCOPED)
    staging = device.allocator.allocate_buffer(
        memory_type=memory_type,
        allowed_usage=usage,
        allocation_size=source.byte_length(),
    )
    semaphore = device.create_semaphore(0)
    device.queue_copy(
        source, staging,
        wait_semaphores=runtime.HalFence.create_at(semaphore, 0),
        signal_semaphores=runtime.HalFence.create_at(semaphore, 1),
    )
    runtime.HalFence.create_at(semaphore, 1).wait()
    mapped = staging.map()
    # IREE's native mapped.asarray uses dtype_descr.itemsize as an int.
    # np.float32 is a scalar type; np.dtype('float32') is the descriptor.
    result = np.array(mapped.asarray(view.shape, np.dtype('float32')), copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Vulkan device result")
    return result
