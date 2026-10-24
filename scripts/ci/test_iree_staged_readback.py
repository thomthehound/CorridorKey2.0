#!/usr/bin/env python3
"""CPU-only contract tests of the IREE Vulkan staging readback sequence."""

from __future__ import annotations

import types
import unittest

import numpy as np

from iree_staged_readback import staged_readback


class _FakeSource:
    def __init__(self, data):
        self.data = np.asarray(data)
    def byte_length(self):
        return self.data.nbytes


class _FakeView:
    def __init__(self, source):
        self.source = source
        self.shape = list(source.data.shape)
    def get_buffer(self):
        return self.source


class _FakeStaging:
    def __init__(self, byte_length):
        self.data = np.empty(byte_length, dtype=np.uint8)
    def map(self):
        return self
    def asarray(self, shape, dtype):
        return self.data.view(dtype).reshape(shape)


class _FakeFence:
    def __init__(self, device, target):
        self.device = device
        self.target = target
    def wait(self):
        if self.device.signaled != self.target:
            raise AssertionError("GPU copy not signaled")
    @staticmethod
    def create_at(semaphore, target):
        return _FakeFence(semaphore, target)


class _FakeAllocator:
    def __init__(self):
        self.staging = None
        self.calls = []
    def allocate_buffer(self, **kwargs):
        self.calls.append(kwargs)
        self.staging = _FakeStaging(kwargs["allocation_size"])
        return self.staging
    def import_host_allocation(self, *args, **kwargs):
        raise AssertionError("Readback must never import a NumPy host pointer")


class _FakeDevice:
    def __init__(self):
        self.allocator = _FakeAllocator()
        self.signaled = 0
        self.copies = 0
    def create_semaphore(self, value):
        self.signaled = value
        return self
    def queue_copy(self, source, dest, *, wait_semaphores, signal_semaphores):
        if self.signaled != wait_semaphores.target:
            raise AssertionError("Incorrect wait fence")
        dest.data[:] = source.data.view(np.uint8).reshape(-1)
        self.signaled = signal_semaphores.target
        self.copies += 1


class StagedReadbackTests(unittest.TestCase):
    def setUp(self):
        self.device = _FakeDevice()
        self.runtime = types.SimpleNamespace(
            MemoryType=types.SimpleNamespace(HOST_LOCAL=1, DEVICE_VISIBLE=2),
            BufferUsage=types.SimpleNamespace(TRANSFER_TARGET=4, MAPPING_SCOPED=8),
            HalFence=_FakeFence,
        )

    def test_readback_uses_staging_and_retains_values_after_staging_reuse(self):
        expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        value = types.SimpleNamespace(
            _buffer_view=_FakeView(_FakeSource(expected)),
            dtype=np.dtype("float32"),
        )
        result = staged_readback(value, self.device, self.runtime)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(self.device.copies, 1)
        self.assertTrue(result.flags.owndata)
        self.assertEqual(self.device.allocator.calls[0]["allocation_size"],
                         expected.nbytes)
        self.device.allocator.staging.data.fill(0)
        np.testing.assert_array_equal(result, expected)

    def test_rejects_wrong_dtype_before_gpu_copy(self):
        value = types.SimpleNamespace(
            _buffer_view=_FakeView(_FakeSource(np.ones(4, dtype=np.int32))),
            dtype=np.dtype("int32"),
        )
        with self.assertRaisesRegex(ValueError, "Expected float32"):
            staged_readback(value, self.device, self.runtime)
        self.assertEqual(self.device.copies, 0)

    def test_rejects_nonfinite_data(self):
        value = types.SimpleNamespace(
            _buffer_view=_FakeView(_FakeSource(
                np.array([1.0, np.nan], dtype=np.float32))),
            dtype=np.dtype("float32"),
        )
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            staged_readback(value, self.device, self.runtime)


if __name__ == "__main__":
    unittest.main()
