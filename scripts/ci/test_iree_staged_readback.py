#!/usr/bin/env python3
"""CPU-only contract tests of the IREE Vulkan staging readback sequence."""

from __future__ import annotations

import types
import unittest

import numpy as np

from iree_staged_readback import VulkanReadback, staged_readback


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
        # Mirror IREE 3.13's numpy::SimpleNewFromData:
        # dtype_descr.itemsize must be a concrete integer, not the
        # descriptor belonging to the np.float32 scalar class.
        if not isinstance(dtype, np.dtype):
            raise TypeError('Expected a NumPy dtype descriptor')
        if not isinstance(dtype.itemsize, int):
            raise TypeError('Invalid dtype descriptor itemsize')
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


    def test_native_transfer_remains_default(self):
        expected = np.arange(16, dtype=np.float32).reshape(2, 8)
        counts = [0]

        def to_host():
            counts[0] += 1
            return expected

        value = types.SimpleNamespace(to_host=to_host)
        reader = VulkanReadback(self.device, self.runtime)
        np.testing.assert_array_equal(reader.read(value), expected)
        np.testing.assert_array_equal(reader.read(value), expected)
        self.assertEqual(reader.strategy, "runtime")
        self.assertEqual(counts[0], 2)
        self.assertEqual(self.device.allocator.calls, [])
        self.assertEqual(self.device.copies, 0)

    def test_alignment_failure_switches_once_and_reuses_staging(self):
        expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        counts = [0]

        def to_host():
            counts[0] += 1
            raise ValueError(
                "host allocation import pointer does not satisfy Vulkan "
                "minImportedHostPointerAlignment 4096"
            )

        value = types.SimpleNamespace(
            _buffer_view=_FakeView(_FakeSource(expected)),
            dtype=np.dtype("float32"),
            to_host=to_host,
        )
        reader = VulkanReadback(self.device, self.runtime)
        result1 = reader.read(value)
        np.testing.assert_array_equal(result1, expected)
        result2 = reader.read(value)
        np.testing.assert_array_equal(result2, expected)
        self.assertEqual(reader.strategy, "staging")
        self.assertEqual(counts[0], 1)
        self.assertEqual(len(self.device.allocator.calls), 1)
        self.assertEqual(self.device.copies, 2)
        self.assertTrue(result1.flags.owndata)
        self.assertTrue(result2.flags.owndata)
        self.device.allocator.staging.data.fill(0)
        np.testing.assert_array_equal(result1, expected)
        np.testing.assert_array_equal(result2, expected)

    def test_other_transfer_failures_are_not_suppressed(self):
        def to_host():
            raise ValueError("DEVICE_LOST: unrelated transfer failure")

        value = types.SimpleNamespace(to_host=to_host)
        reader = VulkanReadback(self.device, self.runtime)
        with self.assertRaisesRegex(ValueError, "DEVICE_LOST"):
            reader.read(value)
        self.assertEqual(reader.strategy, "runtime")
        self.assertEqual(self.device.allocator.calls, [])

    def test_staging_buffers_are_cached_by_size_not_shape(self):
        data1 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        data2 = np.arange(24, dtype=np.float32).reshape(4, 6)
        reader = VulkanReadback(self.device, self.runtime, force_staging=True)
        for data in (data1, data2):
            value = types.SimpleNamespace(
                _buffer_view=_FakeView(_FakeSource(data)),
                dtype=data.dtype,
            )
            np.testing.assert_array_equal(reader.read(value), data)
        self.assertEqual(len(self.device.allocator.calls), 1)
        self.assertEqual(self.device.copies, 2)


if __name__ == "__main__":
    unittest.main()
