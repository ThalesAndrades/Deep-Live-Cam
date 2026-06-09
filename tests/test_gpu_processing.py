"""Unit tests for modules.gpu_processing.

These exercise the pure helpers and the CPU-fallback behaviour of the
``gpu_*`` drop-in replacements.  CUDA is not required: with no GPU build of
OpenCV (and OPENCV_CUDA_PROCESSING unset) ``CUDA_AVAILABLE`` is False, so every
public function takes the plain-cv2 path, which is what we assert here.
"""

import unittest

import cv2
import numpy as np

from modules import gpu_processing as gp


class HelperTests(unittest.TestCase):
    def test_ksize_odd_rounds_up_to_odd(self):
        self.assertEqual(gp._ksize_odd((4, 4)), (5, 5))
        self.assertEqual(gp._ksize_odd((3, 3)), (3, 3))
        self.assertEqual(gp._ksize_odd((6, 6)), (7, 7))

    def test_ksize_odd_preserves_zero(self):
        # (0, 0) tells OpenCV to derive the kernel from sigma; keep it intact.
        self.assertEqual(gp._ksize_odd((0, 0)), (0, 0))

    def test_cv_type_for_matches_channel_count(self):
        gray = np.zeros((4, 4), np.uint8)
        bgr = np.zeros((4, 4, 3), np.uint8)
        bgra = np.zeros((4, 4, 4), np.uint8)
        self.assertEqual(gp._cv_type_for(gray), cv2.CV_8UC1)
        self.assertEqual(gp._cv_type_for(bgr), cv2.CV_8UC3)
        self.assertEqual(gp._cv_type_for(bgra), cv2.CV_8UC4)

    def test_ensure_uint8_clips_and_casts(self):
        src = np.array([[-10.0, 127.5, 300.0]], dtype=np.float32)
        out = gp._ensure_uint8(src)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.tolist(), [[0, 127, 255]])

    def test_ensure_uint8_passes_through_existing_uint8(self):
        src = np.zeros((2, 2), np.uint8)
        # Already uint8: returned as-is (no needless copy/cast).
        self.assertIs(gp._ensure_uint8(src), src)


class CpuFallbackTests(unittest.TestCase):
    """Without a CUDA OpenCV build these run on CPU and must stay correct."""

    def setUp(self):
        self.assertFalse(
            gp.is_gpu_accelerated(),
            "Test environment unexpectedly reports CUDA OpenCV; these tests "
            "assert the CPU fallback path.",
        )
        rng = np.random.default_rng(0)
        self.img = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

    def test_sharpen_zero_strength_is_noop_identity(self):
        # strength <= 0 must short-circuit and return the *same* object.
        self.assertIs(gp.gpu_sharpen(self.img, 0), self.img)
        self.assertIs(gp.gpu_sharpen(self.img, -1), self.img)

    def test_sharpen_changes_pixels_and_keeps_shape_dtype(self):
        out = gp.gpu_sharpen(self.img, 0.8)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse(np.array_equal(out, self.img))

    def test_gaussian_blur_keeps_shape(self):
        out = gp.gpu_gaussian_blur(self.img, (5, 5), 1.0)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_resize_to_explicit_size(self):
        out = gp.gpu_resize(self.img, (24, 16))  # dsize is (width, height)
        self.assertEqual(out.shape[:2], (16, 24))

    def test_flip_horizontal_matches_cv2(self):
        out = gp.gpu_flip(self.img, 1)
        self.assertTrue(np.array_equal(out, cv2.flip(self.img, 1)))

    def test_add_weighted_matches_cv2(self):
        other = self.img[::-1].copy()
        out = gp.gpu_add_weighted(self.img, 0.5, other, 0.5, 0)
        self.assertTrue(np.array_equal(out, cv2.addWeighted(self.img, 0.5, other, 0.5, 0)))


if __name__ == "__main__":
    unittest.main()
