"""Unit tests for the pure path/extension helpers in modules.utilities."""

import os
import tempfile
import unittest

from modules import utilities


class HasImageExtensionTests(unittest.TestCase):
    def test_accepts_known_image_extensions_case_insensitively(self):
        for name in ("a.png", "b.JPG", "c.Jpeg", "/some/dir/d.PNG"):
            self.assertTrue(utilities.has_image_extension(name), name)

    def test_rejects_non_image_extensions(self):
        for name in ("a.mp4", "b.gif", "c.txt", "noextension"):
            self.assertFalse(utilities.has_image_extension(name), name)


class NormalizeOutputPathTests(unittest.TestCase):
    def test_joins_source_and_target_when_output_is_directory(self):
        with tempfile.TemporaryDirectory() as d:
            out = utilities.normalize_output_path("/x/face.jpg", "/y/clip.mp4", d)
            self.assertEqual(out, os.path.join(d, "face-clip.mp4"))

    def test_returns_output_unchanged_when_not_a_directory(self):
        out = utilities.normalize_output_path("/x/face.jpg", "/y/clip.mp4", "/z/out.mp4")
        self.assertEqual(out, "/z/out.mp4")

    def test_returns_output_unchanged_when_paths_missing(self):
        self.assertEqual(utilities.normalize_output_path(None, "/y/clip.mp4", "/o"), "/o")
        self.assertEqual(utilities.normalize_output_path("/x/face.jpg", None, "/o"), "/o")


if __name__ == "__main__":
    unittest.main()
