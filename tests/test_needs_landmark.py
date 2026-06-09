"""Unit tests for modules.face_analyser._needs_landmark.

This guards a real optimisation: the 106-point landmark model must be skipped
when only the face swapper is active, and run when a face enhancer or the mouth
mask needs it.  Heavy native deps (insightface/cv2/numpy/tqdm) are stubbed so
the module imports without a GPU or model files.
"""

import importlib
import sys
import types
import unittest


def _install_import_stubs():
    sys.modules.setdefault(
        "insightface",
        types.SimpleNamespace(app=types.SimpleNamespace(FaceAnalysis=object)),
    )
    sys.modules.setdefault("cv2", types.SimpleNamespace(imread=lambda *a, **k: None))
    sys.modules.setdefault("numpy", types.SimpleNamespace(uint8=object))
    sys.modules.setdefault(
        "tqdm", types.SimpleNamespace(tqdm=lambda iterable=None, **k: iterable)
    )
    sys.modules["modules.typing"] = types.SimpleNamespace(Frame=object)
    sys.modules["modules.cluster_analysis"] = types.SimpleNamespace(
        find_cluster_centroids=lambda *a, **k: [],
        find_closest_centroid=lambda *a, **k: (0, None),
    )
    sys.modules["modules.utilities"] = types.SimpleNamespace(
        get_temp_directory_path=lambda p: p,
        create_temp=lambda p: None,
        extract_frames=lambda p: None,
        clean_temp=lambda p: None,
        get_temp_frame_paths=lambda p: [],
    )


def _load_face_analyser():
    _install_import_stubs()
    sys.modules.pop("modules.face_analyser", None)
    return importlib.import_module("modules.face_analyser")


class NeedsLandmarkTests(unittest.TestCase):
    def setUp(self):
        self.fa = _load_face_analyser()
        import modules.globals as g

        self.g = g
        self._saved = (g.mouth_mask, list(g.frame_processors))

    def tearDown(self):
        self.g.mouth_mask, self.g.frame_processors = self._saved[0], self._saved[1]

    def test_swapper_only_does_not_need_landmarks(self):
        self.g.mouth_mask = False
        self.g.frame_processors = ["face_swapper"]
        self.assertFalse(self.fa._needs_landmark())

    def test_mouth_mask_requires_landmarks(self):
        self.g.mouth_mask = True
        self.g.frame_processors = ["face_swapper"]
        self.assertTrue(self.fa._needs_landmark())

    def test_each_enhancer_requires_landmarks(self):
        self.g.mouth_mask = False
        for proc in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
            self.g.frame_processors = ["face_swapper", proc]
            self.assertTrue(self.fa._needs_landmark(), proc)


if __name__ == "__main__":
    unittest.main()
