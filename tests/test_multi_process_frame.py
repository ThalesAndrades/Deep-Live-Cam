"""Unit tests for modules.processors.frame.core.multi_process_frame.

Guards the parallel frame-processing contract:
- every frame path is handed to ``process_frames`` exactly once, and
- an exception while processing one frame does not abort the others.

Heavy native deps (insightface) are stubbed so the module imports without a
GPU or model files.  numpy / cv2 / tqdm are the lightweight wheels the test
environment already provides.
"""

import sys
import threading
import types
import unittest


def _load_core():
    sys.modules.setdefault(
        "insightface",
        types.SimpleNamespace(app=types.SimpleNamespace(FaceAnalysis=object)),
    )
    import importlib

    return importlib.import_module("modules.processors.frame.core")


class MultiProcessFrameTests(unittest.TestCase):
    def setUp(self):
        self.core = _load_core()
        import modules.globals as g

        self.g = g
        self._saved_threads = g.execution_threads
        g.execution_threads = 4

    def tearDown(self):
        self.g.execution_threads = self._saved_threads

    def test_processes_every_frame_exactly_once(self):
        seen = []
        lock = threading.Lock()

        def fake_process_frames(source_path, paths, progress):
            with lock:
                seen.extend(paths)

        paths = [f"frame_{i}.png" for i in range(50)]
        self.core.multi_process_frame("src.jpg", paths, fake_process_frames)

        self.assertEqual(sorted(seen), sorted(paths))
        self.assertEqual(len(seen), len(paths))

    def test_exception_in_one_frame_does_not_abort_others(self):
        processed = []
        lock = threading.Lock()

        def fake_process_frames(source_path, paths, progress):
            if paths[0] == "frame_7.png":
                raise RuntimeError("boom")
            with lock:
                processed.extend(paths)

        paths = [f"frame_{i}.png" for i in range(20)]
        # Must not raise even though one frame fails.
        self.core.multi_process_frame("src.jpg", paths, fake_process_frames)

        expected = [p for p in paths if p != "frame_7.png"]
        self.assertEqual(sorted(processed), sorted(expected))

    def test_empty_input_is_a_noop(self):
        calls = []
        self.core.multi_process_frame(
            "src.jpg", [], lambda *a, **k: calls.append(a)
        )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
