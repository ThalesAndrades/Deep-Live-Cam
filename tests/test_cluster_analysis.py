"""Unit tests for modules.cluster_analysis.find_closest_centroid.

``cluster_analysis`` imports ``sklearn`` at module load, but the function under
test only depends on numpy.  We stub a minimal ``sklearn.cluster`` module so the
import succeeds without pulling in the (heavy) scikit-learn dependency.
"""

import sys
import types
import unittest

import numpy as np

# Provide a lightweight sklearn stub before importing the module under test.
if "sklearn" not in sys.modules:
    _sklearn = types.ModuleType("sklearn")
    _cluster = types.ModuleType("sklearn.cluster")
    _cluster.KMeans = object  # never instantiated by find_closest_centroid
    _sklearn.cluster = _cluster
    sys.modules["sklearn"] = _sklearn
    sys.modules["sklearn.cluster"] = _cluster

from modules import cluster_analysis  # noqa: E402


class FindClosestCentroidTests(unittest.TestCase):
    def test_returns_index_and_vector_of_most_similar_centroid(self):
        centroids = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        embedding = np.array([0.1, 0.9, 0.2])  # closest to centroid index 1
        index, vector = cluster_analysis.find_closest_centroid(centroids, embedding)
        self.assertEqual(int(index), 1)
        self.assertTrue(np.array_equal(vector, centroids[1]))

    def test_picks_exact_match(self):
        centroids = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        index, vector = cluster_analysis.find_closest_centroid(centroids, [0.0, 1.0])
        self.assertEqual(int(index), 1)
        self.assertTrue(np.array_equal(vector, centroids[1]))

    def test_malformed_input_returns_none(self):
        # Ragged centroids raise inside np.dot -> the function swallows it.
        result = cluster_analysis.find_closest_centroid([[1.0, 0.0], [0.0]], [1.0, 0.0])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
