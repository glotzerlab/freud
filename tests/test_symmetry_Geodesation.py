import unittest

import numpy as np
import numpy.testing as npt

import freud


class TestGeodesation(unittest.TestCase):
    def test_geodesation(self):
        # Test geodesate, vertices, and neighbor_pairs
        iteration = 1
        geo = freud.symmetry.Geodesation(iteration)

        # Test length of vertices and neighbor_pairs
        npt.assert_equal(len(geo.vertices), geo.n_vertices)
        npt.assert_equal(len(geo.neighbor_pairs), (geo.n_vertices - 1) * 6)

        # Test the first and last elements in vertices
        npt.assert_almost_equal(geo.vertices[0], [0, 0.5257311, 0.8506508], decimal=6)
        npt.assert_almost_equal(geo.vertices[-1], [1, 0, 0], decimal=6)

        # Test the first and last elements in neighbor_pairs
        npt.assert_equal(geo.neighbor_pairs[0][0], 0)
        npt.assert_equal(geo.neighbor_pairs[-1][0], geo.n_vertices - 1)

    def test_geodesation5(self):
        iteration = 5
        geo = freud.symmetry.Geodesation(iteration)

        # Test length of vertices and neighbor_pairs
        npt.assert_equal(len(geo.vertices), 5121)
        npt.assert_equal(len(geo.neighbor_pairs), (geo.n_vertices - 1) * 6)

        # Test the first and last elements in vertices
        npt.assert_almost_equal(geo.vertices[0], [0, 0.5257311, 0.8506508], decimal=6)
        npt.assert_almost_equal(geo.vertices[-1], [0.9424223, -0.3344254, 0], decimal=6)

        # Test the first and last elements in neighbor_pairs
        npt.assert_equal(geo.neighbor_pairs[0][0], 0)
        npt.assert_equal(geo.neighbor_pairs[-1][0], geo.n_vertices - 1)


if __name__ == "__main__":
    unittest.main()
