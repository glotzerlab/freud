import numpy as np
import numpy.testing as npt
import freud
import unittest

class TestGeodesation(unittest.TestCase):
    def test_geodesation(self):
        # Test geodesate, getVertexList and getNeighborList
        iteration = 1
        geo = freud.symmetry.Geodesation(iteration)

        vertices = geo.getVertexList()
        neighbors = geo.getNeighborList()

        # Test length of vertexList and neighborList
        npt.assert_equal(len(vertices), geo.getNVertices())
        npt.assert_equal(len(neighbors), (geo.getNVertices() - 1) * 6)

        # Test the first and last elements in vertexList
        npt.assert_almost_equal(vertices[0], [0, 0.5257311, 0.8506508], decimal=3)
        npt.assert_almost_equal(vertices[-1], [1, 0, 0], decimal=3)


        # Test the first and last elements in neighborList
        npt.assert_equal(neighbors[0][0], 0)
        npt.assert_equal(neighbors[-1][0], geo.getNVertices() - 1)


    def test_geodesation5(self):

        iteration = 5
        geo = freud.symmetry.Geodesation(iteration)

        vertices = geo.getVertexList()
        neighbors = geo.getNeighborList()

        # Test length of vertexList and neighborList
        npt.assert_equal(len(vertices), 5121)
        npt.assert_equal(len(neighbors), (geo.getNVertices() - 1) * 6)

        # Test the first and last elements in vertexList
        npt.assert_almost_equal(vertices[0], [0, 0.5257311, 0.8506508], decimal=3)
        npt.assert_almost_equal(vertices[-1][0], 1, decimal=1)

        # Test the first and last elements in neighborList
        npt.assert_equal(neighbors[0][0], 0)
        npt.assert_equal(neighbors[-1][0], geo.getNVertices() - 1)

if __name__ == '__main__':
    unittest.main()
