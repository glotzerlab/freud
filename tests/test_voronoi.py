from freud import box, voronoi
import numpy as np
import numpy.testing as npt
import unittest

class TestVoronoi(unittest.TestCase):
    def test_basic(self):
        L = 10 #Box Dimensions

        fbox = box.Box.square(L)#Initialize Box
        vor = voronoi.Voronoi(fbox)
        positions = np.random.uniform(-L/2, L/2, size=(50, 2))
        vor.compute(positions, buff=L/2)

        result = vor.getVoronoiPolytopes()

        self.assertEqual(len(result), len(positions))

    def test_voronoi_tess(self):
        vor = voronoi.Voronoi(box.Box.square(10))
        pos = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
        vor.compute(pos)
        npt.assert_equal(vor.getVoronoiPolytopes(), [np.array([[ 1.5,  1.5], [ 0.5,  1.5], [ 0.5,  0.5], [ 1.5,  0.5]])])

    def test_voronoi_neighbors(self):
        vor = voronoi.Voronoi(box.Box.square(10))
        pos = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
        vor.computeNeighbors(pos)
        npt.assert_equal(vor.getNeighbors(1), [[1, 3], [0, 2, 4], [5, 1], [0, 6, 4], [3, 5, 1, 7], [8, 2, 4], [3, 7], [6, 8, 4], [5, 7]])
        npt.assert_equal(vor.getNeighbors(2), [[1, 2, 3, 4, 6], [0, 2, 3, 4, 5, 7], [0, 1, 4, 5, 8], [0, 1, 4, 5, 6, 7], [0, 1, 2, 3, 5, 6, 7, 8], [1, 2, 3, 4, 7, 8], [0, 3, 4, 7, 8], [1, 3, 4, 5, 6, 8], [2, 4, 5, 6, 7]])

if __name__ == '__main__':
    unittest.main()
