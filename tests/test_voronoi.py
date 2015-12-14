from freud import trajectory, voronoi
import numpy as np
import numpy.testing as npt
import unittest

class TestVoronoi(unittest.TestCase):
    def test_basic(self):
        L = 10 #Box Dimensions

        box = trajectory.Box(L, L, is2D=True)#Initialize Box
        vor = voronoi.Voronoi(box)
        positions = np.random.uniform(-L/2, L/2, size=(50, 2))
        vor.compute(positions, buff=L/2)

        result = vor.getVoronoiPolytopes()

        self.assertEqual(len(result), len(positions))

if __name__ == '__main__':
    unittest.main()
