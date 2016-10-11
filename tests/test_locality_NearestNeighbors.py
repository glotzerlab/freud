from freud import locality, box
import numpy as np
import numpy.testing as npt
import unittest

class TestNearestNeighbors(unittest.TestCase):
    def test_neighbor_count(self):
        L = 10 #Box Dimensions
        rcut = 3 #Cutoff radius
        N = 40; # number of particles
        num_neighbors = 6;

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors)#Initialize cell list

        points = np.random.uniform(-L/2, L/2, (N, 3))
        cl.compute(fbox, points, points)

        self.assertEqual(cl.getNumNeighbors(), num_neighbors)
        self.assertEqual(len(cl.getNeighbors(0)), num_neighbors)

if __name__ == '__main__':
    unittest.main()
