from freud import locality, trajectory
import numpy as np
import numpy.testing as npt
import unittest

class TestNearestNeighbors(unittest.TestCase):
    def test_neighbor_count(self):
        L = 10 #Box Dimensions
        rcut = 3 #Cutoff radius
        N = 40; # number of particles
        NNeigh = 6;

        box = trajectory.Box(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, NNeigh)#Initialize cell list

        points = np.random.uniform(-L/2, L/2, (N, 3))
        cl.compute(box, points, points)

        self.assertEqual(cl.getNNeigh(), NNeigh)
        self.assertEqual(len(cl.getNeighbors(0)), NNeigh)

if __name__ == '__main__':
    unittest.main()
