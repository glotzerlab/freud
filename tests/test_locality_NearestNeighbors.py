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

    def test_vals(self):
        L = 10 #Box Dimensions
        rcut = 2 #Cutoff radius
        N = 2; # number of particles
        num_neighbors = 1;

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors, strict_cut=False)#Initialize cell list

        points = np.zeros(shape=(2, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [3.0, 0.0, 0.0]
        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.getRsqList()
        npt.assert_equal(neighbor_list[0,0], 1)
        npt.assert_equal(neighbor_list[1,0], 0)
        npt.assert_equal(rsq_list[0,0], 9.0)
        npt.assert_equal(rsq_list[1,0], 9.0)

    def test_two_ways(self):
        L = 10 #Box Dimensions
        rcut = 3 #Cutoff radius
        N = 40; # number of particles
        num_neighbors = 6;

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors)#Initialize cell list

        points = np.random.uniform(-L/2, L/2, (N, 3))
        cl.compute(fbox, points, points)

        neighbor_list = cl.getNeighborList()
        rsq_list = cl.getRsqList()
        # pick particle at random
        pidx = np.random.randint(N)
        nidx = np.random.randint(num_neighbors)
        npt.assert_equal(neighbor_list[pidx,nidx], cl.getNeighbors(pidx)[nidx])
        npt.assert_equal(rsq_list[pidx,nidx], cl.getRsq(pidx)[nidx])

    def test_strict_vals(self):
        L = 10 #Box Dimensions
        rcut = 2 #Cutoff radius
        N = 2; # number of particles
        num_neighbors = 1;

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors, strict_cut=True)#Initialize cell list

        points = np.zeros(shape=(2, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [3.0, 0.0, 0.0]
        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.getRsqList()
        npt.assert_equal(neighbor_list[0,0], cl.getUINTMAX())
        npt.assert_equal(neighbor_list[1,0], cl.getUINTMAX())
        npt.assert_equal(rsq_list[0,0], -1.0)
        npt.assert_equal(rsq_list[1,0], -1.0)

    def test_strict_cutoff(self):
        L = 10 #Box Dimensions
        rcut = 2 #Cutoff radius
        N = 3; # number of particles
        num_neighbors = 2;

        fbox = box.Box.cube(L)#Initialize Box
        cl = locality.NearestNeighbors(rcut, num_neighbors, strict_cut=True)#Initialize cell list

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        cl.compute(fbox, points, points)
        neighbor_list = cl.getNeighborList()
        rsq_list = cl.getRsqList()
        npt.assert_equal(neighbor_list[0,0], 1)
        npt.assert_equal(neighbor_list[0,1], cl.getUINTMAX())
        npt.assert_equal(rsq_list[0,0], 1.0)
        npt.assert_equal(rsq_list[0,1], -1.0)

if __name__ == '__main__':
    unittest.main()
