import numpy as np
from freud import locality, box
import unittest


class TestNeighborList(unittest.TestCase):
    def test_writable(self):
        L = 10  # Box Dimensions
        rcut = 3  # Cutoff radius
        N = 40  # number of particles
        num_neighbors = 6

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(rcut, num_neighbors)

        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        cl.compute(fbox, points, points)

        # index_i shouldn't be writable in general or users may break
        # the ordered property of the neighbor list
        with self.assertRaises(ValueError):
            cl.nlist.index_i[:] = 0

        # if index_i isn't writable, index_j probably also shouldn't be
        with self.assertRaises(ValueError):
            cl.nlist.index_j[:] = 0

        # the weights array may be useful to write to, though
        cl.nlist.weights[18] = 3
        self.assertEqual(cl.nlist.weights[18], 3)

    def test_validation(self):
        L = 10  # Box Dimensions
        rcut = 3  # Cutoff radius
        N = 40  # number of particles
        num_neighbors = 6

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(rcut, num_neighbors)

        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        points2 = points[:N//2]

        cl.compute(fbox, points, points)

        # should fail in validation when we give inconsistent sized arrays
        with self.assertRaises(RuntimeError):
            cl.nlist.filter_r(fbox, points, points2, 2.5)

        # filter_r should work fine after recomputing using both sets of points
        cl.compute(fbox, points, points2)
        cl.nlist.filter_r(fbox, points, points2, 2.5)

    def test_filter(self):
        L = 10  # Box Dimensions
        rcut = 3  # Cutoff radius
        N = 40  # number of particles
        num_neighbors = 6

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(rcut, num_neighbors)

        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)

        cl.compute(fbox, points, points)

        old_size = len(cl.nlist)

        filt = (cl.nlist.index_j.astype(np.int32) -
                cl.nlist.index_i.astype(np.int32)) % 2 == 0
        cl.nlist.filter(filt)

        self.assertLessEqual(len(cl.nlist), old_size)

        # should be able to further filter
        cl.nlist.filter_r(fbox, points, points, 2.5)

    def test_find_first_index(self):
        L = 10  # Box Dimensions
        rcut = 3  # Cutoff radius
        N = 40  # number of particles
        num_neighbors = 6

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(rcut, num_neighbors)

        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)

        cl.compute(fbox, points, points)
        nlist = cl.nlist

        for (idx, i) in enumerate(nlist.index_i):
            self.assertLessEqual(nlist.find_first_index(i), idx)

    def test_segments(self):
        L = 10  # Box Dimensions
        rcut = 3  # Cutoff radius
        N = 40  # number of particles
        num_neighbors = 6

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(rcut, num_neighbors)

        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)

        cl.compute(fbox, points, points)

        ones = np.ones(len(cl.nlist), dtype=np.float32)
        self.assertTrue(np.allclose(np.add.reduceat(ones, cl.nlist.segments),
                                    6))
        self.assertTrue(np.allclose(cl.nlist.neighbor_counts, 6))

    def test_from_arrays(self):
        index_i = [0, 0, 1, 2, 3]
        index_j = [1, 2, 3, 0, 0]

        # implicit weights
        nlist = locality.NeighborList.from_arrays(4, 4, index_i, index_j)
        self.assertTrue(np.allclose(nlist.weights, 1))

        # explicit weights
        weights = np.ones((len(index_i),))*4.
        nlist = locality.NeighborList.from_arrays(
            4, 4, index_i, index_j, weights)
        self.assertTrue(np.allclose(nlist.weights, 4))

        # too few reference particles
        with self.assertRaises(RuntimeError):
            nlist = locality.NeighborList.from_arrays(
                3, 4, index_i, index_j)

        # too few target particles
        with self.assertRaises(RuntimeError):
            nlist = locality.NeighborList.from_arrays(
                4, 3, index_i, index_j)

        # reference particles not sorted
        with self.assertRaises(RuntimeError):
            nlist = locality.NeighborList.from_arrays(
                4, 4, index_j, index_i)

        # mismatched array sizes
        with self.assertRaises(TypeError):
            nlist = locality.NeighborList.from_arrays(
                4, 4, index_i[:-1], index_j)
        with self.assertRaises(TypeError):
            nlist = locality.NeighborList.from_arrays(
                4, 4, index_i, index_j[:-1])
        with self.assertRaises(TypeError):
            weights = np.ones((len(index_i) - 1,))
            nlist = locality.NeighborList.from_arrays(
                4, 4, index_i, index_j, weights)


    def test_indexing(self):
        L = 10  # Box Dimensions
        rcut = 3  # Cutoff radius
        N = 40  # number of particles
        num_neighbors = 6

        # Initialize Box and cell list
        fbox = box.Box.cube(L)
        cl = locality.NearestNeighbors(rcut, num_neighbors)

        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)

        cl.compute(fbox, points, points)
        print(dir(cl.nlist))
        print(locality.__file__)
        for i, (idx_i, idx_j) in enumerate(cl.nlist):
            assert idx_i == cl.nlist.index_i[i]
            assert idx_j == cl.nlist.index_j[i]



if __name__ == '__main__':
    unittest.main()
