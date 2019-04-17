import numpy as np
from freud import locality, box
import unittest


class TestNeighborList(unittest.TestCase):

    def setup_nl(self,
                 L=10,
                 rcut=3,
                 N=40,
                 num_neighbors=6):
        # Define values
        self.L = L
        self.rcut = rcut
        self.N = N
        self.num_neighbors = num_neighbors

        # Initialize Box and cell list
        self.fbox = box.Box.cube(self.L)
        self.cl = locality.NearestNeighbors(self.rcut, self.num_neighbors)
        self.points = np.random.uniform(
            -self.L/2, self.L/2, (self.N, 3)).astype(np.float32)

    def test_writable(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)

        # index_i shouldn't be writable in general or users may break
        # the ordered property of the neighbor list
        with self.assertRaises(ValueError):
            self.cl.nlist.index_i[:] = 0

        # if index_i isn't writable, index_j probably also shouldn't be
        with self.assertRaises(ValueError):
            self.cl.nlist.index_j[:] = 0

        # the weights array may be useful to write to, though
        self.cl.nlist.weights[18] = 3
        self.assertEqual(self.cl.nlist.weights[18], 3)

    def test_validation(self):
        self.setup_nl()
        np.random.seed(0)
        points2 = self.points[:self.N//2]
        self.cl.compute(self.fbox, self.points, self.points)

        # should fail in validation when we give inconsistent sized arrays
        with self.assertRaises(RuntimeError):
            self.cl.nlist.filter_r(self.fbox, self.points, points2, 2.5)

        # filter_r should work fine after recomputing using both sets of points
        self.cl.compute(self.fbox, self.points, points2)
        self.cl.nlist.filter_r(self.fbox, self.points, points2, 2.5)

    def test_filter(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)
        old_size = len(self.cl.nlist)
        filt = (self.cl.nlist.index_j.astype(np.int32) -
                self.cl.nlist.index_i.astype(np.int32)) % 2 == 0
        self.cl.nlist.filter(filt)
        self.assertLessEqual(len(self.cl.nlist), old_size)

        # should be able to further filter
        self.cl.nlist.filter_r(self.fbox, self.points, self.points, 2.5)

    def test_find_first_index(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)
        nlist = self.cl.nlist
        for (idx, i) in enumerate(nlist.index_i):
            self.assertLessEqual(nlist.find_first_index(i), idx)

    def test_segments(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)
        ones = np.ones(len(self.cl.nlist), dtype=np.float32)
        self.assertTrue(
            np.allclose(np.add.reduceat(ones, self.cl.nlist.segments), 6))
        self.assertTrue(np.allclose(self.cl.nlist.neighbor_counts, 6))

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
        self.setup_nl()
        np.random.seed(0)

        # Ensure that empty NeighborLists have the right shape
        self.assertEqual(self.cl.nlist[:].shape, (0, 2))

        # Make sure indexing the NeighborList is the same as indexing arrays
        self.cl.compute(self.fbox, self.points, self.points)
        for i, (idx_i, idx_j) in enumerate(self.cl.nlist):
            self.assertEqual(idx_i, self.cl.nlist.index_i[i])
            self.assertEqual(idx_j, self.cl.nlist.index_j[i])

        for i, j in self.cl.nlist:
            self.assertNotEqual(i, j)

    def test_nl_size(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)
        self.assertEqual(len(self.cl.nlist), len(self.cl.nlist.index_i))
        self.assertEqual(len(self.cl.nlist), len(self.cl.nlist.index_j))

    def test_index_error(self):
        self.setup_nl()
        with self.assertRaises(IndexError):
            nbonds = len(self.cl.nlist)
            self.cl.nlist[nbonds+1]

    def test_index_writable(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)

        with self.assertRaises(TypeError):
            self.cl.nlist[:, 0] = 0

        with self.assertRaises(TypeError):
            self.cl.nlist[1, :] = 0

        with self.assertRaises(TypeError):
            self.cl.nlist[:] = 0

        with self.assertRaises(TypeError):
            self.cl.nlist[0, 0] = 0


if __name__ == '__main__':
    unittest.main()
