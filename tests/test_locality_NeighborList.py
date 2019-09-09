import numpy as np
import freud.locality
import unittest
from util import make_box_and_random_points


class TestNeighborList(unittest.TestCase):

    def setUp(self):
        # Define values
        self.L = 10
        self.N = 40

        self.query_args = dict(mode="nearest", num_neighbors=6, r_guess=3,
                               exclude_ii=True)

        # Initialize Box and cell list
        box, points = make_box_and_random_points(self.L, self.N)
        self.nq = freud.locality.LinkCell(box, self.query_args['r_guess']/10,
                                          points)
        self.nlist = self.nq.query(points,
                                   self.query_args).toNeighborList()

    def test_writable(self):
        # index_i shouldn't be writable in general or users may break
        # the ordered property of the neighbor list
        with self.assertRaises(ValueError):
            self.nlist.index_i[:] = 0

        # if index_i isn't writable, index_j probably also shouldn't be
        with self.assertRaises(ValueError):
            self.nlist.index_j[:] = 0

        # the weights array may be useful to write to, though
        self.nlist.weights[18] = 3
        self.assertEqual(self.nlist.weights[18], 3)

    def test_validation(self):
        points2 = self.nq.points[:self.N//2]

        # should fail in validation when we give inconsistent sized arrays
        with self.assertRaises(RuntimeError):
            self.nlist.filter_r(self.nq.box, self.nq.points, points2, 2.5)

        # filter_r should work fine after recomputing using both sets of points
        nlist = self.nq.query(points2, self.query_args).toNeighborList()
        nlist.filter_r(self.nq.box, self.nq.points, points2, 2.5)

    def test_filter(self):
        old_size = len(self.nlist)
        filt = (self.nlist.index_j.astype(np.int32) -
                self.nlist.index_i.astype(np.int32)) % 2 == 0
        self.nlist.filter(filt)
        self.assertLessEqual(len(self.nlist), old_size)

        # should be able to further filter
        self.nlist.filter_r(self.nq.box, self.nq.points, self.nq.points, 2.5)

    def test_find_first_index(self):
        nlist = self.nlist
        for (idx, i) in enumerate(nlist.index_i):
            self.assertLessEqual(nlist.find_first_index(i), idx)

    def test_segments(self):
        ones = np.ones(len(self.nlist), dtype=np.float32)
        self.assertTrue(
            np.allclose(np.add.reduceat(ones, self.nlist.segments), 6))
        self.assertTrue(np.allclose(self.nlist.neighbor_counts, 6))

    def test_from_arrays(self):
        index_i = [0, 0, 1, 2, 3]
        index_j = [1, 2, 3, 0, 0]

        # implicit weights
        nlist = freud.locality.NeighborList.from_arrays(4, 4, index_i, index_j)
        self.assertTrue(np.allclose(nlist.weights, 1))

        # explicit weights
        weights = np.ones((len(index_i),))*4.
        nlist = freud.locality.NeighborList.from_arrays(
            4, 4, index_i, index_j, weights)
        self.assertTrue(np.allclose(nlist.weights, 4))

        # too few reference particles
        with self.assertRaises(RuntimeError):
            nlist = freud.locality.NeighborList.from_arrays(
                3, 4, index_i, index_j)

        # too few target particles
        with self.assertRaises(RuntimeError):
            nlist = freud.locality.NeighborList.from_arrays(
                4, 3, index_i, index_j)

        # reference particles not sorted
        with self.assertRaises(RuntimeError):
            nlist = freud.locality.NeighborList.from_arrays(
                4, 4, index_j, index_i)

        # mismatched array sizes
        with self.assertRaises(ValueError):
            nlist = freud.locality.NeighborList.from_arrays(
                4, 4, index_i[:-1], index_j)
        with self.assertRaises(ValueError):
            nlist = freud.locality.NeighborList.from_arrays(
                4, 4, index_i, index_j[:-1])
        with self.assertRaises(ValueError):
            weights = np.ones((len(index_i) - 1,))
            nlist = freud.locality.NeighborList.from_arrays(
                4, 4, index_i, index_j, weights)

    def test_indexing(self):
        # Ensure that empty NeighborLists have the right shape
        nlist = self.nq.query(np.empty((0, 3)),
                              self.query_args).toNeighborList()
        self.assertEqual(nlist[:].shape, (0, 2))

        # Make sure indexing the NeighborList is the same as indexing arrays
        for i, (idx_i, idx_j) in enumerate(self.nlist):
            self.assertEqual(idx_i, self.nlist.index_i[i])
            self.assertEqual(idx_j, self.nlist.index_j[i])

        print(self.nlist[:])
        for i, j in self.nlist:
            self.assertNotEqual(i, j)

    def test_nl_size(self):
        self.assertEqual(len(self.nlist), len(self.nlist.index_i))
        self.assertEqual(len(self.nlist), len(self.nlist.index_j))

    def test_index_error(self):
        with self.assertRaises(IndexError):
            nbonds = len(self.nlist)
            self.nlist[nbonds+1]

    def test_index_writable(self):
        with self.assertRaises(TypeError):
            self.nlist[:, 0] = 0

        with self.assertRaises(TypeError):
            self.nlist[1, :] = 0

        with self.assertRaises(TypeError):
            self.nlist[:] = 0

        with self.assertRaises(TypeError):
            self.nlist[0, 0] = 0


if __name__ == '__main__':
    unittest.main()
