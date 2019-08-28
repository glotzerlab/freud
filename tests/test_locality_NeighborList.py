import numpy as np
from freud import locality
import unittest
from util import make_box_and_random_points


class TestNeighborList(unittest.TestCase):

    def setup_nl(self,
                 L=10,
                 r_max=3,
                 N=40,
                 num_neighbors=6):
        # Define values
        self.L = L
        self.r_max = r_max
        self.N = N
        self.num_neighbors = num_neighbors

        # Initialize Box and cell list
        self.cl = locality.NearestNeighbors(self.r_max, self.num_neighbors)
        self.fbox, self.points = make_box_and_random_points(L, N)

    def test_writable(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)

        # query_point_index shouldn't be writable in general or users may break
        # the ordered property of the neighbor list
        with self.assertRaises(ValueError):
            self.cl.nlist.query_point_index[:] = 0

        # if query_point_index isn't writable, point_index probably also shouldn't be
        with self.assertRaises(ValueError):
            self.cl.nlist.point_index[:] = 0

        # the weights array may be useful to write to, though
        self.cl.nlist.weights[18] = 3
        self.assertEqual(self.cl.nlist.weights[18], 3)

    def test_filter_r(self):
        self.setup_nl()
        np.random.seed(0)
        points2 = self.points[:self.N//2]
        self.cl.compute(self.fbox, self.points, self.points)

        self.cl.compute(self.fbox, self.points, points2)
        self.cl.nlist.filter_r(2.5)

    def test_filter(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)
        old_size = len(self.cl.nlist)
        filt = (self.cl.nlist.point_index.astype(np.int32) -
                self.cl.nlist.query_point_index.astype(np.int32)) % 2 == 0
        self.cl.nlist.filter(filt)
        self.assertLessEqual(len(self.cl.nlist), old_size)

        # should be able to further filter
        self.cl.nlist.filter_r(2.5)

    def test_find_first_index(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)
        nlist = self.cl.nlist
        for (idx, i) in enumerate(nlist.query_point_index):
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
        query_point_index = [0, 0, 1, 2, 3]
        point_index = [1, 2, 3, 0, 0]
        distances = np.ones(len(query_point_index))

        # implicit weights
        nlist = locality.NeighborList.from_arrays(
            4, 4, query_point_index, point_index, distances)
        self.assertTrue(np.allclose(nlist.weights, 1))

        # explicit weights
        weights = np.ones(len(query_point_index))*4.
        nlist = locality.NeighborList.from_arrays(
            4, 4, query_point_index, point_index, distances, weights)
        self.assertTrue(np.allclose(nlist.weights, 4))

        # too few reference particles
        with self.assertRaises(RuntimeError):
            nlist = locality.NeighborList.from_arrays(
                3, 4, query_point_index, point_index, distances)

        # too few target particles
        with self.assertRaises(RuntimeError):
            nlist = locality.NeighborList.from_arrays(
                4, 3, query_point_index, point_index, distances)

        # query particles not sorted
        with self.assertRaises(RuntimeError):
            nlist = locality.NeighborList.from_arrays(
                4, 4, point_index, query_point_index, distances)

        # mismatched array sizes
        with self.assertRaises(ValueError):
            nlist = locality.NeighborList.from_arrays(
                4, 4, query_point_index[:-1], point_index, distances)
        with self.assertRaises(ValueError):
            nlist = locality.NeighborList.from_arrays(
                4, 4, query_point_index, point_index[:-1], distances)
        with self.assertRaises(ValueError):
            nlist = locality.NeighborList.from_arrays(
                4, 4, query_point_index, point_index, distances[:-1])
        with self.assertRaises(ValueError):
            weights = np.ones((len(query_point_index) - 1,))
            nlist = locality.NeighborList.from_arrays(
                4, 4, query_point_index, point_index, distances, weights)

    def test_indexing(self):
        self.setup_nl()
        np.random.seed(0)

        # Ensure that empty NeighborLists have the right shape
        self.assertEqual(self.cl.nlist[:].shape, (0, 2))

        # Make sure indexing the NeighborList is the same as indexing arrays
        self.cl.compute(self.fbox, self.points, self.points)
        for i, (idx_i, idx_j) in enumerate(self.cl.nlist):
            self.assertEqual(idx_i, self.cl.nlist.query_point_index[i])
            self.assertEqual(idx_j, self.cl.nlist.point_index[i])

        for i, j in self.cl.nlist:
            self.assertNotEqual(i, j)

    def test_nl_size(self):
        self.setup_nl()
        np.random.seed(0)
        self.cl.compute(self.fbox, self.points, self.points)
        self.assertEqual(len(self.cl.nlist), len(self.cl.nlist.query_point_index))
        self.assertEqual(len(self.cl.nlist), len(self.cl.nlist.point_index))

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
