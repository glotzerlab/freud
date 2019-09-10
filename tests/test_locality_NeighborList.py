import numpy as np
import numpy.testing as npt
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

        # if query_point_index isn't writable, point_index shouldn't be
        with self.assertRaises(ValueError):
            self.cl.nlist.point_index[:] = 0

        # the weights array may be useful to write to, though
        # TODO: weights aren't writable since changing to ManagedArray
        with self.assertRaises(ValueError):
            self.cl.nlist.weights[18] = 3

    def test_filter_r_max(self):
        self.setup_nl()
        np.random.seed(0)
        points2 = self.points[:self.N//2]
        filter_max_distance = 2.5

        self.cl.compute(self.fbox, self.points, points2)
        old_size = len(self.cl.nlist)

        # Make sure weights are 1
        npt.assert_equal(self.cl.nlist.weights, 1)

        # Compute neighbor pairs that we expect to be kept after the filter
        kept_neighbors = self.cl.nlist[
            self.cl.nlist.distances < filter_max_distance]

        self.cl.nlist.filter_r(filter_max_distance)
        new_size = len(self.cl.nlist)

        self.assertGreater(old_size, 0)
        self.assertGreater(new_size, 0)
        self.assertLessEqual(new_size, old_size)

        # Make sure weights are still 1 after resize
        npt.assert_equal(self.cl.nlist.weights, 1)

        # Make sure distances are filtered
        npt.assert_array_less(self.cl.nlist.distances,
                              np.full(new_size, filter_max_distance))

        npt.assert_equal(kept_neighbors, self.cl.nlist)

    def test_filter_r_max_min(self):
        self.setup_nl()
        np.random.seed(0)
        points2 = self.points[:self.N//2]
        filter_min_distance = 1.5
        filter_max_distance = 2.5

        self.cl.compute(self.fbox, self.points, points2)
        old_size = len(self.cl.nlist)

        # Make sure weights are 1
        npt.assert_equal(self.cl.nlist.weights, 1)

        # Compute neighbor pairs that we expect to be kept after the filter
        kept_neighbors = self.cl.nlist[np.logical_and(
            self.cl.nlist.distances < filter_max_distance,
            self.cl.nlist.distances >= filter_min_distance)]

        self.cl.nlist.filter_r(filter_max_distance, filter_min_distance)
        new_size = len(self.cl.nlist)

        self.assertGreater(old_size, 0)
        self.assertGreater(new_size, 0)
        self.assertLessEqual(new_size, old_size)

        # Make sure weights are still 1 after resize
        npt.assert_equal(self.cl.nlist.weights, 1)

        # Make sure distances are filtered
        npt.assert_array_less(self.cl.nlist.distances,
                              np.full(new_size, filter_max_distance))
        npt.assert_array_less(np.full(new_size, filter_min_distance),
                              self.cl.nlist.distances)

        npt.assert_equal(kept_neighbors, self.cl.nlist)

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

        # copy of existing nlist by arrays
        weights = np.random.rand(len(query_point_index))
        nlist = locality.NeighborList.from_arrays(
            4, 4, query_point_index, point_index, distances, weights)
        nlist2 = locality.NeighborList.from_arrays(
            4, 4, nlist.query_point_index, nlist.point_index,
            nlist.distances, nlist.weights)
        npt.assert_equal(nlist.query_point_index, nlist2.query_point_index)
        npt.assert_equal(nlist.point_index, nlist2.point_index)
        npt.assert_equal(nlist.distances, nlist2.distances)
        npt.assert_equal(nlist.weights, nlist2.weights)
        npt.assert_equal(nlist.neighbor_counts, nlist2.neighbor_counts)
        npt.assert_equal(nlist.segments, nlist2.segments)

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
        self.assertEqual(len(self.cl.nlist),
                         len(self.cl.nlist.query_point_index))
        self.assertEqual(len(self.cl.nlist),
                         len(self.cl.nlist.point_index))

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

    def test_copy(self):
        self.setup_nl()
        self.cl.compute(self.fbox, self.points, self.points)
        nlist = self.cl.nlist
        nlist2 = nlist.copy()
        npt.assert_equal(nlist[:], nlist2[:])
        npt.assert_equal(nlist.distances, nlist2.distances)
        npt.assert_equal(nlist.weights, nlist2.weights)
        npt.assert_equal(nlist.segments, nlist2.segments)
        npt.assert_equal(nlist.neighbor_counts, nlist2.neighbor_counts)


if __name__ == '__main__':
    unittest.main()
