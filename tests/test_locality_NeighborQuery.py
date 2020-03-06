import numpy as np
import numpy.testing as npt
import freud
from collections import Counter
import itertools
import unittest

import util

"""
Define helper functions for getting the neighbors of a point. Note that
querying doesn't guarantee k results per ref point, but rather per point. As a
result, we need to be careful with the usage of these functions, and swap their
usage when we hit freud 2.0 (where we will reverse the ordering).
"""


def get_ref_point_neighbors(nl, i):
    return {x[0] for x in nl if x[1] == i}


def get_point_neighbors(nl, i):
    return {x[1] for x in nl if x[0] == i}


def nlist_equal(nlist1, nlist2):
    return set((i, j) for i, j in nlist1) == set((i, j) for i, j in nlist2)


class NeighborQueryTest(object):
    @classmethod
    def build_query_object(cls, box, ref_points, r_max=None):
        raise RuntimeError(
            "The build_query_object function must be defined for every "
            "subclass of NeighborQuery in a separate subclass of "
            "unittest.TestCase")

    def test_query_create(self):
        L = 10  # Box Dimensions
        r_max = 2.01  # Cutoff radius
        box = freud.box.Box.cube(L)

        # It's not allowed to have an empty NeighborQuery
        for empty_points in (
            [],
            [[]],
            np.zeros(0, dtype=np.float32),
            np.zeros(shape=(0, 3), dtype=np.float32)
        ):
            with self.assertRaises(ValueError):
                self.build_query_object(box, empty_points, r_max)

        # It's not allowed to have one point as a 1D array
        with self.assertRaises(ValueError):
            points = np.zeros(shape=(3), dtype=np.float32)
            self.build_query_object(box, points, r_max)

        # Create a NeighborQuery with one point
        points = np.zeros(shape=(1, 3), dtype=np.float32)
        self.build_query_object(box, points, r_max)

        # Create a NeighborQuery with ten points
        points = np.zeros(shape=(10, 3), dtype=np.float32)
        self.build_query_object(box, points, r_max)

    def test_query_ball(self):
        L = 10  # Box Dimensions
        r_max = 2.01  # Cutoff radius
        N = 4  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        nq = self.build_query_object(box, points, r_max)

        # particle 0 has 3 bonds
        npt.assert_equal(len(list(nq.query(
            points[[0]], dict(mode='ball', r_max=r_max)))), 3)
        # particle 1 has 4 bonds
        npt.assert_equal(len(list(nq.query(
            points[[1]], dict(mode='ball', r_max=r_max)))), 4)
        # particle 2 has 3 bonds
        npt.assert_equal(len(list(nq.query(
            points[[2]], dict(mode='ball', r_max=r_max)))), 3)
        # particle 3 has 4 bonds
        npt.assert_equal(len(list(nq.query(
            points[[3]], dict(mode='ball', r_max=r_max)))), 4)

        # Check NeighborList length without self-exclusions.
        nlist = nq.query(
            points, dict(mode='ball', r_max=r_max,
                         exclude_ii=True)).toNeighborList()
        nlist_neighbors = sorted(list(zip(nlist.query_point_indices,
                                          nlist.point_indices)))
        # When excluding, everything has one less neighbor.
        npt.assert_equal(len(nlist_neighbors), 10)

        # now move particle 0 out of range...
        points[0] = 5

        nq = self.build_query_object(box, points, r_max)

        # particle 0 has 1 bonds
        npt.assert_equal(len(list(nq.query(
            points[[0]], dict(mode='ball', r_max=r_max)))), 1)
        # particle 1 has 3 bonds
        npt.assert_equal(len(list(nq.query(
            points[[1]], dict(mode='ball', r_max=r_max)))), 3)
        # particle 2 has 3 bonds
        npt.assert_equal(len(list(nq.query(
            points[[2]], dict(mode='ball', r_max=r_max)))), 3)
        # particle 3 has 3 bonds
        npt.assert_equal(len(list(nq.query(
            points[[3]], dict(mode='ball', r_max=r_max)))), 3)

    def test_query_mode_inference(self):
        """Check that the generic querying method correctly infers modes."""
        L = 10  # Box Dimensions
        r_max = 2.01  # Cutoff radius
        num_neighbors = 3
        N = 10  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box
        points = np.random.rand(N, 3).astype(np.float32)
        nq = self.build_query_object(box, points, r_max)

        # Test ball query.
        result1 = list(nq.query(points, dict(mode='ball', r_max=r_max)))
        result2 = list(nq.query(points, dict(r_max=r_max)))
        npt.assert_equal(result1, result2)

        # Test ball query with exclusion.
        result1 = list(nq.query(
            points, dict(mode='ball', r_max=r_max, exclude_ii=True)))
        result2 = list(nq.query(
            points, dict(r_max=r_max, exclude_ii=True)))
        npt.assert_equal(result1, result2)

        # Test number of neighbors.
        result1 = list(nq.query(
            points, dict(mode='nearest', num_neighbors=num_neighbors)))
        result2 = list(nq.query(points, dict(num_neighbors=num_neighbors)))
        npt.assert_equal(result1, result2)

        # Test number of neighbors with exclusion.
        result1 = list(nq.query(
            points, dict(mode='nearest', num_neighbors=num_neighbors,
                         exclude_ii=True)))
        result2 = list(nq.query(
            points, dict(num_neighbors=num_neighbors, exclude_ii=True)))
        npt.assert_equal(result1, result2)

    def test_query_invalid(self):
        """Check that mode inference fails for invalid combinations."""
        L = 10  # Box Dimensions
        r_max = 2.01  # Cutoff radius
        num_neighbors = 3
        N = 10  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box
        points = np.random.rand(N, 3).astype(np.float32)
        nq = self.build_query_object(box, points, r_max)

        # Test failure cases.
        with self.assertRaises(RuntimeError):
            list(nq.query(points, dict(mode='ball',
                                       num_neighbors=num_neighbors,
                                       r_max=r_max)))

    def test_r_min(self):
        """Test filtering with r_min."""
        L = 10  # Box Dimensions
        N = 4  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        nq = self.build_query_object(box, points, L/10)

        # Test with ball query.
        result = list(nq.query(points, dict(mode='ball', r_max=2.9, r_min=1.1,
                                            exclude_ii=True)))
        npt.assert_equal(get_point_neighbors(result, 0), {3})
        npt.assert_equal(get_point_neighbors(result, 1), {2})
        npt.assert_equal(get_point_neighbors(result, 2), {1})
        npt.assert_equal(get_point_neighbors(result, 3), {0})

        # Test with nearest neighbor query.
        result = list(nq.query(points, dict(mode='nearest', num_neighbors=3,
                                            r_min=1.1, exclude_ii=True)))
        npt.assert_equal(get_point_neighbors(result, 0), {2, 3})
        npt.assert_equal(get_point_neighbors(result, 1), {2})
        npt.assert_equal(get_point_neighbors(result, 2), {0, 1})
        npt.assert_equal(get_point_neighbors(result, 3), {0})

    def test_query_nearest(self):
        L = 10  # Box Dimensions
        N = 4  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        nq = self.build_query_object(box, points, L/10)

        result = list(nq.query(points, dict(mode='nearest', num_neighbors=3)))
        npt.assert_equal(get_point_neighbors(result, 0), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 1), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 2), {1, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 3), {1, 2, 3})

        # All other points are neighbors when self-neighbors are excluded.
        result = list(nq.query(points, dict(mode='nearest', num_neighbors=3,
                               exclude_ii=True)))
        npt.assert_equal(get_point_neighbors(result, 0), {1, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 1), {0, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 2), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 3), {0, 1, 2})

        # Test overflow case. Need to sort because the nlist output of
        # query is sorted by ref_point by construction.
        npt.assert_equal(list(nq.query(points, dict(mode='nearest',
                                                    num_neighbors=5,
                                                    exclude_ii=True))), result)

        # Test setting r_max as a filter
        result = list(nq.query(points, dict(mode='nearest', num_neighbors=3,
                                            r_max=1.9, exclude_ii=True)))
        npt.assert_equal(get_point_neighbors(result, 0), {1})
        npt.assert_equal(get_point_neighbors(result, 1), {0, 3})
        npt.assert_equal(get_point_neighbors(result, 2), {3})
        npt.assert_equal(get_point_neighbors(result, 3), {1, 2})

    def test_query_ball_to_nlist(self):
        """Test that generated NeighborLists are identical to the results of
        querying"""
        L = 10  # Box Dimensions
        N = 400  # number of particles

        box, ref_points = freud.data.make_random_system(L, N, seed=0)
        points = np.random.rand(N, 3) * L

        nq = self.build_query_object(box, ref_points, L/10)

        result_list = list(nq.query(points, dict(mode='ball', r_max=2)))
        result_list = [(b[0], b[1]) for b in result_list]
        nlist = nq.query(points, dict(mode='ball', r_max=2)).toNeighborList()
        list_nlist = list(zip(nlist.query_point_indices, nlist.point_indices))

        npt.assert_equal(set(result_list), set(list_nlist))

    def test_query_to_nlist(self):
        """Test that generated NeighborLists are identical to the results of
        querying"""
        L = 10  # Box Dimensions
        N = 400  # number of particles

        box, ref_points = freud.data.make_random_system(L, N, seed=0)
        _, points = freud.data.make_random_system(L, N, seed=1)

        nq = self.build_query_object(box, ref_points, L/10)

        result_list = list(nq.query(points, dict(mode='ball', r_max=2)))
        result_list = [(b[0], b[1]) for b in result_list]
        nlist = nq.query(points, dict(mode='ball', r_max=2)).toNeighborList()
        list_nlist = list(zip(nlist.query_point_indices, nlist.point_indices))

        npt.assert_equal(set(result_list), set(list_nlist))

    def test_reciprocal(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for one set of points"""
        L, r_max, N = (10, 2.01, 1024)

        box, points = freud.data.make_random_system(L, N, seed=0)
        nq = self.build_query_object(box, points, r_max)
        result = list(nq.query(points, dict(mode='ball', r_max=r_max)))

        ij = {(x[0], x[1]) for x in result}
        ji = set((j, i) for (i, j) in ij)

        self.assertEqual(ij, ji)

    def test_reciprocal_twoset(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for two sets of
        different points
        """
        L, r_max, N = (10, 2.01, 1024)

        box, points = freud.data.make_random_system(L, N, seed=0)
        _, points2 = freud.data.make_random_system(L, N//6, seed=1)
        nq = self.build_query_object(box, points, r_max)
        nq2 = self.build_query_object(box, points2, r_max)

        result = list(nq.query(points2, dict(mode='ball', r_max=r_max)))
        result2 = list(nq2.query(points, dict(mode='ball', r_max=r_max)))

        ij = {(x[0], x[1]) for x in result}
        ij2 = {(x[1], x[0]) for x in result2}

        self.assertEqual(ij, ij2)

    def test_exclude_ii(self):
        L, r_max, N = (10, 2.01, 1024)

        box, points = freud.data.make_random_system(L, N)
        points2 = points[:N//6]
        nq = self.build_query_object(box, points, r_max)
        result = list(nq.query(points2, dict(mode='ball', r_max=r_max)))

        ij1 = {(x[0], x[1]) for x in result}

        result2 = list(nq.query(points2, dict(mode='ball', r_max=r_max,
                                exclude_ii=True)))

        ij2 = {(x[0], x[1]) for x in result2}

        self.assertTrue(all((i, i) not in ij2 for i in range(N)))

        ij2.update((i, i) for i in range(points2.shape[0]))

        self.assertEqual(ij1, ij2)

    def test_exhaustive_search(self):
        L, r_max, N = (10, 1.999, 32)

        box = freud.box.Box.cube(L)
        seed = 0

        for i in range(10):
            _, points = freud.data.make_random_system(L, N, seed=seed+i)
            all_vectors = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            all_vectors = box.wrap(
                all_vectors.reshape((-1, 3))).reshape(all_vectors.shape)
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < r_max**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            nq = self.build_query_object(box, points, r_max)
            result = list(nq.query(points, dict(mode='ball', r_max=r_max,
                                                exclude_ii=True)))
            ijs = {(x[1], x[0]) for x in result}
            counts = Counter([x[1] for x in result])
            counts_list = [counts[j] for j in range(N)]

            try:
                self.assertEqual(exhaustive_ijs, ijs)
            except AssertionError:
                print('Failed neighbors, random seed: {} (i={})'.format(
                    seed, i))
                raise

            try:
                self.assertEqual(exhaustive_counts_list, counts_list)
            except AssertionError:
                print('Failed neighbor counts, random seed: {} (i={})'.format(
                    seed, i))
                raise

    def test_exhaustive_search_asymmetric(self):
        L, r_max, N = (10, 1.999, 32)

        box = freud.box.Box.cube(L)
        seed = 0

        for i in range(10):
            np.random.seed(seed + i)
            points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
            points2 = np.random.uniform(
                -L/2, L/2, (N//2, 3)).astype(np.float32)
            all_vectors = points[:, np.newaxis, :] - points2[np.newaxis, :, :]
            all_vectors = box.wrap(
                all_vectors.reshape((-1, 3))).reshape(all_vectors.shape)
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < r_max**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            nq = self.build_query_object(box, points2, r_max)
            result = list(nq.query(points, dict(mode='ball', r_max=r_max)))
            ijs = {(x[0], x[1]) for x in result}
            counts = Counter([x[0] for x in result])
            counts_list = [counts[j] for j in range(N)]

            try:
                self.assertEqual(exhaustive_ijs, ijs)
            except AssertionError:
                print('Failed neighbors, random seed: {} (i={})'.format(
                    seed, i))
                raise

            try:
                self.assertEqual(exhaustive_counts_list, counts_list)
            except AssertionError:
                print('Failed neighbor counts, random seed: {} (i={})'.format(
                    seed, i))
                raise

    def test_attributes(self):
        """Ensure that mixing old and new APIs throws an error"""
        L = 10

        box = freud.box.Box.cube(L)

        points = np.zeros(shape=(4, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]

        nq = self.build_query_object(box, points, L/10)
        npt.assert_array_equal(nq.points, points)
        self.assertEqual(nq.box, box)

    def test_no_bonds(self):
        L = 10
        box = freud.box.Box.cube(L)

        # make a sc lattice
        lattice_xs = np.linspace(-float(L)/2, float(L)/2, L, endpoint=False)
        positions = list(itertools.product(lattice_xs, lattice_xs, lattice_xs))
        positions = np.array(positions, dtype=np.float32)

        # r_max is slightly smaller than the distance for any particle
        nq = self.build_query_object(box, positions, L/10)
        result = list(nq.query(positions, dict(mode='ball', r_max=0.99,
                                               exclude_ii=True)))

        self.assertEqual(len(result), 0)

    def test_corner_2d(self):
        """Check an extreme case where finding enough nearest neighbors
        requires going beyond normally allowed cutoff."""
        L = 2.1
        box = freud.box.Box.square(L)

        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        nq = self.build_query_object(box, positions, L/10)
        result = list(nq.query(positions[[0]], dict(mode='nearest',
                                                    num_neighbors=3)))
        self.assertEqual(get_point_neighbors(result, 0), {0, 1, 2})

        # Check the effect of points != ref_points
        positions[:, :2] -= 0.1
        result = list(nq.query(positions[[0]], dict(mode='nearest',
                                                    num_neighbors=3)))
        self.assertEqual(get_point_neighbors(result, 0), {0, 1, 2})

        # Since the initial position set aligns exactly with cell boundaries,
        # make sure that the correctness is not affected by that artifact.
        nq = self.build_query_object(box, positions, L/10)
        result = list(nq.query(positions[[0]], dict(mode='nearest',
                                                    num_neighbors=3)))
        self.assertEqual(get_point_neighbors(result, 0), {0, 1, 2})

    def test_random_system_query(self):
        np.random.seed(0)
        L = 10
        box = freud.box.Box.cube(L)

        # Generate random points
        for N in [10, 100, 500]:
            positions = box.wrap(L/2 * np.random.rand(N, 3))
            ks = [1, 5]
            if N > 10:
                ks.extend([10, 50])
            for k in ks:
                nq = self.build_query_object(box, positions, L/10)

                nlist = nq.query(
                    positions, dict(num_neighbors=k,
                                    exclude_ii=True)).toNeighborList()
                assert len(nlist) == k * N,\
                    ("Wrong nlist length for N = {},"
                     "num_neighbors = {}, length = {}").format(
                        N, k, len(nlist))
                nlist_array = nlist[:]
                for i in range(N):
                    assert not ([i, i] == nlist_array).all(axis=1).any()

                nlist = nq.query(
                    positions, dict(num_neighbors=k,
                                    exclude_ii=False)).toNeighborList()
                assert len(nlist) == k * N,\
                    ("Wrong nlist length for N = {}, "
                     "num_neighbors = {}, length = {}").format(
                        N, k, len(nlist))
                nlist_array = nlist[:]
                for i in range(N):
                    assert ([i, i] == nlist_array).all(axis=1).any()

    def test_duplicate_cell_shells(self):
        box = freud.box.Box.square(5)
        points = [[-1.5, 0, 0]]
        ref_points = [[0.9, 0, 0]]
        r_max = 2.45
        cell_width = 1
        nq = self.build_query_object(box, ref_points, cell_width)
        q = nq.query(points, dict(r_max=r_max))
        self.assertEqual(len(list(q)), 1)
        q = nq.query(points, dict(num_neighbors=1000))
        self.assertEqual(len(list(q)), 1)

    def test_duplicate_cell_shells2(self):
        positions = [[1.5132198, 6.67087, 3.1856632],
                     [1.3913784, -2.3667011, 4.5227165],
                     [-3.6133137, 9.043476, 0.8957424]]
        box = freud.box.Box.cube(21)
        r_max = 10
        nq = self.build_query_object(box, positions, r_max)
        q = nq.query(positions[0], dict(r_max=r_max))
        self.assertEqual(len(list(q)), 3)
        q = nq.query(positions[0], dict(num_neighbors=1000))
        self.assertEqual(len(list(q)), 3)

    def test_2d_box_3d_points(self):
        """Test that points with z != 0 fail if the box is 2D."""
        L = 10  # Box Dimensions
        r_max = 2.01  # Cutoff radius

        box = freud.box.Box.square(L)  # Initialize Box
        points = np.array([[0, 0, 0], [0, 1, 1]])
        with self.assertRaises(ValueError):
            self.build_query_object(box, points, r_max)

    def test_plot_3d(self):
        box = freud.box.Box(3, 4, 5, 1, 0.5, 0.1)
        points = np.array([[0, 0, 0], [1, 1, 0]])
        r_max = 1
        nq = self.build_query_object(box, points, r_max)
        nq.plot()

    def test_plot_2d(self):
        box = freud.box.Box(3, 3, 0, 1, 0, 0, is2D=True)
        points = np.array([[0, 0, 0], [1, 1, 0]])
        r_max = 1
        nq = self.build_query_object(box, points, r_max)
        nq.plot()


class TestNeighborQueryAABB(NeighborQueryTest, unittest.TestCase):
    @classmethod
    def build_query_object(cls, box, ref_points, r_max=None):
        return freud.locality.AABBQuery(box, ref_points)

    def test_throws(self):
        """Test that specifying too large an r_max value throws an error"""
        L = 5

        box = freud.box.Box.square(L)
        points = [[0, 0, 0], [1, 1, 0], [1, -1, 0]]
        aq = freud.locality.AABBQuery(box, points)
        with self.assertRaises(RuntimeError):
            list(aq.query(points, dict(r_max=L)))

    def test_chaining(self):
        N = 500
        L = 10
        r_max = 1
        box, points = freud.data.make_random_system(L, N)
        nlist1 = freud.locality.AABBQuery(box, points).query(
            points, dict(r_max=r_max, exclude_ii=True)).toNeighborList()
        abq = freud.locality.AABBQuery(box, points)
        nlist2 = abq.query(points, dict(r_max=r_max,
                                        exclude_ii=True)).toNeighborList()
        self.assertTrue(nlist_equal(nlist1, nlist2))

    def test_r_guess_scale(self):
        """Ensure that r_guess and scale have no effect on query results."""
        np.random.seed(0)
        L = 10
        box = freud.box.Box.cube(L)

        N = 100
        positions = box.wrap(L/2 * np.random.rand(N, 3))
        nq = self.build_query_object(box, positions, L/10)

        k = 10
        r_guess_vals = [L/20, L/10, L/5]
        scales = [1.01, 1.1, 1.3]

        original_nlist = None
        for r_guess in r_guess_vals:
            for scale in scales:
                nlist = nq.query(
                    positions, dict(num_neighbors=k, exclude_ii=True,
                                    r_guess=r_guess,
                                    scale=scale)).toNeighborList()
                if original_nlist is not None:
                    self.assertTrue(nlist_equal(nlist, original_nlist))
                else:
                    original_nlist = nlist


class TestNeighborQueryLinkCell(NeighborQueryTest, unittest.TestCase):
    @classmethod
    def build_query_object(cls, box, ref_points, r_max=None):
        if r_max is None:
            raise ValueError("Building LinkCells requires passing an r_max.")
        return freud.locality.LinkCell(box, ref_points, r_max)

    def test_chaining(self):
        N = 500
        L = 10
        r_max = 1
        box, points = freud.data.make_random_system(L, N)
        nlist1 = freud.locality.LinkCell(box, points, 1.0).query(
            points, dict(r_max=r_max, exclude_ii=True)).toNeighborList()
        lc = freud.locality.LinkCell(box, points, 1.0)
        nlist2 = lc.query(points, dict(r_max=r_max,
                                       exclude_ii=True)).toNeighborList()
        self.assertTrue(nlist_equal(nlist1, nlist2))

    def test_default_cell_width(self):
        """Check that using a default cell width works."""
        N = 500
        L = 10
        r_max = 1
        box, points = freud.data.make_random_system(L, N)
        nlist1 = freud.locality.LinkCell(box, points).query(
            points, dict(r_max=r_max, exclude_ii=True)).toNeighborList()
        lc = freud.locality.LinkCell(box, points, 1.0)
        nlist2 = lc.query(points, dict(r_max=r_max,
                                       exclude_ii=True)).toNeighborList()
        self.assertTrue(nlist_equal(nlist1, nlist2))


class TestMultipleMethods(unittest.TestCase):
    """Check that different methods of making a NeighborList give the same
    result."""

    def test_alternating_points(self):
        lattice_size = 10
        # big box to ignore periodicity
        box = freud.box.Box.square(lattice_size*5)
        query_points, points = util.make_alternating_lattice(lattice_size)
        r_max = 1.6
        num_neighbors = 12

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "nearest", r_max, num_neighbors, False)
        nlist = test_set[-1][1]
        for nq, neighbors in test_set:
            if not isinstance(nq, freud.locality.NeighborQuery):
                continue
            check_nlist = nq.query(
                query_points, neighbors).toNeighborList()
            self.assertTrue(nlist_equal(nlist, check_nlist))


if __name__ == '__main__':
    unittest.main()
