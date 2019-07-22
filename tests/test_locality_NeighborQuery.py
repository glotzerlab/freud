import numpy as np
import numpy.testing as npt
import freud
from collections import Counter
import itertools
import unittest
from util import makeBoxAndRandomPoints

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


class TestNeighborQuery(object):
    @classmethod
    def build_query_object(cls, box, ref_points, rcut=None):
        raise RuntimeError(
            "The build_query_object function must be defined for every "
            "subclass of NeighborQuery in a separate subclass of "
            "unittest.TestCase")

    def test_query_ball(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        nq = self.build_query_object(box, points, rcut)

        # particle 0 has 3 bonds
        npt.assert_equal(len(list(nq.queryBall(points[[0]], rcut))), 3)
        # particle 1 has 4 bonds
        npt.assert_equal(len(list(nq.queryBall(points[[1]], rcut))), 4)
        # particle 2 has 3 bonds
        npt.assert_equal(len(list(nq.queryBall(points[[2]], rcut))), 3)
        # particle 3 has 4 bonds
        npt.assert_equal(len(list(nq.queryBall(points[[3]], rcut))), 4)

        # Check NeighborList length without self-exclusions.
        npt.assert_equal(
            len(nq.queryBall(points, rcut, exclude_ii=False).toNList()), 14)

        # When excluding, check that everything has one less neighbor and that
        # the toNList method correctly excludes neighbors.
        list_of_neighbors = sorted(list(
            nq.queryBall(points, rcut, exclude_ii=True)))
        # Remove distances, which are not used here.
        list_of_neighbors = [[p[0], p[1]] for p in list_of_neighbors]
        nlist = nq.queryBall(points, rcut, exclude_ii=True).toNList()
        nlist_neighbors = sorted(list(zip(nlist.index_i, nlist.index_j)))

        # Number of neighbors is one less per-particle than before.
        npt.assert_equal(len(list_of_neighbors), 10)
        npt.assert_equal(len(nlist_neighbors), 10)

        # Actual set of neighbors should be identical from both methods.
        npt.assert_array_equal(list_of_neighbors, nlist_neighbors)

        # now move particle 0 out of range...
        points[0] = 5

        # particle 0 has no bonds now
        npt.assert_equal(len(list(nq.queryBall(points[[0]], rcut))), 0)

    def test_query_ball_generic(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        nq = self.build_query_object(box, points, rcut)

        nlist = nq._queryGeneric(points, dict(mode='ball', rmax=rcut))
        nlist_neighbors = sorted(list(zip(nlist.index_i, nlist.index_j)))
        # particle 0 has 2 bonds
        npt.assert_equal(sum(nlist.index_i == 0), 3)
        # particle 1 has 3 bonds
        npt.assert_equal(sum(nlist.index_i == 1), 4)
        # particle 2 has 2 bonds
        npt.assert_equal(sum(nlist.index_i == 2), 3)
        # particle 3 has 3 bonds
        npt.assert_equal(sum(nlist.index_i == 3), 4)

        # Check NeighborList length without self-exclusions.
        nlist = nq._queryGeneric(
            points, dict(mode='ball', rmax=rcut, exclude_ii=True))
        nlist_neighbors = sorted(list(zip(nlist.index_i, nlist.index_j)))
        # When excluding, everything has one less neighbor.
        npt.assert_equal(len(nlist_neighbors), 10)

        # now move particle 0 out of range...
        points[0] = 5

        nq = freud.locality.LinkCell(box, rcut, points)
        nlist = nq._queryGeneric(points, dict(mode='ball', rmax=rcut))
        nlist_neighbors = sorted(list(zip(nlist.index_i, nlist.index_j)))
        # particle 0 has 0 bonds
        npt.assert_equal(sum(nlist.index_i == 0), 1)
        # particle 1 has 2 bonds
        npt.assert_equal(sum(nlist.index_i == 1), 3)
        # particle 2 has 2 bonds
        npt.assert_equal(sum(nlist.index_i == 2), 3)
        # particle 3 has 2 bonds
        npt.assert_equal(sum(nlist.index_i == 3), 3)

    def test_query(self):
        L = 10  # Box Dimensions
        N = 4  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        nq = self.build_query_object(box, points, L/10)

        result = list(nq.query(points, 3))
        npt.assert_equal(get_point_neighbors(result, 0), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 1), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 2), {1, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 3), {1, 2, 3})

        # All other points are neighbors when self-neighbors are excluded.
        result = list(nq.query(points, 3, exclude_ii=True))
        npt.assert_equal(get_point_neighbors(result, 0), {1, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 1), {0, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 2), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 3), {0, 1, 2})

        # Make sure the result is not changed by conversion to a NeighborList.
        nlist = nq.query(points, 3, exclude_ii=True).toNList()
        nlist_result = sorted(list(zip(nlist.index_i, nlist.index_j)))
        original_result = sorted([(x[0], x[1]) for x in result])
        npt.assert_equal(nlist_result, original_result)

        # Test overflow case
        npt.assert_equal(list(nq.query(points, 5, exclude_ii=True)), result)

    def test_query_generic(self):
        L = 10  # Box Dimensions
        N = 4  # number of particles

        box = freud.box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        nq = self.build_query_object(box, points, L/10)

        nlist = nq._queryGeneric(points, dict(mode='nearest', nn=3))
        result = list(zip(nlist.index_i, nlist.index_j))
        npt.assert_equal(get_point_neighbors(result, 0), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 1), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 2), {1, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 3), {1, 2, 3})

        # All other points are neighbors when self-neighbors are excluded.
        nlist = nq._queryGeneric(
            points, dict(mode='nearest', nn=3, exclude_ii=True))
        result = list(zip(nlist.index_i, nlist.index_j))
        npt.assert_equal(get_point_neighbors(result, 0), {1, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 1), {0, 2, 3})
        npt.assert_equal(get_point_neighbors(result, 2), {0, 1, 3})
        npt.assert_equal(get_point_neighbors(result, 3), {0, 1, 2})

        # Test overflow case. Need to sort because the nlist output of
        # _queryGeneric is sorted by ref_point by construction.
        all_results = sorted(
            [(x[0], x[1]) for x in nq.query(points, 5, exclude_ii=True)])
        npt.assert_equal(all_results, result)

    def test_query_ball_to_nlist(self):
        """Test that generated NeighborLists are identical to the results of
        querying"""
        L = 10  # Box Dimensions
        N = 400  # number of particles

        box, ref_points = makeBoxAndRandomPoints(L, N, seed=0)
        points = np.random.rand(N, 3) * L

        nq = self.build_query_object(box, ref_points, L/10)

        result_list = list(nq.queryBall(points, 2))
        result_list = [(b[0], b[1]) for b in result_list]
        nlist = nq.queryBall(points, 2).toNList()
        list_nlist = list(zip(nlist.index_i, nlist.index_j))

        npt.assert_equal(set(result_list), set(list_nlist))

    def test_query_to_nlist(self):
        """Test that generated NeighborLists are identical to the results of
        querying"""
        L = 10  # Box Dimensions
        N = 400  # number of particles

        box, ref_points = makeBoxAndRandomPoints(L, N, seed=0)
        _, points = makeBoxAndRandomPoints(L, N, seed=1)

        nq = self.build_query_object(box, ref_points, L/10)

        result_list = list(nq.query(points, 2))
        result_list = [(b[0], b[1]) for b in result_list]
        nlist = nq.query(points, 2).toNList()
        list_nlist = list(zip(nlist.index_i, nlist.index_j))

        npt.assert_equal(set(result_list), set(list_nlist))

    def test_reciprocal(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for one set of points"""
        L, rcut, N = (10, 2.01, 1024)

        box, points = makeBoxAndRandomPoints(L, N, seed=0)
        nq = self.build_query_object(box, points, rcut)
        result = list(nq.queryBall(points, rcut))

        ij = {(x[0], x[1]) for x in result}
        ji = set((j, i) for (i, j) in ij)

        self.assertEqual(ij, ji)

    def test_reciprocal_twoset(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for two sets of
        different points
        """
        L, rcut, N = (10, 2.01, 1024)

        box, points = makeBoxAndRandomPoints(L, N, seed=0)
        _, points2 = makeBoxAndRandomPoints(L, N//6, seed=1)
        nq = self.build_query_object(box, points, rcut)
        nq2 = self.build_query_object(box, points2, rcut)

        result = list(nq.queryBall(points2, rcut))
        result2 = list(nq2.queryBall(points, rcut))

        ij = {(x[0], x[1]) for x in result}
        ij2 = {(x[1], x[0]) for x in result2}

        self.assertEqual(ij, ij2)

    def test_exclude_ii(self):
        L, rcut, N = (10, 2.01, 1024)

        box, points = makeBoxAndRandomPoints(L, N)
        points2 = points[:N//6]
        nq = self.build_query_object(box, points, rcut)
        result = list(nq.queryBall(points2, rcut))

        ij1 = {(x[0], x[1]) for x in result}

        result2 = list(nq.queryBall(points2, rcut, exclude_ii=True))

        ij2 = {(x[0], x[1]) for x in result2}

        self.assertTrue(all((i, i) not in ij2 for i in range(N)))

        ij2.update((i, i) for i in range(points2.shape[0]))

        self.assertEqual(ij1, ij2)

    def test_exhaustive_search(self):
        L, rcut, N = (10, 1.999, 32)

        box = freud.box.Box.cube(L)
        seed = 0

        for i in range(10):
            _, points = makeBoxAndRandomPoints(L, N, seed=seed+i)
            all_vectors = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            box.wrap(all_vectors.reshape((-1, 3)))
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < rcut**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            nq = self.build_query_object(box, points, rcut)
            result = list(nq.queryBall(points, rcut, exclude_ii=True))
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
        L, rcut, N = (10, 1.999, 32)

        box = freud.box.Box.cube(L)
        seed = 0

        for i in range(10):
            np.random.seed(seed + i)
            points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
            points2 = np.random.uniform(
                -L/2, L/2, (N//2, 3)).astype(np.float32)
            all_vectors = points[:, np.newaxis, :] - points2[np.newaxis, :, :]
            box.wrap(all_vectors.reshape((-1, 3)))
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < rcut**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            nq = self.build_query_object(box, points2, rcut)
            result = list(nq.queryBall(points, rcut))
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

        # rcut is slightly smaller than the distance for any particle
        nq = self.build_query_object(box, positions, L/10)
        result = list(nq.queryBall(positions, 0.99, exclude_ii=True))

        self.assertEqual(len(result), 0)

    def test_corner_2d(self):
        """Check an extreme case where finding enough nearest neighbors
        requires going beyond normally allowed cutoff."""
        L = 2.1
        box = freud.box.Box.square(L)

        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        nq = self.build_query_object(box, positions, L/10)
        result = list(nq.query(positions[[0]], 3))
        self.assertEqual(get_point_neighbors(result, 0), {0, 1, 2})

        # Check the effect of points != ref_points
        positions[:, :2] -= 0.1
        result = list(nq.query(positions[[0]], 3))
        self.assertEqual(get_point_neighbors(result, 0), {0, 1, 2})

        # Since the initial position set aligns exactly with cell boundaries,
        # make sure that the correctness is not affected by that artifact.
        nq = self.build_query_object(box, positions, L/10)
        result = list(nq.query(positions[[0]], 3))
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

                nlist = nq.query(positions, k=k, exclude_ii=True).toNList()
                assert len(nlist) == k * N,\
                    'Wrong nlist length for N = {}, k= {}, length={}'.format(
                        N, k, len(nlist))
                nlist_array = nlist[:]
                for i in range(N):
                    assert not ([i, i] == nlist_array).all(axis=1).any()

                nlist = nq.query(positions, k=k, exclude_ii=False).toNList()
                assert len(nlist) == k * N,\
                    'Wrong nlist length for N = {}, k= {}, length={}'.format(
                        N, k, len(nlist))
                nlist_array = nlist[:]
                for i in range(N):
                    assert ([i, i] == nlist_array).all(axis=1).any()


class TestNeighborQueryAABB(TestNeighborQuery, unittest.TestCase):
    @classmethod
    def build_query_object(cls, box, ref_points, rcut=None):
        return freud.locality.AABBQuery(box, ref_points)

    def test_throws(self):
        """Test that specifying too large an rcut value throws an error"""
        L = 5

        box = freud.box.Box.square(L)
        points = [[0, 0, 0], [1, 1, 0], [1, -1, 0]]
        aq = freud.locality.AABBQuery(box, points)
        with self.assertRaises(RuntimeError):
            list(aq.queryBall(points, L))

    def test_chaining(self):
        N = 500
        L = 10
        rcut = 1
        box, points = makeBoxAndRandomPoints(L, N)
        nlist1 = freud.locality.AABBQuery(box, points).queryBall(
            points, rcut, exclude_ii=True).toNList()
        abq = freud.locality.AABBQuery(box, points)
        nlist2 = abq.queryBall(points, rcut, exclude_ii=True).toNList()
        self.assertTrue(nlist_equal(nlist1, nlist2))


class TestNeighborQueryLinkCell(TestNeighborQuery, unittest.TestCase):
    @classmethod
    def build_query_object(cls, box, ref_points, rcut=None):
        if rcut is None:
            raise ValueError("Building LinkCells requires passing an rcut.")
        return freud.locality.LinkCell(box, rcut, ref_points)

    def test_throws(self):
        """Ensure that mixing old and new APIs throws an error"""
        L = 10
        rcut = 1.0

        box = freud.box.Box.cube(L)
        with self.assertRaises(RuntimeError):
            points = np.zeros(shape=(2, 3), dtype=np.float32)
            freud.locality.LinkCell(box, rcut, points).compute(box, points)

        with self.assertRaises(RuntimeError):
            points = np.zeros(shape=(2, 3), dtype=np.float32)
            freud.locality.LinkCell(box, rcut).query(points, rcut)

    def test_chaining(self):
        N = 500
        L = 10
        rcut = 1
        box, points = makeBoxAndRandomPoints(L, N)
        nlist1 = freud.locality.LinkCell(box, 1.0, points).queryBall(
            points, rcut, exclude_ii=True).toNList()
        lc = freud.locality.LinkCell(box, 1.0, points)
        nlist2 = lc.queryBall(points, rcut, exclude_ii=True).toNList()
        self.assertTrue(nlist_equal(nlist1, nlist2))


if __name__ == '__main__':
    unittest.main()
