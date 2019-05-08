import numpy as np
import numpy.testing as npt
from freud import locality, box
from collections import Counter
import itertools
import sys
import unittest
import warnings


class TestNeighborQueryAABB(unittest.TestCase):
    def test_query_ball(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        aq = locality.AABBQuery(fbox, points)

        # particle 0 has 3 bonds
        npt.assert_equal(len(list(aq.queryBall(points[[0]], rcut))), 3)
        # particle 1 has 4 bonds
        npt.assert_equal(len(list(aq.queryBall(points[[1]], rcut))), 4)
        # particle 2 has 3 bonds
        npt.assert_equal(len(list(aq.queryBall(points[[2]], rcut))), 3)
        # particle 3 has 4 bonds
        npt.assert_equal(len(list(aq.queryBall(points[[3]], rcut))), 4)

        # Check NeighborList length without self-exclusions.
        npt.assert_equal(
            len(aq.queryBall(points, rcut, exclude_ii=False).toNList()), 14)

        # When excluding, check that everything has one less neighbor and that
        # the toNList method correctly excludes neighbors.
        list_of_neighbors = sorted(list(
            aq.queryBall(points, rcut, exclude_ii=True)))
        # Remove distances, which are not used here.
        list_of_neighbors = [[p[0], p[1]] for p in list_of_neighbors]
        nlist = aq.queryBall(points, rcut, exclude_ii=True).toNList()
        nlist_neighbors = sorted(list(zip(nlist.index_i, nlist.index_j)))

        # Number of neighbors is one less per-particle than before.
        npt.assert_equal(len(list_of_neighbors), 10)
        npt.assert_equal(len(nlist_neighbors), 10)

        # Actual set of neighbors should be identical from both methods.
        npt.assert_array_equal(list_of_neighbors, nlist_neighbors)

        # now move particle 0 out of range...
        points[0] = 5

        # particle 0 has no bonds now
        npt.assert_equal(len(list(aq.queryBall(points[[0]], rcut))), 0)

    def test_query(self):
        L = 10  # Box Dimensions
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        aq = locality.AABBQuery(fbox, points)

        result = list(aq.query(points, 3))
        npt.assert_equal({x[1] for x in result if x[0] == 0}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 1}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 2}, {1, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 3}, {1, 2, 3})

        # All points are neighbors in this case
        result = list(aq.query(points, 3, exclude_ii=True))
        npt.assert_equal({x[1] for x in result if x[0] == 0}, {1, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 1}, {0, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 2}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 3}, {0, 1, 2})

        # Test overflow case
        npt.assert_equal(list(aq.query(points, 5, exclude_ii=True)), result)

    def test_query_ball_to_nlist(self):
        """Test that generated NeighborLists are identical to the results of
        querying"""
        L = 10  # Box Dimensions
        N = 400  # number of particles

        fbox = box.Box.cube(L)
        np.random.seed(0)
        ref_points = np.random.rand(N, 3)*L
        points = np.random.rand(N, 3)*L

        aq = locality.AABBQuery(fbox, ref_points)

        result_list = list(aq.queryBall(points, 2))
        result_list = [(b[0], b[1]) for b in result_list]
        nlist = aq.queryBall(points, 2).toNList()
        list_nlist = list(zip(nlist.index_j, nlist.index_i))

        npt.assert_equal(set(result_list), set(list_nlist))

    def test_query_to_nlist(self):
        """Test that generated NeighborLists are identical to the results of
        querying"""
        L = 10  # Box Dimensions
        N = 400  # number of particles

        fbox = box.Box.cube(L)
        np.random.seed(0)
        ref_points = np.random.rand(N, 3)*L
        points = np.random.rand(N, 3)*L

        aq = locality.AABBQuery(fbox, ref_points)

        result_list = list(aq.query(points, 2))
        result_list = [(b[0], b[1]) for b in result_list]
        nlist = aq.query(points, 2).toNList()
        list_nlist = list(zip(nlist.index_j, nlist.index_i))

        npt.assert_equal(set(result_list), set(list_nlist))

    def test_reciprocal(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for one set of points"""
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        aq = locality.AABBQuery(fbox, points)
        result = list(aq.queryBall(points, rcut))

        ij = {(x[0], x[1]) for x in result}
        ji = set((j, i) for (i, j) in ij)

        self.assertEqual(ij, ji)

    def test_reciprocal_twoset(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for two sets of
        different points
        """
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        points2 = np.random.uniform(-L/2, L/2, (N//6, 3)).astype(np.float32)
        aq = locality.AABBQuery(fbox, points)
        aq2 = locality.AABBQuery(fbox, points2)

        result = list(aq.queryBall(points2, rcut))
        result2 = list(aq2.queryBall(points, rcut))

        ij = {(x[0], x[1]) for x in result}
        ij2 = {(x[1], x[0]) for x in result2}

        self.assertEqual(ij, ij2)

    def test_exclude_ii(self):
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        points2 = points[:N//6]
        aq = locality.AABBQuery(fbox, points)
        result = list(aq.queryBall(points2, rcut))

        ij1 = {(x[0], x[1]) for x in result}

        result2 = list(aq.queryBall(points2, rcut, exclude_ii=True))

        ij2 = {(x[0], x[1]) for x in result2}

        self.assertTrue(all((i, i) not in ij2 for i in range(N)))

        ij2.update((i, i) for i in range(points2.shape[0]))

        self.assertEqual(ij1, ij2)

    def test_exhaustive_search(self):
        L, rcut, N = (10, 1.999, 32)

        fbox = box.Box.cube(L)
        seed = 0

        for i in range(10):
            np.random.seed(seed + i)
            points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
            all_vectors = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            fbox.wrap(all_vectors.reshape((-1, 3)))
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < rcut**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            aq = locality.AABBQuery(fbox, points)
            result = list(aq.queryBall(points, rcut, exclude_ii=True))
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

    def test_exhaustive_search_assymmetric(self):
        L, rcut, N = (10, 1.999, 32)

        fbox = box.Box.cube(L)
        seed = 0

        for i in range(10):
            np.random.seed(seed + i)
            points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
            points2 = np.random.uniform(
                -L/2, L/2, (N//2, 3)).astype(np.float32)
            all_vectors = points[:, np.newaxis, :] - points2[np.newaxis, :, :]
            fbox.wrap(all_vectors.reshape((-1, 3)))
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < rcut**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            aq = locality.AABBQuery(fbox, points)
            result = list(aq.queryBall(points2, rcut))
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

    def test_attributes(self):
        """Ensure that mixing old and new APIs throws an error"""
        L = 10

        fbox = box.Box.cube(L)

        points = np.zeros(shape=(4, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]

        aq = locality.AABBQuery(fbox, points)
        npt.assert_array_equal(aq.points, points)
        self.assertEqual(aq.box, fbox)

    def test_no_bonds(self):
        N = 10
        fbox = box.Box.cube(N)

        # make a sc lattice
        lattice_xs = np.linspace(-float(N)/2, float(N)/2, N, endpoint=False)
        positions = list(itertools.product(lattice_xs, lattice_xs, lattice_xs))
        positions = np.array(positions, dtype=np.float32)

        # rcut is slightly smaller than the distance for any particle
        aq = locality.AABBQuery(fbox, positions)
        result = list(aq.queryBall(positions, 0.99, exclude_ii=True))

        self.assertEqual(len(result), 0)


class TestNeighborQueryLinkCell(unittest.TestCase):
    def test_query_generic(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        lc = locality.LinkCell(fbox, rcut, points)

        nlist = lc._queryGeneric(points, dict(mode='ball', rmax=rcut))
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
        nlist = lc._queryGeneric(points, dict(mode='ball', rmax=rcut, exclude_ii=True))
        nlist_neighbors = sorted(list(zip(nlist.index_i, nlist.index_j)))
        # When excluding, everything has one less neighbor.
        npt.assert_equal(len(nlist_neighbors), 10)

        # now move particle 0 out of range...
        points[0] = 5

        lc = locality.LinkCell(fbox, rcut, points)
        nlist = lc._queryGeneric(points, dict(mode='ball', rmax=rcut))
        nlist_neighbors = sorted(list(zip(nlist.index_i, nlist.index_j)))
        # particle 0 has 0 bonds
        npt.assert_equal(sum(nlist.index_i == 0), 1)
        # particle 1 has 2 bonds
        npt.assert_equal(sum(nlist.index_i == 1), 3)
        # particle 2 has 2 bonds
        npt.assert_equal(sum(nlist.index_i == 2), 3)
        # particle 3 has 2 bonds
        npt.assert_equal(sum(nlist.index_i == 3), 3)

    def test_query_ball(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        lc = locality.LinkCell(fbox, rcut, points)

        # particle 0 has 2 bonds
        npt.assert_equal(len(list(lc.queryBall(points[[0]], rcut))), 3)
        # particle 1 has 3 bonds
        npt.assert_equal(len(list(lc.queryBall(points[[1]], rcut))), 4)
        # particle 2 has 2 bonds
        npt.assert_equal(len(list(lc.queryBall(points[[2]], rcut))), 3)
        # particle 3 has 3 bonds
        npt.assert_equal(len(list(lc.queryBall(points[[3]], rcut))), 4)

        # Check NeighborList length without self-exclusions.
        npt.assert_equal(
            len(lc.queryBall(points, rcut, exclude_ii=False).toNList()), 14)

        # When excluding, everything has one less neighbor.
        npt.assert_equal(
            len(list(lc.queryBall(points, rcut, exclude_ii=True))), 10)

        # Check NeighborList length with self-exclusions.
        npt.assert_equal(
            len(lc.queryBall(points, rcut, exclude_ii=True).toNList()), 10)

        # Now move particle 0 out of range. It now only has one neighbor
        # (itself), while the other three are all neighbors with each other.
        points[0] = 5
        lc = locality.LinkCell(fbox, rcut, points)

        # particle 0 has 1 bond
        npt.assert_equal(len(list(lc.queryBall(points[[0]], rcut))), 1)
        # particle 1 has 2 bonds
        npt.assert_equal(len(list(lc.queryBall(points[[1]], rcut))), 3)
        # particle 2 has 2 bonds
        npt.assert_equal(len(list(lc.queryBall(points[[2]], rcut))), 3)
        # particle 3 has 2 bonds
        npt.assert_equal(len(list(lc.queryBall(points[[3]], rcut))), 3)

    def test_query(self):
        L = 10  # Box Dimensions
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]
        lc = locality.LinkCell(fbox, L/10, points)

        result = list(lc.query(points, 3))
        npt.assert_equal({x[1] for x in result if x[0] == 0}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 1}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 2}, {1, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 3}, {1, 2, 3})

        # All points are neighbors in this case
        result = list(lc.query(points, 3, exclude_ii=True))
        npt.assert_equal({x[1] for x in result if x[0] == 0}, {1, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 1}, {0, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 2}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 3}, {0, 1, 2})

        npt.assert_equal(list(lc.query(points, 5, exclude_ii=True)), result)

    def test_query_ball_to_nlist(self):
        """Test that generated NeighborLists are identical to the results of
        querying"""
        L = 10  # Box Dimensions
        N = 400  # number of particles

        fbox = box.Box.cube(L)
        np.random.seed(0)
        ref_points = np.random.rand(N, 3)*L
        points = np.random.rand(N, 3)*L

        lc = locality.LinkCell(fbox, L/10, ref_points)

        result_list = list(lc.queryBall(points, 2))
        result_list = [(b[0], b[1]) for b in result_list]
        nlist = lc.queryBall(points, 2).toNList()
        list_nlist = list(zip(nlist.index_j, nlist.index_i))

        npt.assert_equal(set(result_list), set(list_nlist))

    def test_query_to_nlist(self):
        """Test that generated NeighborLists are identical to the results of
        querying"""
        L = 10  # Box Dimensions
        N = 400  # number of particles

        fbox = box.Box.cube(L)
        np.random.seed(0)
        ref_points = np.random.rand(N, 3)*L
        points = np.random.rand(N, 3)*L

        lc = locality.LinkCell(fbox, L/10, ref_points)

        result_list = list(lc.query(points, 2))
        result_list = [(b[0], b[1]) for b in result_list]
        nlist = lc.query(points, 2).toNList()
        list_nlist = list(zip(nlist.index_j, nlist.index_i))

        npt.assert_equal(set(result_list), set(list_nlist))

    def test_reciprocal(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for one set of points"""
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        lc = locality.LinkCell(fbox, rcut, points)
        result = list(lc.queryBall(points, rcut))

        ij = {(x[0], x[1]) for x in result}
        ji = set((j, i) for (i, j) in ij)

        self.assertEqual(ij, ji)

    def test_reciprocal_twoset(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for two sets of
        different points
        """
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        points2 = np.random.uniform(-L/2, L/2, (N//6, 3)).astype(np.float32)
        lc = locality.LinkCell(fbox, rcut, points)
        lc2 = locality.LinkCell(fbox, rcut, points2)

        result = list(lc.queryBall(points2, rcut))
        result2 = list(lc2.queryBall(points, rcut))

        ij = {(x[0], x[1]) for x in result}
        ij2 = {(x[1], x[0]) for x in result2}

        self.assertEqual(ij, ij2)

    def test_exclude_ii(self):
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        points2 = points[:N//6]
        lc = locality.LinkCell(fbox, rcut, points)
        result = list(lc.queryBall(points2, rcut))

        ij1 = {(x[0], x[1]) for x in result}

        result2 = list(lc.queryBall(points2, rcut, exclude_ii=True))

        ij2 = {(x[0], x[1]) for x in result2}

        self.assertTrue(all((i, i) not in ij2 for i in range(N)))

        ij2.update((i, i) for i in range(points2.shape[0]))

        self.assertEqual(ij1, ij2)

    def test_exhaustive_search(self):
        L, rcut, N = (10, 1.999, 32)

        fbox = box.Box.cube(L)
        seed = 0

        for i in range(10):
            np.random.seed(seed + i)
            points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
            all_vectors = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            fbox.wrap(all_vectors.reshape((-1, 3)))
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < rcut**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            lc = locality.LinkCell(fbox, rcut, points)
            result = list(lc.queryBall(points, rcut, exclude_ii=True))
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

    def test_exhaustive_search_assymmetric(self):
        L, rcut, N = (10, 1.999, 32)

        fbox = box.Box.cube(L)
        seed = 0

        for i in range(10):
            np.random.seed(seed + i)
            points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
            points2 = np.random.uniform(
                -L/2, L/2, (N//2, 3)).astype(np.float32)
            all_vectors = points[:, np.newaxis, :] - points2[np.newaxis, :, :]
            fbox.wrap(all_vectors.reshape((-1, 3)))
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < rcut**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            lc = locality.LinkCell(fbox, rcut, points)
            result = list(lc.queryBall(points2, rcut))
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

    def test_throws(self):
        """Ensure that mixing old and new APIs throws an error"""
        L = 10
        rcut = 1.0

        fbox = box.Box.cube(L)
        with self.assertRaises(RuntimeError):
            points = np.zeros(shape=(2, 3), dtype=np.float32)
            locality.LinkCell(fbox, rcut, points).compute(fbox, points)

        with self.assertRaises(RuntimeError):
            points = np.zeros(shape=(2, 3), dtype=np.float32)
            locality.LinkCell(fbox, rcut).query(points, rcut)

    def test_no_bonds(self):
        N = 10
        rcut = 0.99
        fbox = box.Box.cube(N)

        # make a sc lattice
        lattice_xs = np.linspace(-float(N)/2, float(N)/2, N, endpoint=False)
        positions = list(itertools.product(lattice_xs, lattice_xs, lattice_xs))
        positions = np.array(positions, dtype=np.float32)

        # rcut is slightly smaller than the distance for any particle
        lc = locality.LinkCell(fbox, rcut, positions)
        result = list(lc.queryBall(positions, rcut, exclude_ii=True))

        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
