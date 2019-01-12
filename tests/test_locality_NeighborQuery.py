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
        npt.assert_equal(len(aq.queryBall(points[[0]], rcut)), 3)
        # particle 1 has 4 bonds
        npt.assert_equal(len(aq.queryBall(points[[1]], rcut)), 4)
        # particle 2 has 3 bonds
        npt.assert_equal(len(aq.queryBall(points[[2]], rcut)), 3)
        # particle 3 has 4 bonds
        npt.assert_equal(len(aq.queryBall(points[[3]], rcut)), 4)

        # When excluding, everything has one less neighbor.
        npt.assert_equal(len(aq.queryBall(points, rcut, exclude_ii=True)), 10)

        # now move particle 0 out of range...
        points[0] = 5

        # particle 0 has no bonds now
        npt.assert_equal(len(aq.queryBall(points[[0]], rcut)), 0)

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

        result = aq.query(points, 3)
        npt.assert_equal({x[1] for x in result if x[0] == 0}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 1}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 2}, {1, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 3}, {1, 2, 3})

        # All points are neighbors in this case
        result = aq.query(points, 3, exclude_ii=True)
        npt.assert_equal({x[1] for x in result if x[0] == 0}, {1, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 1}, {0, 2, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 2}, {0, 1, 3})
        npt.assert_equal({x[1] for x in result if x[0] == 3}, {0, 1, 2})

        # Test overflow case
        npt.assert_equal(aq.query(points, 5, exclude_ii=True), result)

    def test_reciprocal(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for one set of points"""
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        aq = locality.AABBQuery(fbox, points)
        result = aq.queryBall(points, rcut)

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

        result = aq.queryBall(points2, rcut)
        result2 = aq2.queryBall(points, rcut)

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
        result = aq.queryBall(points2, rcut)

        ij1 = {(x[0], x[1]) for x in result}

        result2 = aq.queryBall(points2, rcut, exclude_ii=True)

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
            result = aq.queryBall(points, rcut, exclude_ii=True)
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
            result = aq.queryBall(points2, rcut)
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
        npt.assert_array_equal(aq.ref_points, points)
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
        result = aq.queryBall(positions, 0.99, exclude_ii=True)

        self.assertEqual(len(result), 0)


class TestNeighborQueryLinkCell(unittest.TestCase):
    def test_first_index(self):
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
        npt.assert_equal(len(lc.queryBall(points[[0]], rcut)), 3)
        # particle 1 has 3 bonds
        npt.assert_equal(len(lc.queryBall(points[[1]], rcut)), 4)
        # particle 2 has 2 bonds
        npt.assert_equal(len(lc.queryBall(points[[2]], rcut)), 3)
        # particle 3 has 3 bonds
        npt.assert_equal(len(lc.queryBall(points[[3]], rcut)), 4)

        # When excluding, everything has one less neighbor.
        npt.assert_equal(len(lc.queryBall(points, rcut, exclude_ii=True)), 10)

        # now move particle 0 out of range...
        points[0] = 5

        # particle 0 has 0 bonds
        npt.assert_equal(len(lc.queryBall(points[[0]], rcut)), 0)
        # particle 1 has 2 bonds
        npt.assert_equal(len(lc.queryBall(points[[1]], rcut)), 4)
        # particle 2 has 2 bonds
        npt.assert_equal(len(lc.queryBall(points[[2]], rcut)), 3)
        # particle 3 has 2 bonds
        npt.assert_equal(len(lc.queryBall(points[[3]], rcut)), 4)

    def test_reciprocal(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for one set of points"""
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        lc = locality.LinkCell(fbox, rcut, points)
        result = lc.queryBall(points, rcut)

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

        result = lc.queryBall(points2, rcut)
        result2 = lc2.queryBall(points, rcut)

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
        result = lc.queryBall(points2, rcut)

        ij1 = {(x[0], x[1]) for x in result}

        result2 = lc.queryBall(points2, rcut, exclude_ii=True)

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
            result = lc.queryBall(points, rcut, exclude_ii=True)
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
            result = lc.queryBall(points2, rcut)
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
        result = lc.queryBall(positions, rcut, exclude_ii=True)

        self.assertEqual(len(result), 0)
