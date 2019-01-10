import numpy as np
import numpy.testing as npt
from freud import locality, box
from collections import Counter
import itertools
import sys
import unittest
import warnings


class TestAABBQuery(unittest.TestCase):
    def test_first_index(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box
        aq = locality.AABBQuery()

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]

        aq.compute(fbox, rcut, points)
        # particle 0 has 2 bonds
        npt.assert_equal(aq.nlist.find_first_index(0), 0)
        # particle 1 has 3 bonds
        npt.assert_equal(aq.nlist.find_first_index(1), 2)
        # particle 2 has 2 bonds
        npt.assert_equal(aq.nlist.find_first_index(2), 5)
        # particle 3 has 3 bonds
        npt.assert_equal(aq.nlist.find_first_index(3), 7)

        # now move particle 0 out of range...
        points[0] = 5
        aq.compute(fbox, rcut, points)

        # particle 0 has 0 bonds
        npt.assert_equal(aq.nlist.find_first_index(0), 0)
        # particle 1 has 2 bonds
        npt.assert_equal(aq.nlist.find_first_index(1), 0)
        # particle 2 has 2 bonds
        npt.assert_equal(aq.nlist.find_first_index(2), 2)
        # particle 3 has 2 bonds
        npt.assert_equal(aq.nlist.find_first_index(3), 4)

    def test_reciprocal(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for one set of points"""
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        aq = locality.AABBQuery().compute(fbox, rcut, points)

        ij = set(zip(aq.nlist.index_i, aq.nlist.index_j))
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
        aq = locality.AABBQuery().compute(fbox, rcut, points, points2)
        aq2 = locality.AABBQuery().compute(fbox, rcut, points2, points)

        ij = set(zip(aq.nlist.index_i, aq.nlist.index_j))
        ij2 = set(zip(aq2.nlist.index_j, aq2.nlist.index_i))

        self.assertEqual(ij, ij2)

    def test_exclude_ii(self):
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        points2 = points[:N//6]
        aq = locality.AABBQuery().compute(
            fbox, rcut, points, points2, exclude_ii=False)

        ij1 = set(zip(aq.nlist.index_i, aq.nlist.index_j))

        aq.compute(fbox, rcut, points, points2, exclude_ii=True)

        ij2 = set(zip(aq.nlist.index_i, aq.nlist.index_j))

        self.assertTrue(all((i, i) not in ij2 for i in range(N)))

        ij2.update((i, i) for i in range(points2.shape[0]))

        self.assertEqual(ij1, ij2)

    def test_exhaustive_search(self):
        L, rcut, N = (10, 1.999, 32)

        fbox = box.Box.cube(L)
        seed = 0
        aq = locality.AABBQuery()

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

            aq.compute(fbox, rcut, points, points, exclude_ii=True)
            ijs = set(zip(aq.nlist.index_i, aq.nlist.index_j))
            counts_list = aq.nlist.neighbor_counts.tolist()

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
        aq = locality.AABBQuery()

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

            aq.compute(fbox, rcut, points, points2, exclude_ii=False)
            ijs = set(zip(aq.nlist.index_i, aq.nlist.index_j))
            counts_list = aq.nlist.neighbor_counts.tolist()

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
        L = 10

        fbox = box.Box.cube(L)
        with self.assertRaises(RuntimeError):
            locality.AABBQuery().compute(fbox, L/1.9999, np.zeros((1, 3)))

        fbox = box.Box(L, 2*L, 2*L)
        with self.assertRaises(RuntimeError):
            locality.AABBQuery().compute(fbox, L/1.9999, np.zeros((1, 3)))

    def test_no_bonds(self):
        N = 10
        fbox = box.Box.cube(N)

        # make a sc lattice
        lattice_xs = np.linspace(-float(N)/2, float(N)/2, N, endpoint=False)
        positions = list(itertools.product(lattice_xs, lattice_xs, lattice_xs))
        positions = np.array(positions, dtype=np.float32)

        # rcut is slightly smaller than the distance for any particle
        aq = locality.AABBQuery()
        nlist = aq.compute(fbox, 0.99, positions, positions).nlist

        self.assertEqual(nlist.neighbor_counts.tolist(),
                         np.zeros((N**3,), dtype=np.uint32).tolist())


if __name__ == '__main__':
    unittest.main()
