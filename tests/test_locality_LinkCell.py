import numpy as np
import numpy.testing as npt
from freud import locality, box
from freud.errors import FreudDeprecationWarning
from collections import Counter
import itertools
import sys
import unittest
import warnings


class TestLinkCell(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_unique_neighbors(self):
        L = 10  # Box Dimensions
        rcut = 3  # Cutoff radius

        # Initialize Box, initialize and compute cell list
        fbox = box.Box.cube(L)
        cl = locality.LinkCell(fbox, rcut)
        cl.compute(fbox, np.zeros((1, 3), dtype=np.float32))

        # Ensure deprecated method works
        cl.computeCellList(fbox, np.zeros((1, 3), dtype=np.float32))

        # 27 is the total number of cells
        for i in range(27):
            neighbors = cl.getCellNeighbors(i)
            self.assertEqual(
                len(np.unique(neighbors)), 27,
                msg="Cell %d does not have 27 unique adjacent cell indices, "
                "it has %d" % (i, len(np.unique(neighbors))))

    def test_bug2100(self):
        L = 31  # Box Dimensions
        rcut = 3  # Cutoff radius

        # Initialize test points across periodic boundary condition
        testpoints = np.array([[-5.0, 0, 0],
                               [2.05, 0, 0]], dtype=np.float32)

        # Initialize Box, initialize and compute cell list
        fbox = box.Box.cube(L)
        cl = locality.LinkCell(fbox, rcut)
        cl.compute(fbox, testpoints)

        # Get cell index
        cell_index0 = cl.getCell(testpoints[0])
        cell_index1 = cl.getCell(testpoints[1])

        # Get cell neighbors
        neighbors0 = cl.getCellNeighbors(cell_index0)
        neighbors1 = cl.getCellNeighbors(cell_index1)

        # Check if particle 0 is in a cell neighboring particle 1
        # np.where returns [[index]] if found, otherwise [[]]
        test0 = np.where(neighbors1 == cell_index0)[0]
        test1 = np.where(neighbors0 == cell_index1)[0]
        self.assertEqual(len(test0), len(test1))

    def test_symmetric(self):
        current_version = sys.version_info
        if current_version.major < 3:
            self.assertEqual(1, 1)
        else:
            L = 10  # Box Dimensions
            rcut = 2  # Cutoff radius
            N = 40  # number of particles

            # Initialize test points randomly
            np.random.seed(0)
            points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
            fbox = box.Box.cube(L)  # Initialize Box
            cl = locality.LinkCell(fbox, rcut)  # Initialize cell list
            cl.compute(fbox, points)  # Compute cell list

            neighbors_ij = set()
            for i in range(N):
                cells = cl.getCellNeighbors(cl.getCell(points[i]))
                for cell in cells:
                    neighbors_ij.update([(i, j) for j in cl.itercell(cell)])

            neighbors_ji = set((j, i) for (i, j) in neighbors_ij)
            # if i is a neighbor of j, then j should be a neighbor of i
            self.assertEqual(neighbors_ij, neighbors_ji)

    def test_first_index(self):
        L = 10  # Box Dimensions
        rcut = 2.01  # Cutoff radius
        N = 4  # number of particles

        fbox = box.Box.cube(L)  # Initialize Box
        lc = locality.LinkCell(fbox, rcut)

        points = np.zeros(shape=(N, 3), dtype=np.float32)
        points[0] = [0.0, 0.0, 0.0]
        points[1] = [1.0, 0.0, 0.0]
        points[2] = [3.0, 0.0, 0.0]
        points[3] = [2.0, 0.0, 0.0]

        lc.compute(fbox, points)
        # particle 0 has 2 bonds`
        npt.assert_equal(lc.nlist.find_first_index(0), 0)
        # particle 1 has 3 bonds
        npt.assert_equal(lc.nlist.find_first_index(1), 2)
        # particle 2 has 2 bonds
        npt.assert_equal(lc.nlist.find_first_index(2), 5)
        # particle 3 has 3 bonds
        npt.assert_equal(lc.nlist.find_first_index(3), 7)

        # now move particle 0 out of range...
        points[0] = 5
        lc.compute(fbox, points)

        # particle 0 has 0 bonds
        npt.assert_equal(lc.nlist.find_first_index(0), 0)
        # particle 1 has 2 bonds
        npt.assert_equal(lc.nlist.find_first_index(1), 0)
        # particle 2 has 2 bonds
        npt.assert_equal(lc.nlist.find_first_index(2), 2)
        # particle 3 has 2 bonds
        npt.assert_equal(lc.nlist.find_first_index(3), 4)

    def test_reciprocal(self):
        """Test that, for a random set of points, for each (i, j) neighbor
        pair there also exists a (j, i) neighbor pair for one set of points"""
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        lc = locality.LinkCell(fbox, rcut).compute(fbox, points)

        ij = set(zip(lc.nlist.index_i, lc.nlist.index_j))
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
        lc = locality.LinkCell(fbox, rcut).compute(fbox, points, points2)
        lc2 = locality.LinkCell(fbox, rcut).compute(fbox, points2, points)

        ij = set(zip(lc.nlist.index_i, lc.nlist.index_j))
        ij2 = set(zip(lc2.nlist.index_j, lc2.nlist.index_i))

        self.assertEqual(ij, ij2)

    def test_exclude_ii(self):
        L, rcut, N = (10, 2.01, 1024)

        fbox = box.Box.cube(L)
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        points2 = points[:N//6]
        lc = locality.LinkCell(fbox, rcut).compute(
            fbox, points, points2, exclude_ii=False)

        ij1 = set(zip(lc.nlist.index_i, lc.nlist.index_j))

        lc.compute(fbox, points, points2, exclude_ii=True)

        ij2 = set(zip(lc.nlist.index_i, lc.nlist.index_j))

        self.assertTrue(all((i, i) not in ij2 for i in range(N)))

        ij2.update((i, i) for i in range(points2.shape[0]))

        self.assertEqual(ij1, ij2)

    def test_exhaustive_search(self):
        L, rcut, N = (10, 1.999, 32)

        fbox = box.Box.cube(L)
        seed = 0
        lc = locality.LinkCell(fbox, rcut)

        for i in range(10):
            np.random.seed(seed + i)
            points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
            all_vectors = points[np.newaxis, :, :] - points[:, np.newaxis, :]
            fbox.wrap(all_vectors.reshape((-1, 3)))
            all_rsqs = np.sum(all_vectors**2, axis=-1)
            (exhaustive_i, exhaustive_j) = np.where(np.logical_and(
                all_rsqs < rcut**2, all_rsqs > 0))

            exhaustive_ijs = set(zip(exhaustive_i, exhaustive_j))
            exhaustive_counts = Counter(exhaustive_i)
            exhaustive_counts_list = [exhaustive_counts[j] for j in range(N)]

            lc.compute(fbox, points, points, exclude_ii=True)
            ijs = set(zip(lc.nlist.index_i, lc.nlist.index_j))
            counts_list = lc.nlist.neighbor_counts.tolist()

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

        with self.assertRaises(RuntimeError):
            fbox = box.Box.cube(L)
            locality.LinkCell(fbox, L/1.9999)

        fbox = box.Box(L, 2*L, 2*L)
        locality.LinkCell(fbox, L/2.0001)
        with self.assertRaises(RuntimeError):
            locality.LinkCell(fbox, L/1.9999)

    def test_no_bonds(self):
        N = 10
        fbox = box.Box.cube(N)

        # make a sc lattice
        lattice_xs = np.linspace(-float(N)/2, float(N)/2, N, endpoint=False)
        positions = list(itertools.product(lattice_xs, lattice_xs, lattice_xs))
        positions = np.array(positions, dtype=np.float32)

        # rcut is slightly smaller than the distance for any particle
        lc = locality.LinkCell(fbox, 0.99)
        nlist = lc.compute(fbox, positions, positions).nlist

        self.assertEqual(nlist.neighbor_counts.tolist(),
                         np.zeros((N**3,), dtype=np.uint32).tolist())


if __name__ == '__main__':
    unittest.main()
