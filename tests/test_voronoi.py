import numpy as np
import numpy.testing as npt
from freud import box, voronoi, common
import unittest
import util


@util.skipIfMissing('scipy.spatial')
class TestVoronoi(unittest.TestCase):
    def test_basic(self):
        # Test that voronoi tessellations of random systems have the same
        # number of points and polytopes
        L = 10  # Box length
        N = 50  # Number of particles
        fbox = box.Box.square(L)  # Initialize box
        vor = voronoi.Voronoi(fbox)
        np.random.seed(0)
        # Generate random points in the box
        positions = np.random.uniform(-L/2, L/2, size=(N, 2))
        # Add a z-component of 0
        positions = np.insert(positions, 2, 0, axis=1).astype(np.float32)
        vor.compute(positions, box=fbox, buff=L/2)

        result = vor.getVoronoiPolytopes()

        npt.assert_equal(len(result), len(positions))

    def test_voronoi_tess_2d(self):
        # Test that the voronoi polytope works for a 2D system
        L = 10  # Box length
        fbox = box.Box.square(L)
        vor = voronoi.Voronoi(fbox)
        # Make a regular grid
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float32)
        vor.compute(positions)
        npt.assert_equal(
            vor.getVoronoiPolytopes(),
            [np.array([[1.5,  1.5, 0], [0.5,  1.5, 0],
                       [0.5,  0.5, 0], [1.5,  0.5, 0]])])
        # Verify the cell areas
        vor.computeVolumes()
        npt.assert_equal(vor.getVolumes(), [1])

    def test_voronoi_tess_3d(self):
        # Test that the voronoi polytope works for a 3D system
        L = 10  # Box length
        fbox = box.Box.cube(L)
        vor = voronoi.Voronoi(fbox)
        # Make a regular grid
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0],
             [0, 0, 1], [0, 1, 1], [0, 2, 1],
             [1, 0, 1], [1, 1, 1], [1, 2, 1],
             [2, 0, 1], [2, 1, 1], [2, 2, 1],
             [0, 0, 2], [0, 1, 2], [0, 2, 2],
             [1, 0, 2], [1, 1, 2], [1, 2, 2],
             [2, 0, 2], [2, 1, 2], [2, 2, 2]]).astype(np.float32)
        vor.compute(positions)
        npt.assert_equal(
            vor.getVoronoiPolytopes(),
            [np.array([[1.5, 1.5, 1.5], [1.5, 0.5, 1.5], [1.5, 0.5, 0.5],
                       [1.5, 1.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 1.5],
                       [0.5, 1.5, 0.5], [0.5, 1.5, 1.5]])])
        # Verify the cell volumes
        vor.computeVolumes()
        npt.assert_equal(vor.getVolumes(), [1])

    def test_voronoi_neighbors(self):
        # Test that voronoi neighbors in the first and second shells are
        # correct in 2D
        L = 10  # Box length
        fbox = box.Box.square(L)
        vor = voronoi.Voronoi(fbox)
        # Make a regular grid
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float32)
        vor.computeNeighbors(positions)
        npt.assert_equal(
            vor.getNeighbors(1),
            [[1, 3], [0, 2, 4], [5, 1], [0, 6, 4], [3, 5, 1, 7], [8, 2, 4],
             [3, 7], [6, 8, 4], [5, 7]])
        npt.assert_equal(
            vor.getNeighbors(2),
            [[1, 2, 3, 4, 6], [0, 2, 3, 4, 5, 7], [0, 1, 4, 5, 8],
             [0, 1, 4, 5, 6, 7], [0, 1, 2, 3, 5, 6, 7, 8],
             [1, 2, 3, 4, 7, 8], [0, 3, 4, 7, 8], [1, 3, 4, 5, 6, 8],
             [2, 4, 5, 6, 7]])

    def test_voronoi_neighbors_wrapped(self):
        # Test that voronoi neighbors in the first shell are
        # correct for a wrapped 3D system

        L = 3.0  # Box length
        fbox = box.Box.cube(L)
        rbuf = L/2

        # Make a simple cubic structure
        positions = np.array([[i + 0.5 - L/2,
                               j + 0.5 - L/2,
                               k + 0.5 - L/2]
                              for i in range(int(L))
                              for j in range(int(L))
                              for k in range(int(L))]).astype(np.float32)
        vor = voronoi.Voronoi(fbox)
        vor.computeNeighbors(positions, fbox, rbuf)
        nlist = vor.getNeighborList()

        unique_indices, counts = np.unique(nlist.index_i, return_counts=True)

        # Every particle should have six neighbors
        npt.assert_equal(counts, np.full(len(unique_indices), 6))

    def test_voronoi_weights_fcc(self):
        # Test that voronoi neighbor weights are computed properly for 3D FCC

        L = 3
        fbox, positions = util.make_fcc(nx=L, ny=L, nz=L)
        rbuf = np.max(fbox.L)/2

        vor = voronoi.Voronoi(fbox)
        vor.computeNeighbors(positions, fbox, rbuf)
        nlist = vor.getNeighborList()

        # Drop the tiny facets that come from numerical imprecision
        nlist = nlist.filter(nlist.weights > 1e-12)

        unique_indices, counts = np.unique(nlist.index_i, return_counts=True)

        # Every FCC particle should have 12 neighbors
        npt.assert_equal(counts, np.full(len(unique_indices), 12))

        # Every facet area should be sqrt(2)/2
        npt.assert_allclose(nlist.weights,
                            np.full(len(nlist.weights), 0.5*np.sqrt(2)),
                            atol=1e-5)
        # Every cell should have volume 2
        vor.compute(positions)
        npt.assert_allclose(vor.computeVolumes().getVolumes(),
                            np.full(len(vor.getVoronoiPolytopes()), 2.),
                            atol=1e-5)

    def test_nlist_symmetric(self):
        # Test that the voronoi neighborlist is symmetric
        L = 10  # Box length
        rbuf = 3  # Cutoff radius
        N = 40  # Number of particles

        fbox = box.Box.cube(L)  # Initialize box
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        vor = voronoi.Voronoi(fbox)
        vor.computeNeighbors(points, fbox, rbuf)
        nlist = vor.getNeighborList()

        ijs = set(zip(nlist.index_i, nlist.index_j))
        jis = set(zip(nlist.index_j, nlist.index_i))

        # we shouldn't have duplicate (i, j) pairs in the
        # resulting neighbor list
        npt.assert_equal(len(ijs), len(nlist))
        npt.assert_equal(len(ijs), len(jis))

        # every (i, j) pair should have a corresponding (j, i) pair
        self.assertTrue(all((j, i) in jis for (i, j) in ijs))


if __name__ == '__main__':
    unittest.main()
