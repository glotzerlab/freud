import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


@util.skipIfMissing('scipy.spatial')
class TestVoronoi(unittest.TestCase):
    def test_basic(self):
        # Test that voronoi tessellations of random systems have the same
        # number of points and polytopes
        L = 10  # Box length
        N = 50  # Number of particles
        box = freud.box.Box.square(L)  # Initialize box
        vor = freud.voronoi.Voronoi(box)
        np.random.seed(0)
        # Generate random points in the box
        positions = np.random.uniform(-L/2, L/2, size=(N, 2))
        # Add a z-component of 0
        positions = np.insert(positions, 2, 0, axis=1).astype(np.float32)
        vor.compute(positions, box=box, buff=L/2)

        result = vor.polytopes

        npt.assert_equal(len(result), len(positions))

    def test_voronoi_tess_2d(self):
        # Test that the voronoi polytope works for a 2D system
        L = 10  # Box length
        box = freud.box.Box.square(L)
        vor = freud.voronoi.Voronoi(box)
        # Make a regular grid
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float32)
        vor.compute(positions)
        polytope_centers = set(tuple(point)
                               for point in vor.polytopes[0].tolist())
        check_centers = set([(1.5, 1.5, 0), (0.5, 1.5, 0),
                             (0.5, 0.5, 0), (1.5, 0.5, 0)])
        self.assertEqual(polytope_centers, check_centers)

        # Verify the cell areas
        vor.computeVolumes()
        npt.assert_equal(vor.volumes, [1])

    def test_voronoi_tess_3d(self):
        # Test that the voronoi polytope works for a 3D system
        L = 10  # Box length
        box = freud.box.Box.cube(L)
        vor = freud.voronoi.Voronoi(box)
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
        polytope_centers = set(tuple(point)
                               for point in vor.polytopes[0].tolist())
        check_centers = set([(1.5, 1.5, 1.5), (1.5, 0.5, 1.5), (1.5, 0.5, 0.5),
                             (1.5, 1.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 1.5),
                             (0.5, 1.5, 0.5), (0.5, 1.5, 1.5)])
        self.assertEqual(polytope_centers, check_centers)

        # Verify the cell volumes
        vor.computeVolumes()
        npt.assert_equal(vor.volumes, [1])

    def test_voronoi_neighbors(self):
        # Test that voronoi neighbors in the first and second shells are
        # correct in 2D
        L = 10  # Box length
        box = freud.box.Box.square(L)
        vor = freud.voronoi.Voronoi(box)
        # Make a regular grid
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float32)
        vor.computeNeighbors(positions)
        npt.assert_equal(
            vor.getNeighbors(1),
            [[1, 3], [0, 2, 4], [1, 5], [0, 4, 6], [1, 3, 5, 7], [2, 4, 8],
             [3, 7], [4, 6, 8], [5, 7]])
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
        box = freud.box.Box.cube(L)
        rbuf = L/2

        # Make a simple cubic structure
        positions = np.array([[i + 0.5 - L/2,
                               j + 0.5 - L/2,
                               k + 0.5 - L/2]
                              for i in range(int(L))
                              for j in range(int(L))
                              for k in range(int(L))]).astype(np.float32)
        vor = freud.voronoi.Voronoi(box)
        vor.computeNeighbors(positions, box, rbuf)
        nlist = vor.nlist

        # Drop the tiny facets that come from numerical imprecision
        nlist = nlist.filter(nlist.weights > 1e-7)

        unique_indices, counts = np.unique(nlist.index_i, return_counts=True)

        # Every particle should have six neighbors
        npt.assert_equal(counts, np.full(len(unique_indices), 6))

    def test_voronoi_weights_fcc(self):
        # Test that voronoi neighbor weights are computed properly for 3D FCC

        L = 3
        box, positions = util.make_fcc(nx=L, ny=L, nz=L)
        rbuf = np.max(box.L)/2

        vor = freud.voronoi.Voronoi(box)
        vor.computeNeighbors(positions, box, rbuf)
        nlist = vor.nlist

        # Drop the tiny facets that come from numerical imprecision
        nlist = nlist.filter(nlist.weights > 1e-7)

        unique_indices, counts = np.unique(nlist.index_i, return_counts=True)

        # Every FCC particle should have 12 neighbors
        npt.assert_equal(counts, np.full(len(unique_indices), 12))

        # Every facet area should be sqrt(2)/2
        npt.assert_allclose(nlist.weights,
                            np.full(len(nlist.weights), 0.5*np.sqrt(2)),
                            atol=1e-5)
        # Every cell should have volume 2
        vor.compute(positions)
        npt.assert_allclose(vor.computeVolumes().volumes,
                            np.full(len(vor.polytopes), 2.),
                            atol=1e-5)

    def test_nlist_symmetric(self):
        # Test that the voronoi neighborlist is symmetric
        L = 10  # Box length
        rbuf = 3  # Cutoff radius
        N = 40  # Number of particles

        box = freud.box.Box.cube(L)  # Initialize box
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
        vor = freud.voronoi.Voronoi(box)
        vor.computeNeighbors(points, box, rbuf)
        nlist = vor.nlist

        ijs = set(zip(nlist.index_i, nlist.index_j))
        jis = set(zip(nlist.index_j, nlist.index_i))

        # we shouldn't have duplicate (i, j) pairs in the
        # resulting neighbor list
        npt.assert_equal(len(ijs), len(nlist))
        npt.assert_equal(len(ijs), len(jis))

        # every (i, j) pair should have a corresponding (j, i) pair
        self.assertTrue(all((j, i) in jis for (i, j) in ijs))

    def test_repr(self):
        box = freud.box.Box.cube(10)
        vor = freud.voronoi.Voronoi(box)
        self.assertEqual(str(vor), str(eval(repr(vor))))

    def test_repr_png(self):
        L = 10  # Box length
        box = freud.box.Box.square(L)
        vor = freud.voronoi.Voronoi(box)
        # Make a regular grid
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float32)
        vor.compute(positions)
        vor._repr_png_()

        L = 10  # Box length
        box = freud.box.Box.cube(L)
        vor = freud.voronoi.Voronoi(box)
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
        self.assertEqual(vor._repr_png_(), None)


if __name__ == '__main__':
    unittest.main()
