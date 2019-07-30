import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


def sort_rounded_xyz_array(arr, decimals=4):
    arr = np.asarray(arr)
    arr = arr.round(decimals)
    indices = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[indices]


class TestVoroPlusPlus(unittest.TestCase):
    def test_basic(self):
        # Test that voronoi tessellations of random systems have the same
        # number of points and polytopes
        L = 10  # Box length
        N = 50  # Number of particles
        box = freud.box.Box.square(L)  # Initialize box
        vor = freud.locality._VoroPlusPlus()
        np.random.seed(0)
        # Generate random points in the box
        positions = np.random.uniform(-L/2, L/2, size=(N, 2))
        # Add a z-component of 0
        positions = np.insert(positions, 2, 0, axis=1).astype(np.float64)
        vor.compute(box, positions)

        result = vor.polytopes

        npt.assert_equal(len(result), len(positions))

    def test_voronoi_tess_2d(self):
        # Test that the voronoi polytope works for a 2D system
        L = 10  # Box length
        box = freud.box.Box.square(L)
        vor = freud.locality._VoroPlusPlus()
        # Make a regular grid
        positions = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float64)
        vor.compute(box, positions)
        center_polytope = sort_rounded_xyz_array(vor.polytopes[4])
        expected_polytope = sort_rounded_xyz_array(
            [[1.5, 1.5, 0], [0.5, 1.5, 0], [0.5, 0.5, 0], [1.5, 0.5, 0]])
        npt.assert_almost_equal(center_polytope, expected_polytope)

        # # Verify the cell areas
        npt.assert_almost_equal(vor.volumes[4], 1)
        npt.assert_almost_equal(np.sum(vor.volumes), box.volume)

    def test_voronoi_tess_3d(self):
        # Test that the voronoi polytope works for a 3D system
        L = 10  # Box length
        box = freud.box.Box.cube(L)
        vor = freud.locality._VoroPlusPlus()
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
             [2, 0, 2], [2, 1, 2], [2, 2, 2]]).astype(np.float64)
        vor.compute(box, positions)

        center_polytope = sort_rounded_xyz_array(vor.polytopes[13])
        expected_polytope = sort_rounded_xyz_array(
            [[1.5, 1.5, 1.5], [1.5, 0.5, 1.5], [1.5, 0.5, 0.5],
             [1.5, 1.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 1.5],
             [0.5, 1.5, 0.5], [0.5, 1.5, 1.5]])
        npt.assert_almost_equal(center_polytope, expected_polytope)

        # Verify the cell volumes
        npt.assert_almost_equal(vor.volumes[13], 1)
        npt.assert_almost_equal(np.sum(vor.volumes), box.volume)

    def test_voronoi_neighbors_wrapped(self):
        # Test that voronoi neighbors in the first shell are
        # correct for a wrapped 3D system

        L = 3.0  # Box length
        box = freud.box.Box.cube(L)

        # Make a simple cubic structure
        positions = np.array([[i + 0.5 - L/2,
                               j + 0.5 - L/2,
                               k + 0.5 - L/2]
                              for i in range(int(L))
                              for j in range(int(L))
                              for k in range(int(L))]).astype(np.float64)
        vor = freud.locality._VoroPlusPlus()
        vor.compute(box, positions)
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

        vor = freud.locality._VoroPlusPlus()
        vor.compute(box, positions)
        nlist = vor.nlist

        # Drop the tiny facets that come from numerical imprecision
        nlist = nlist.filter(nlist.weights > 1e-5)

        unique_indices, counts = np.unique(nlist.index_i, return_counts=True)

        # Every FCC particle should have 12 neighbors
        npt.assert_equal(counts, np.full(len(unique_indices), 12))

        # Every facet area should be sqrt(2)/2
        npt.assert_allclose(nlist.weights,
                            np.full(len(nlist.weights), 0.5*np.sqrt(2)),
                            atol=1e-5)
        # Every cell should have volume 2
        vor.compute(box, positions)
        npt.assert_allclose(vor.compute(box, positions).volumes,
                            np.full(len(vor.polytopes), 2.),
                            atol=1e-5)

    def test_nlist_symmetric(self):
        # Test that the voronoi neighborlist is symmetric
        L = 10  # Box length
        N = 40  # Number of particles

        box = freud.box.Box.cube(L)  # Initialize box
        np.random.seed(0)
        points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float64)
        vor = freud.locality._VoroPlusPlus()
        vor.compute(box, points)
        nlist = vor.nlist

        ijs = set(zip(nlist.index_i, nlist.index_j))
        jis = set(zip(nlist.index_j, nlist.index_i))

        # we shouldn't have duplicate (i, j) pairs in the
        # resulting neighbor list
        # npt.assert_equal(len(ijs), len(nlist))
        # npt.assert_equal(len(ijs), len(jis))

        # every (i, j) pair should have a corresponding (j, i) pair
        self.assertTrue(all((j, i) in jis for (i, j) in ijs))

    def test_repr(self):
        vor = freud.locality._VoroPlusPlus()
        self.assertEqual(str(vor), str(eval(repr(vor))))


if __name__ == '__main__':
    unittest.main()
