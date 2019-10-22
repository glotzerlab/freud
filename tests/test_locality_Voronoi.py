import numpy as np
import numpy.testing as npt
import freud
import unittest


def sort_rounded_xyz_array(arr, decimals=4):
    """The order of Voronoi polytopes' vertices is not well-defined. Instead of
    testing a fixed array, arrays must be sorted by their rounded
    representations in order to compare their values.
    """
    arr = np.asarray(arr)
    arr = arr.round(decimals)
    indices = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[indices]


class TestVoronoi(unittest.TestCase):
    def test_basic_2d(self):
        # Test that voronoi tessellations of random systems have the same
        # number of points and polytopes
        L = 10  # Box length
        N = 50  # Number of particles
        box, points = freud.data.make_random_system(L, N, is2D=True)
        vor = freud.locality.Voronoi()
        vor.compute((box, points))

        npt.assert_equal(len(vor.polytopes), len(points))
        npt.assert_equal(len(vor.volumes), len(points))
        npt.assert_almost_equal(np.sum(vor.volumes), box.volume)

    def test_basic_3d(self):
        # Test that voronoi tessellations of random systems have the same
        # number of points and polytopes
        L = 10  # Box length
        N = 50  # Number of particles
        box, points = freud.data.make_random_system(L, N, is2D=False)
        vor = freud.locality.Voronoi()
        vor.compute((box, points))

        npt.assert_equal(len(vor.polytopes), len(points))
        npt.assert_equal(len(vor.volumes), len(points))
        npt.assert_almost_equal(np.sum(vor.volumes), box.volume)

    def test_voronoi_tess_2d(self):
        # Test that the voronoi polytope works for a 2D system
        L = 10  # Box length
        box = freud.box.Box.square(L)
        vor = freud.locality.Voronoi()
        # Make a regular grid
        points = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float64)
        vor.compute((box, points))
        center_polytope = sort_rounded_xyz_array(vor.polytopes[4])
        expected_polytope = sort_rounded_xyz_array(
            [[1.5, 1.5, 0], [0.5, 1.5, 0], [0.5, 0.5, 0], [1.5, 0.5, 0]])
        npt.assert_almost_equal(center_polytope, expected_polytope)

        # Verify the cell areas
        npt.assert_almost_equal(vor.volumes[4], 1)
        npt.assert_almost_equal(np.sum(vor.volumes), box.volume)

        # Verify the neighbor list weights
        npt.assert_almost_equal(
            vor.nlist.weights[vor.nlist.query_point_indices == 4], 1)

        # Double the points (still inside the box) and test again
        points *= 2
        vor.compute((box, points))
        center_polytope = sort_rounded_xyz_array(vor.polytopes[4])
        expected_polytope = sort_rounded_xyz_array(
            [[3, 3, 0], [1, 3, 0], [1, 1, 0], [3, 1, 0]])
        npt.assert_almost_equal(center_polytope, expected_polytope)
        npt.assert_almost_equal(vor.volumes[4], 4)
        npt.assert_almost_equal(np.sum(vor.volumes), box.volume)
        npt.assert_almost_equal(
            vor.nlist.weights[vor.nlist.query_point_indices == 4], 2)

    def test_voronoi_tess_3d(self):
        # Test that the voronoi polytope works for a 3D system
        L = 10  # Box length
        box = freud.box.Box.cube(L)
        vor = freud.locality.Voronoi()
        # Make a regular grid
        points = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0],
             [0, 0, 1], [0, 1, 1], [0, 2, 1],
             [1, 0, 1], [1, 1, 1], [1, 2, 1],
             [2, 0, 1], [2, 1, 1], [2, 2, 1],
             [0, 0, 2], [0, 1, 2], [0, 2, 2],
             [1, 0, 2], [1, 1, 2], [1, 2, 2],
             [2, 0, 2], [2, 1, 2], [2, 2, 2]]).astype(np.float64)
        vor.compute((box, points))

        center_polytope = sort_rounded_xyz_array(vor.polytopes[13])
        expected_polytope = sort_rounded_xyz_array(
            [[1.5, 1.5, 1.5], [1.5, 0.5, 1.5], [1.5, 0.5, 0.5],
             [1.5, 1.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 1.5],
             [0.5, 1.5, 0.5], [0.5, 1.5, 1.5]])
        npt.assert_almost_equal(center_polytope, expected_polytope)

        # Verify the cell volumes
        npt.assert_almost_equal(vor.volumes[13], 1)
        npt.assert_almost_equal(np.sum(vor.volumes), box.volume)

        # Verify the neighbor list weights
        npt.assert_almost_equal(
            vor.nlist.weights[vor.nlist.query_point_indices == 13], 1)

        # Double the points (still inside the box) and test again
        points *= 2
        vor.compute((box, points))
        center_polytope = sort_rounded_xyz_array(vor.polytopes[13])
        expected_polytope = sort_rounded_xyz_array(
            [[3, 3, 3], [3, 1, 3], [3, 1, 1],
             [3, 3, 1], [1, 1, 1], [1, 1, 3],
             [1, 3, 1], [1, 3, 3]])
        npt.assert_almost_equal(center_polytope, expected_polytope)
        npt.assert_almost_equal(vor.volumes[13], 8)
        npt.assert_almost_equal(np.sum(vor.volumes), box.volume)
        npt.assert_almost_equal(
            vor.nlist.weights[vor.nlist.query_point_indices == 13], 4)

    def test_voronoi_neighbors_wrapped(self):
        # Test that voronoi neighbors in the first shell are correct for a
        # wrapped 3D system, also tests multiple compute calls

        n = 10
        structure_neighbors = {
            'sc': (freud.data.UnitCell.sc, 6),
            'bcc': (freud.data.UnitCell.bcc, 14),
            'fcc': (freud.data.UnitCell.fcc, 12),
        }
        vor = freud.locality.Voronoi()

        for func, neighbors in structure_neighbors.values():
            box, points = func().generate_system(n)
            vor.compute((box, points))
            nlist = vor.nlist

            # Drop the tiny facets that come from numerical imprecision
            nlist = nlist.filter(nlist.weights > 1e-5)

            unique_indices, counts = np.unique(nlist.query_point_indices,
                                               return_counts=True)

            # Every particle should have six neighbors
            npt.assert_equal(counts, neighbors)
            npt.assert_almost_equal(np.sum(vor.volumes), box.volume)

    def test_voronoi_weights_fcc(self):
        # Test that voronoi neighbor weights are computed properly for 3D FCC

        n = 3
        box, points = freud.data.UnitCell.fcc().generate_system(n, scale=2)

        vor = freud.locality.Voronoi()
        vor.compute((box, points))
        nlist = vor.nlist

        # Drop the tiny facets that come from numerical imprecision
        nlist = nlist.filter(nlist.weights > 1e-5)

        # Every FCC particle should have 12 neighbors
        npt.assert_equal(nlist.neighbor_counts, np.full(len(points), 12))

        # Every facet area should be sqrt(2)/2
        npt.assert_allclose(nlist.weights,
                            np.full(len(nlist.weights), 0.5*np.sqrt(2)),
                            atol=1e-5)

        # Every cell should have volume 2
        vor.compute((box, points))
        npt.assert_allclose(vor.compute((box, points)).volumes,
                            np.full(len(vor.polytopes), 2.),
                            atol=1e-5)

    def test_nlist_symmetric(self):
        # Test that the voronoi neighborlist is symmetric
        L = 10  # Box length
        N = 40  # Number of particles

        box, points = freud.data.make_random_system(L, N, is2D=False)
        vor = freud.locality.Voronoi()
        vor.compute((box, points))
        nlist = vor.nlist

        ijs = set(zip(nlist.query_point_indices, nlist.point_indices))
        jis = set(zip(nlist.point_indices, nlist.query_point_indices))

        # every (i, j) pair should have a corresponding (j, i) pair
        self.assertTrue(all((j, i) in jis for (i, j) in ijs))

    def test_repr(self):
        vor = freud.locality.Voronoi()
        self.assertEqual(str(vor), str(eval(repr(vor))))

    def test_attributes(self):
        # Test that the class attributes are protected
        L = 10  # Box length
        N = 40  # Number of particles
        vor = freud.locality.Voronoi()
        with self.assertRaises(AttributeError):
            vor.nlist
        with self.assertRaises(AttributeError):
            vor.polytopes
        with self.assertRaises(AttributeError):
            vor.volumes
        box, points = freud.data.make_random_system(L, N, is2D=False)
        vor.compute((box, points))

        # Ensure attributes are accessible after calling compute
        vor.nlist
        vor.polytopes
        vor.volumes

    def test_repr_png(self):
        L = 10  # Box length
        box = freud.box.Box.square(L)
        vor = freud.locality.Voronoi()

        with self.assertRaises(AttributeError):
            vor.plot()
        self.assertEqual(vor._repr_png_(), None)

        # Make a regular grid
        points = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0]]).astype(np.float32)
        vor.compute((box, points))
        vor._repr_png_()

        L = 10  # Box length
        box = freud.box.Box.cube(L)
        vor = freud.locality.Voronoi()
        # Make a regular grid
        points = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0],
             [1, 0, 0], [1, 1, 0], [1, 2, 0],
             [2, 0, 0], [2, 1, 0], [2, 2, 0],
             [0, 0, 1], [0, 1, 1], [0, 2, 1],
             [1, 0, 1], [1, 1, 1], [1, 2, 1],
             [2, 0, 1], [2, 1, 1], [2, 2, 1],
             [0, 0, 2], [0, 1, 2], [0, 2, 2],
             [1, 0, 2], [1, 1, 2], [1, 2, 2],
             [2, 0, 2], [2, 1, 2], [2, 2, 2]]).astype(np.float32)
        vor.compute((box, points))
        self.assertEqual(vor._repr_png_(), None)


if __name__ == '__main__':
    unittest.main()
