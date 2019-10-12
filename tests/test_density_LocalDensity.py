import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


def get_fraction(dist, r_max, diameter):
    """Compute what fraction of a point of the provided diameter at distance
    dist is contained in a sphere of radius r_max."""
    if dist < r_max - diameter/2:
        return 1
    if dist > r_max + diameter/2:
        return 0
    else:
        return -dist/diameter + r_max/diameter + 0.5


class TestLD(unittest.TestCase):
    """Test fixture for LocalDensity"""

    def setUp(self):
        """Initialize a box with randomly placed particles"""
        box_size = 10
        num_points = 10000
        self.box, self.pos = util.make_box_and_random_points(
            box_size, num_points)
        self.r_max = 3
        self.diameter = 1
        self.ld = freud.density.LocalDensity(self.r_max, self.diameter)

        # Test access
        with self.assertRaises(AttributeError):
            self.ld.density
        with self.assertRaises(AttributeError):
            self.ld.num_neighbors
        with self.assertRaises(AttributeError):
            self.ld.box

    def test_density(self):
        """Test that LocalDensity computes the correct density at each point"""

        r_max = self.r_max + 0.5*self.diameter
        test_set = util.make_raw_query_nlist_test_set(
            self.box, self.pos, self.pos, "ball", r_max, 0, True)
        for nq, neighbors in test_set:
            self.ld.compute(nq, neighbors=neighbors)

            # Test access
            self.ld.density
            self.ld.num_neighbors
            self.ld.box

            self.assertTrue(self.ld.box == freud.box.Box.cube(10))

            npt.assert_array_less(np.fabs(self.ld.density - 10.0), 1.5)

            npt.assert_array_less(
                np.fabs(self.ld.num_neighbors - 1130.973355292), 200)

    def test_ref_points(self):
        """Test that LocalDensity can compute a correct density at each point
        using the reference points as the data points."""
        self.ld.compute((self.box, self.pos))
        density = self.ld.density

        npt.assert_array_less(np.fabs(density - 10.0), 1.5)

        neighbors = self.ld.num_neighbors
        npt.assert_array_less(np.fabs(neighbors - 1130.973355292), 200)

    def test_repr(self):
        self.assertEqual(str(self.ld), str(eval(repr(self.ld))))

    def test_points_ne_query_points(self):
        box = freud.box.Box.cube(10)
        points = np.array([[0, 0, 0], [1, 0, 0]])
        query_points = np.array([[0, 1, 0], [-1, -1, 0], [4, 0, 0]])
        diameter = 1
        r_max = 2

        v_around = 4/3 * (r_max**3) * np.pi

        ld = freud.density.LocalDensity(r_max, diameter)
        ld.compute((box, points), query_points)

        cd0 = 2/v_around
        cd1 = (1 + get_fraction(np.linalg.norm(points[1] - query_points[1]),
                                r_max, diameter)) / v_around
        correct_density = [cd0, cd1, 0]
        npt.assert_allclose(ld.density, correct_density, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
