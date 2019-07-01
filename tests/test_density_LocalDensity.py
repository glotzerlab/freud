import numpy as np
import numpy.testing as npt
import freud
import unittest


def getFraction(dist, rcut, diameter):
    if dist < rcut - diameter/2:
        return 1
    if dist > rcut + diameter/2:
        return 0
    else:
        return -dist/diameter + rcut/diameter + 0.5


class TestLD(unittest.TestCase):
    """Test fixture for LocalDensity"""

    def setUp(self):
        """Initialize a box with randomly placed particles"""
        self.box = freud.box.Box.cube(10)
        np.random.seed(0)
        self.pos = np.array(np.random.random(size=(10000, 3)),
                            dtype=np.float32) * 10 - 5
        self.ld = freud.density.LocalDensity(3, 1, 1)

        # Test access
        with self.assertRaises(AttributeError):
            self.ld.density
        with self.assertRaises(AttributeError):
            self.ld.num_neighbors
        with self.assertRaises(AttributeError):
            self.ld.box

    @unittest.skip('Skipping slow LocalDensity test.')
    def test_compute_api(self):
        # test 2 args, no keyword
        self.ld.compute(self.box, self.pos)
        # test 3 args, no keyword
        self.ld.compute(self.box, self.pos, self.pos)
        # test 2 args, keyword
        self.ld.compute(box=self.box, ref_points=self.pos)
        # test 3 args, keyword
        self.ld.compute(box=self.box, ref_points=self.pos, points=self.pos)

    def test_density(self):
        """Test that LocalDensity computes the correct density at each point"""
        self.ld.compute(self.box, self.pos, self.pos)

        # Test access
        self.ld.density
        self.ld.num_neighbors
        self.ld.box

        self.assertTrue(self.ld.box == freud.box.Box.cube(10))

        npt.assert_array_less(np.fabs(self.ld.density - 10.0), 1.5)

        npt.assert_array_less(
            np.fabs(self.ld.num_neighbors - 1130.973355292), 200)

    @unittest.skip('Skipping slow LocalDensity test.')
    def test_refpoints(self):
        """Test that LocalDensity can compute a correct density at each point
        using the reference points as the data points."""
        self.ld.compute(self.box, self.pos)
        density = self.ld.density

        npt.assert_array_less(np.fabs(density - 10.0), 1.5)

        neighbors = self.ld.num_neighbors
        npt.assert_array_less(np.fabs(neighbors - 1130.973355292), 200)

    def test_repr(self):
        self.assertEqual(str(self.ld), str(eval(repr(self.ld))))

    def test_ref_point_ne_point(self):
        box = freud.box.Box.cube(10)
        ref_points = np.array([[0, 0, 0], [1, 0, 0]])
        points = np.array([[0, 1, 0], [-1, -1, 0]])
        volume = 1
        diameter = 1
        rcut = 2

        v_around = 4/3 * (rcut**3) * np.pi

        ld = freud.density.LocalDensity(rcut, volume, diameter)
        ld.compute(box, ref_points, points)

        cd0 = 2/v_around
        cd1 = (1 + getFraction(np.sqrt(5), rcut, diameter))/v_around
        correct_density = [cd0, cd1]

        npt.assert_allclose(ld.density, correct_density, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
