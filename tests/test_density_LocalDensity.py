import numpy as np
import numpy.testing as npt
import freud
import unittest


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


if __name__ == '__main__':
    unittest.main()
