import freud
import matplotlib
import unittest
import numpy as np
import numpy.testing as npt
matplotlib.use('agg')


class TestDiffractionPattern(unittest.TestCase):
    def test_compute(self):
        dp = freud.diffraction.DiffractionPattern()
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

    def test_attribute_access(self):
        dp = freud.diffraction.DiffractionPattern()
        box, positions = freud.data.UnitCell.fcc().generate_system(4)

        with self.assertRaises(AttributeError):
            dp.diffraction
        with self.assertRaises(AttributeError):
            dp.k_vectors
        with self.assertRaises(AttributeError):
            dp.plot()

        dp.compute((box, positions), zoom=1, peak_width=4)
        dp.diffraction
        dp.k_vectors
        dp.plot()
        dp._repr_png_()

    def test_center_unordered(self):
        """
        Assert the center of the image is an intensity peak for an unordered
        system.
        """
        dp = freud.diffraction.DiffractionPattern(output_size=99)
        box, positions = freud.data.make_random_system(10, 100)
        dp.compute(system=(box, positions))
        pattern = np.asarray(dp.diffraction)

        # make sure the pixel at the center is part of a peak at the origin,
        # meaning its value is of the same order of magnitude as the max value
        self.assertTrue(pattern[49, 49] > .1 * np.max(pattern))

        # assert the group of pixels in the center has the highest intensity
        group_sum = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                indices = [(m, n) for m in range(11*i, 11*(i+1))
                                  for n in range(11*j, 11*(j+1))]
                group_sum[i, j] = pattern[indices].sum()

        npt.assert_almost_equal(group_sum[4, 4], np.max(group_sum))

    def test_center_ordered(self):
        """
        Assert the center of the image is an intensity peak for an ordered
        system.
        """
        dp = freud.diffraction.DiffractionPattern(output_size=100)
        box, positions = freud.data.UnitCell.bcc().generate_system(10)
        dp.compute(system=(box, positions))
        pattern = dp.diffraction
        max_val = np.max(pattern)

        # similar assertion as the above test
        self.assertTrue(pattern[50, 50] > .1 * max_val)
        self.assertTrue(pattern[49, 49] > .1 * max_val)
        self.assertTrue(pattern[49, 50] > .1 * max_val)
        self.assertTrue(pattern[50, 49] > .1 * max_val)

    def test_zero_particles(self):
        """Assert that all values are equal if there are no particles."""
        dp = freud.diffraction.DiffractionPattern()
        box = freud.box.Box(10, 10, 10)

        # trying to get freud to accept an empty box
        dp.compute(system=(box, None))
        pattern = dp.diffraction

        npt.testing.assert_almost_equal(np.max(pattern), np.min(pattern))

    def test_repr(self):
        dp = freud.diffraction.DiffractionPattern()
        self.assertEqual(str(dp), str(eval(repr(dp))))

        # Use non-default arguments for all parameters
        dp = freud.diffraction.DiffractionPattern(
            grid_size=500)
        self.assertEqual(str(dp), str(eval(repr(dp))))

    def test_k_vector(self):
        dp = freud.diffraction.DiffractionPattern()
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

        default_size = dp.diffraction.shape[0]
        default_shape = (default_size, default_size, 3)
        npt.assert_equal(dp.k_vectors.shape, default_shape)


if __name__ == '__main__':
    unittest.main()
