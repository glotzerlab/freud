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

    def test_center_value(self):
        """
        Assert the center value in the image is the largest for odd image size.
        """
        dp = freud.diffraction.DiffractionPattern(output_size=101)
        box, positions = freud.data.UnitCell.bcc().generate_system(10)
        dp.compute(system=(box, positions))
        pattern = dp.diffraction

        npt.assert_almost_equal(pattern[50, 50], np.max(pattern))

    def test_center_values(self):
        """
        Assert the 4 values in the center of the image are the largest for
        even image size.
        """
        dp = freud.diffraction.DiffractionPattern(output_size=100)
        box, positions = freud.data.UnitCell.bcc().generate_system(10)
        dp.compute(system=(box, positions))
        pattern = dp.diffraction
        max_val = np.max(pattern)

        npt.assert_almost_equal(pattern[50, 50], max_val)
        npt.assert_almost_equal(pattern[49, 49], max_val)
        npt.assert_almost_equal(pattern[49, 50], max_val)
        npt.assert_alomst_equal(pattern[50, 49], max_val)

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
