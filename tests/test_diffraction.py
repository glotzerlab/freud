import freud
import unittest
import numpy.testing as npt


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

        dp.compute((box, positions))
        dp.diffraction
        dp.k_vectors
        dp.plot()
        dp._repr_png_()

    def test_repr(self):
        dp = freud.diffraction.DiffractionPattern()
        self.assertEqual(str(dp), str(eval(repr(dp))))

        # Use non-default arguments for all parameters
        dp = freud.diffraction.DiffractionPattern(
            grid_size=500, zoom=1, peak_width=2)
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
