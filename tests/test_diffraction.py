import freud
import matplotlib
import unittest
import numpy as np
import numpy.testing as npt
import rowan
matplotlib.use('agg')


class TestDiffractionPattern(unittest.TestCase):
    def test_compute(self):
        dp = freud.diffraction.DiffractionPattern()
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

    def test_attribute_access(self):
        grid_size = 234
        output_size = 123
        dp = freud.diffraction.DiffractionPattern(grid_size=grid_size)
        self.assertEqual(dp.grid_size, grid_size)
        self.assertEqual(dp.output_size, grid_size)
        dp = freud.diffraction.DiffractionPattern(
            grid_size=grid_size, output_size=output_size)
        self.assertEqual(dp.grid_size, grid_size)
        self.assertEqual(dp.output_size, output_size)

        box, positions = freud.data.UnitCell.fcc().generate_system(4)

        with self.assertRaises(AttributeError):
            dp.diffraction
        with self.assertRaises(AttributeError):
            dp.k_values
        with self.assertRaises(AttributeError):
            dp.k_vectors
        with self.assertRaises(AttributeError):
            dp.plot()

        dp.compute((box, positions), zoom=1, peak_width=4)
        diff = dp.diffraction
        vals = dp.k_values
        vecs = dp.k_vectors
        dp.plot()
        dp._repr_png_()

        # Make sure old data is not invalidated by new call to compute()
        box2, positions2 = freud.data.UnitCell.bcc().generate_system(3)
        dp.compute((box2, positions2), zoom=1, peak_width=4)
        self.assertFalse(np.array_equal(dp.diffraction, diff))
        self.assertFalse(np.array_equal(dp.k_values, vals))
        self.assertFalse(np.array_equal(dp.k_vectors, vecs))

    def test_attribute_shapes(self):
        grid_size = 234
        output_size = 123
        dp = freud.diffraction.DiffractionPattern(
            grid_size=grid_size, output_size=output_size)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

        self.assertEqual(dp.diffraction.shape, (output_size, output_size))
        self.assertEqual(dp.k_values.shape, (output_size,))
        self.assertEqual(dp.k_vectors.shape, (output_size, output_size, 3))
        self.assertEqual(dp.to_image().shape, (output_size, output_size, 4))

    def test_center_unordered(self):
        """Assert the center of the image is an intensity peak for an
        unordered system.
        """
        box, positions = freud.data.make_random_system(
            box_size=10, num_points=1000)

        # Test different parities (odd/even)
        for grid_size in [255, 256]:
            for output_size in [255, 256]:
                dp = freud.diffraction.DiffractionPattern(
                    grid_size=grid_size, output_size=output_size)

                # Use a random view orientation and a random zoom
                for view_orientation in rowan.random.rand(100):
                    zoom = 1 + 10*np.random.rand()
                    dp.compute(
                        system=(box, positions),
                        view_orientation=view_orientation,
                        zoom=zoom)

                    # The pixel at the center (k=0) is the maximum value
                    diff = dp.diffraction
                    max_index = np.unravel_index(np.argmax(diff), diff.shape)
                    self.assertEqual(
                        max_index, (output_size//2, output_size//2))

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

    def test_one_particle(self):
        """Assert that all values are close to one with only one point."""
        dp = freud.diffraction.DiffractionPattern(output_size=101)
        box = freud.Box.cube(100)
        points = [[0., 0., 0.]]

        dp.compute(system=(box, points))
        pattern = dp.diffraction

        # pattern should basically be 1 everywhere
        self.assertTrue(np.min(pattern) > .95)

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
