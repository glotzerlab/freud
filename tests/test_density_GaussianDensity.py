import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestDensity(unittest.TestCase):
    @util.skipIfMissing('scipy.fftpack')
    def test_random_point_with_cell_list(self):
        from scipy.fftpack import fft, fftshift
        width = 100
        r_max = 10.0
        sigma = 0.1
        num_points = 10000
        box_size = r_max*3.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True)
        for w in (width, (width, width), [width, width]):
            diff = freud.density.GaussianDensity(w, r_max, sigma)

            # Test access
            with self.assertRaises(AttributeError):
                diff.box
            with self.assertRaises(AttributeError):
                diff.density

            diff.compute((box, points))

            # Test access
            diff.box
            diff.density

            myDiff = diff.density
            myFFT = fft(fft(myDiff[:, :], axis=1), axis=0)
            myDiff = (myFFT * np.conj(myFFT)).real
            myDiff = fftshift(myDiff)[:, :]
            npt.assert_equal(np.where(myDiff == np.max(myDiff)),
                             (np.array([50]), np.array([50])))

    def test_change_box_dimension(self):
        width = 100
        r_max = 10.0
        sigma = 0.1
        num_points = 100
        box_size = r_max*3.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True)
        diff = freud.density.GaussianDensity(width, r_max, sigma)

        diff.compute((box, points))

        test_box = freud.box.Box.cube(box_size)
        diff.compute((test_box, points))

    def test_repr(self):
        diff = freud.density.GaussianDensity(100, 10.0, 0.1)
        self.assertEqual(str(diff), str(eval(repr(diff))))

        # Use both signatures
        diff3 = freud.density.GaussianDensity((98, 99, 100), 10.0, 0.1)
        self.assertEqual(str(diff3), str(eval(repr(diff3))))

    def test_repr_png(self):
        width = 100
        r_max = 10.0
        sigma = 0.1
        num_points = 100
        box_size = r_max*3.1
        box, points = freud.data.make_random_system(
            box_size, num_points, is2D=True)
        diff = freud.density.GaussianDensity(width, r_max, sigma)

        with self.assertRaises(AttributeError):
            diff.plot()
        self.assertEqual(diff._repr_png_(), None)

        diff.compute((box, points))
        diff.plot()

        diff = freud.density.GaussianDensity(width, r_max, sigma)
        test_box = freud.box.Box.cube(box_size)
        diff.compute((test_box, points))
        diff.plot()
        self.assertEqual(diff._repr_png_(), None)


if __name__ == '__main__':
    unittest.main()
