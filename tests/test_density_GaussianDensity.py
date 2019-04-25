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
        rcut = 10.0
        sigma = 0.1
        num_points = 10000
        box_size = rcut*3.1
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        points[:, 2] = 0
        diff = freud.density.GaussianDensity(width, rcut, sigma)
        testBox = freud.box.Box.square(box_size)
        diff.compute(testBox, points)
        myDiff = diff.gaussian_density
        myFFT = fft(fft(myDiff[:, :], axis=1), axis=0)
        myDiff = (myFFT * np.conj(myFFT)).real
        myDiff = fftshift(myDiff)[:, :]
        npt.assert_equal(np.where(myDiff == np.max(myDiff)),
                         (np.array([50]), np.array([50])))

    def test_repr(self):
        diff = freud.density.GaussianDensity(100, 10.0, 0.1)
        self.assertEqual(str(diff), str(eval(repr(diff))))


if __name__ == '__main__':
    unittest.main()
