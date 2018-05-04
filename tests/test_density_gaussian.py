import numpy as np
import numpy.testing as npt
from scipy.fftpack import fft, fftshift
# need this if run from root. If run in tests you're fine. Not sure how this will impact jenkins
# import sys
# sys.path = sys.path[::-1]
from freud import box, density
import unittest

class TestDensity(unittest.TestCase):
    def test_random_point_with_cell_list(self):
        width = 100
        rcut = 10.0
        sigma=0.1
        num_points = 10000
        box_size = rcut*3.1
        np.random.seed(0)
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        points[:,2] = 0
        diff = density.GaussianDensity(width, rcut, sigma)
        testBox = box.Box.square(box_size)
        diff.compute(testBox, points)
        myDiff = diff.getGaussianDensity()
        myFFT = fft(fft(myDiff[:,:], axis=1), axis=0)
        myDiff = (myFFT * np.conj(myFFT)).real
        myDiff = fftshift(myDiff)[:,:]
        npt.assert_equal(np.where(myDiff==np.max(myDiff)), (np.array([50]), np.array([50])))

if __name__ == '__main__':
    unittest.main()
