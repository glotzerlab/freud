import numpy
import numpy as np
import numpy.testing as npt
from freud import box, density, parallel
import unittest

class TestR(unittest.TestCase):
    def test_generateR(self):
        rmax = 51.23
        dr = 0.1
        nbins = int(rmax / dr)

        # make sure the radius for each bin is generated correctly
        r_list = np.zeros(nbins, dtype=np.float32)
        for i in range(nbins):
            r1 = i * dr
            r2 = r1 + dr
            r_list[i] = 2.0/3.0 * (r2**3.0 - r1**3.0) / (r2**2.0 - r1**2.0)

        ocf = density.ComplexCF(rmax, dr)

        npt.assert_almost_equal(ocf.getR(), r_list, decimal=3)

class TestOCF(unittest.TestCase):
    def test_random_point_with_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*3.1
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64)*np.pi*2.0
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size),points, comp, points, conj)

        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)

    def test_random_point_without_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*2
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64)*np.pi*2.0
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size),points, comp, points, conj)

        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)

    def test_value_point_with_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*3.1
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size),points, comp, points, conj)

        correct = np.ones(int(rmax/dr), dtype=np.float32) + 1j * np.zeros(int(rmax/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)

    def test_value_point_without_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*2
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size),points, comp, points, conj)

        correct = np.ones(int(rmax/dr), dtype=np.float32) + 1j * np.zeros(int(rmax/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)


def test_summation():
    # Cause the correlation function to add lots of small numbers together
    # This leads to vastly different results with different numbers of threads if the summation is not done
    # robustly
    N = 20000
    phi = numpy.zeros(N, dtype=numpy.complex128)
    phi[:] = numpy.random.rand(N)
    pos2d = numpy.array(numpy.random.random(size=(N,3)), dtype=numpy.float32)*1000 - 500
    pos2d[:,2] = 0
    fbox = box.Box.square(1000)

    # With a small number of particles, we won't get the average exactly right, so we need to check for
    # different behavior with different numbers of threads
    parallel.setNumThreads(1);
    # A very large bin size exacerbates the problem
    cf = density.ComplexCF(500, 40)
    cf.compute(fbox, pos2d, phi, pos2d, phi)
    c1 = cf.getCounts();
    f1 = numpy.real(cf.getRDF());

    parallel.setNumThreads(20);
    cf.compute(fbox, pos2d, phi, pos2d, phi)
    c2 = cf.getCounts();
    f2 = numpy.real(cf.getRDF());

    numpy.testing.assert_allclose(f1, f2);
    numpy.testing.assert_array_equal(c1, c2);

if __name__ == '__main__':
    unittest.main()
