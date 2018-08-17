import numpy as np
import numpy.testing as npt
from freud import box, density, parallel
import unittest


class TestCorrelationFunction(unittest.TestCase):
    def test_type_check(self):
        boxlen = 10
        N = 500
        rmax, dr = 3, 0.1

        bx = box.Box.cube(boxlen)

        np.random.seed(0)
        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N, 3)),
                            dtype=np.float32)
        values = np.ones((N,)) + 0j
        corrfun = density.ComplexCF(rmax, dr)

        values = np.asarray(values, dtype=np.complex128)
        corrfun.compute(bx, points, values, points, values.conj())
        assert True


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

        npt.assert_almost_equal(ocf.R, r_list, decimal=3)


class TestOCF(unittest.TestCase):

    def test_random_point_with_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*3.1
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size), points, comp, points, conj)

        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)

    @unittest.skip("Skipping to test with CircleCI")
    def test_random_point_without_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*2
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size), points, comp, points, conj)

        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        npt.assert_allclose(ocf.rdf, correct, atol=absolute_tolerance)

    def test_value_point_with_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*3.1
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size), points, comp, points, conj)

        correct = np.ones(int(rmax/dr), dtype=np.float32) + \
            1j * np.zeros(int(rmax/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)

    @unittest.skip("Skipping to test with CircleCI")
    def test_value_point_without_cell_list(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*2
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.cos(ang) + 1j * np.sin(ang)
        conj = np.cos(ang) - 1j * np.sin(ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size), points, comp, points, conj)

        correct = np.ones(int(rmax/dr), dtype=np.float32) + \
            1j * np.zeros(int(rmax/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.rdf, correct, atol=absolute_tolerance)


class TestSummation(unittest.TestCase):
    @unittest.skip("Skipping to test with CircleCI")
    def test_summation():
        # Causes the correlation function to add lots of small numbers together
        # This leads to vastly different results with different numbers of
        # threads if the summation is not done robustly
        N = 20000
        phi = np.zeros(N, dtype=np.complex128)
        np.random.seed(0)
        phi[:] = np.random.rand(N)
        pos2d = np.array(np.random.random(size=(N, 3)), dtype=np.float32) \
            * 1000 - 500
        pos2d[:, 2] = 0
        fbox = box.Box.square(1000)

        # With a small number of particles, we won't get the average exactly
        # right, so we check for different behavior with different numbers of
        # threads
        parallel.setNumThreads(1)
        # A very large bin size exacerbates the problem
        cf = density.ComplexCF(500, 40)
        cf.compute(fbox, pos2d, phi, pos2d, phi)
        c1 = cf.counts
        f1 = np.real(cf.rdf)

        parallel.setNumThreads(20)
        cf.compute(fbox, pos2d, phi, pos2d, phi)
        c2 = cf.counts
        f2 = np.real(cf.rdf)

        np.testing.assert_allclose(f1, f2)
        np.testing.assert_array_equal(c1, c2)


if __name__ == '__main__':
    unittest.main()
