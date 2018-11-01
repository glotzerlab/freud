import numpy as np
import numpy.testing as npt
from freud import box, density, parallel
import unittest
from freud.errors import FreudDeprecationWarning
import warnings
import os


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
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_random_points(self):
        rmax = 10.0
        dr = 1.0
        num_points = 1000
        box_size = rmax*3.1
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        comp = np.exp(1j*ang)
        ocf = density.ComplexCF(rmax, dr)
        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        ocf.accumulate(box.Box.square(box_size), points, comp,
                       points, np.conj(comp))
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
        ocf.compute(box.Box.square(box_size), points, comp,
                    points, np.conj(comp))
        npt.assert_allclose(ocf.getRDF(), correct, atol=absolute_tolerance)
        self.assertEqual(box.Box.square(box_size), ocf.box)
        self.assertEqual(box.Box.square(box_size), ocf.getBox())

    def test_zero_points(self):
        rmax = 10.0
        dr = 1.0
        num_points = 1000
        box_size = rmax*3.1
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.exp(1j*ang)
        ocf = density.ComplexCF(rmax, dr)
        ocf.accumulate(box.Box.square(box_size), points, comp,
                       points, np.conj(comp))

        correct = np.ones(int(rmax/dr), dtype=np.float32) + \
            1j * np.zeros(int(rmax/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)

    def test_counts(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10
        box_size = rmax*2.1
        fbox = box.Box.square(box_size)
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(
            np.float32) * box_size - box_size/2
        points[:, 2] = 0
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.exp(1j*ang)

        vectors = points[np.newaxis, :, :] - points[:, np.newaxis, :]
        vector_lengths = np.array(
            [[np.linalg.norm(fbox.wrap(vectors[i][j]))
              for i in range(num_points)]
             for j in range(num_points)])

        # Subtract len(points) to exclude the zero i-i distances
        correct = np.sum(vector_lengths < rmax) - len(points)

        ocf = density.FloatCF(rmax, dr)
        ocf.compute(box.Box.square(box_size), points, comp,
                    points, np.conj(comp))
        self.assertEqual(np.sum(ocf.getCounts()), correct)
        self.assertEqual(np.sum(ocf.counts), correct)


class TestSummation(unittest.TestCase):
    @unittest.skipIf('CI' in os.environ, 'Skipping test on CI')
    def test_summation(self):
        # Causes the correlation function to add lots of small numbers together
        # This leads to vastly different results with different numbers of
        # threads if the summation is not done robustly
        N = 20000
        L = 1000
        np.random.seed(0)
        phi = np.random.rand(N)
        pos2d = np.random.uniform(-L/2, L/2, size=(N, 3))
        pos2d[:, 2] = 0
        fbox = box.Box.square(L)

        # With a small number of particles, we won't get the average exactly
        # right, so we check for different behavior with different numbers of
        # threads
        parallel.setNumThreads(1)
        # A very large bin size exacerbates the problem
        cf = density.ComplexCF(L/2.1, 40)
        cf.compute(fbox, pos2d, phi)
        c1 = cf.counts
        f1 = np.real(cf.RDF)

        parallel.setNumThreads(20)
        cf.compute(fbox, pos2d, phi)
        c2 = cf.counts
        f2 = np.real(cf.RDF)

        npt.assert_allclose(f1, f2)
        npt.assert_array_equal(c1, c2)


if __name__ == '__main__':
    unittest.main()
