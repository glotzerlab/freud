import numpy as np
import numpy.testing as npt
import freud
import unittest


class TestComplexCF(unittest.TestCase):
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

        ocf = freud.density.ComplexCF(rmax, dr)

        npt.assert_allclose(ocf.R, r_list, atol=1e-3)

    def test_attribute_access(self):
        rmax = 10.0
        dr = 1.0
        num_points = 100
        box_size = rmax*3.1
        box = freud.box.Box.square(box_size)
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        ocf = freud.density.ComplexCF(rmax, dr)

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            ocf.RDF
        with self.assertRaises(AttributeError):
            ocf.box
        with self.assertRaises(AttributeError):
            ocf.counts

        ocf.accumulate(box, points, ang)

        # Test if accessible now
        ocf.RDF
        ocf.box
        ocf.counts

        # reset
        ocf.reset()

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            ocf.RDF
        with self.assertRaises(AttributeError):
            ocf.box
        with self.assertRaises(AttributeError):
            ocf.counts

        ocf.compute(box, points, ang)

        # Test if accessible now
        ocf.RDF
        ocf.box
        ocf.counts

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
        ocf = freud.density.ComplexCF(rmax, dr)
        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        ocf.accumulate(freud.box.Box.square(box_size), points, comp,
                       points, np.conj(comp))
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
        ocf.compute(freud.box.Box.square(box_size), points, comp,
                    points, np.conj(comp))
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
        self.assertEqual(freud.box.Box.square(box_size), ocf.box)

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
        ocf = freud.density.ComplexCF(rmax, dr)
        ocf.accumulate(freud.box.Box.square(box_size), points, comp,
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
        box = freud.box.Box.square(box_size)
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(
            np.float32) * box_size - box_size/2
        points[:, 2] = 0
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.exp(1j*ang)

        vectors = points[np.newaxis, :, :] - points[:, np.newaxis, :]
        vector_lengths = np.array(
            [[np.linalg.norm(box.wrap(vectors[i][j]))
              for i in range(num_points)]
             for j in range(num_points)])

        # Subtract len(points) to exclude the zero i-i distances
        correct = np.sum(vector_lengths < rmax) - len(points)

        ocf = freud.density.FloatCF(rmax, dr)
        ocf.compute(freud.box.Box.square(box_size), points, comp,
                    points, np.conj(comp))
        self.assertEqual(np.sum(ocf.counts), correct)

    @unittest.skip('Skipping slow summation test.')
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
        box = freud.box.Box.square(L)

        # With a small number of particles, we won't get the average exactly
        # right, so we check for different behavior with different numbers of
        # threads
        freud.parallel.setNumThreads(1)
        # A very large bin size exacerbates the problem
        cf = freud.density.ComplexCF(L/2.1, 40)
        cf.compute(box, pos2d, phi)
        c1 = cf.counts
        f1 = np.real(cf.RDF)

        freud.parallel.setNumThreads(20)
        cf.compute(box, pos2d, phi)
        c2 = cf.counts
        f2 = np.real(cf.RDF)

        npt.assert_allclose(f1, f2)
        npt.assert_array_equal(c1, c2)

    def test_repr(self):
        cf = freud.density.ComplexCF(1000, 40)
        self.assertEqual(str(cf), str(eval(repr(cf))))

    def test_repr_png(self):
        rmax = 10.0
        dr = 1.0
        num_points = 100
        box_size = rmax*3.1
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        comp = np.exp(1j*ang)
        ocf = freud.density.ComplexCF(rmax, dr)

        with self.assertRaises(AttributeError):
            ocf.plot()
        self.assertEqual(ocf._repr_png_(), None)

        ocf.accumulate(freud.box.Box.square(box_size), points, comp,
                       points, np.conj(comp))
        ocf._repr_png_()

    def test_ref_points_ne_points(self):
        # Helper function to give complex number representation of a point
        def value_func(_p):
            return _p[0] + 1j*_p[1]

        rmax = 10.0
        dr = 0.1
        box_size = rmax*5
        box = freud.box.Box.square(box_size)

        ocf = freud.density.ComplexCF(rmax, dr)

        points = []
        values = []
        supposed_RDF = []
        N = 300

        # We are essentially generating all n-th roots of unities
        # scalar multiplied by the each bin centers
        # with the value of a point being its complex number representation.
        # Therefore, the RDF should be uniformly zero
        # since the roots of unities add up to zero, if we set our ref_point in
        # the origin.
        # Nice proof for this fact is that when the n-th roots of unities
        # are viewed as vectors, we can draw a regular n-gon
        # so that we start at the origin and come back to origin.
        for r in ocf.R:
            for k in range(N):
                point = [r * np.cos(2*np.pi*k/N), r * np.sin(2*np.pi*k/N), 0]
                points.append(point)
                values.append(value_func(point))

        supposed_RDF = np.zeros(ocf.R.shape)

        ref_points = [[0, 0, 0]]
        for rv in [0, 1, 2, 7]:
            ref_values = [rv]

            ocf.compute(box, ref_points, ref_values, points, values)
            correct = supposed_RDF

            npt.assert_allclose(ocf.RDF, correct, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
