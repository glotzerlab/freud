import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


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
        box = freud.box.Box.square(box_size)
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        comp = np.exp(1j*ang)
        correct = np.zeros(int(rmax/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        test_set = util.makeRawQueryNlistTestSet(
            box, points, points, 'ball', rmax, 0, True)
        for ts in test_set:
            ocf = freud.density.ComplexCF(rmax, dr)
            ocf.accumulate(box, ts[0], comp, points, np.conj(comp),
                           query_args={"exclude_ii": True}, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.compute(box, ts[0], comp, points, np.conj(comp),
                        query_args={"exclude_ii": True}, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            self.assertEqual(box, ocf.box)

            ocf.reset()
            ocf.accumulate(box, ts[0], comp, values=np.conj(comp), nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.compute(box, ts[0], comp, values=np.conj(comp), nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)

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
                       points, np.conj(comp), query_args={"exclude_ii": True})

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
        ocf = freud.density.ComplexCF(rmax, dr)
        ocf.compute(freud.box.Box.square(box_size), points, comp,
                    points, np.conj(comp), query_args={"exclude_ii": True})
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
                       points, np.conj(comp), query_args={"exclude_ii": True})
        ocf._repr_png_()

    def test_query_nn(self):
        """Test nearest-neighbor-based querying."""
        box_size = 8
        rmax = 3
        dr = 1
        box = freud.box.Box.cube(box_size)
        ref_points = np.array([[0, 0, 0]],
                              dtype=np.float32)
        points = np.array([[0.4, 0.0, 0.0],
                           [0.9, 0.0, 0.0],
                           [0.0, 1.4, 0.0],
                           [0.0, 1.9, 0.0],
                           [0.0, 0.0, 2.4],
                           [0.0, 0.0, 2.9]],
                          dtype=np.float32)
        ref_values = np.ones(ref_points.shape[0])
        values = np.ones(points.shape[0])

        cf = freud.density.ComplexCF(rmax, dr)
        cf.compute(box, ref_points, ref_values, points, values,
                   query_args={'mode': 'nearest', 'nn': 1})
        npt.assert_array_equal(cf.RDF, [1, 1, 1])
        npt.assert_array_equal(cf.counts, [2, 2, 2])

        cf.compute(box, points, values, ref_points, ref_values,
                   query_args={'mode': 'nearest', 'nn': 1})
        npt.assert_array_equal(cf.RDF, [1, 0, 0])
        npt.assert_array_equal(cf.counts, [1, 0, 0])

        ref_values = [1+1j]
        values = [1+1j, 1+1j, 2+2j, 2+2j, 3+3j, 3+3j]
        cf.compute(box, ref_points, ref_values, points, np.conj(values),
                   query_args={'mode': 'nearest', 'nn': 1})
        npt.assert_array_equal(cf.RDF, [2, 4, 6])
        npt.assert_array_equal(cf.counts, [2, 2, 2])

        cf.compute(box, ref_points, ref_values, points, values,
                   query_args={'mode': 'nearest', 'nn': 1})
        npt.assert_array_equal(cf.RDF, [2j, 4j, 6j])
        npt.assert_array_equal(cf.counts, [2, 2, 2])

        cf.compute(box, points, values, ref_points, np.conj(ref_values),
                   query_args={'mode': 'nearest', 'nn': 1})
        npt.assert_array_equal(cf.RDF, [2, 0, 0])
        npt.assert_array_equal(cf.counts, [1, 0, 0])

        cf.compute(box, points, values, ref_points, ref_values,
                   query_args={'mode': 'nearest', 'nn': 1})
        npt.assert_array_equal(cf.RDF, [2j, 0, 0])
        npt.assert_array_equal(cf.counts, [1, 0, 0])


if __name__ == '__main__':
    unittest.main()
