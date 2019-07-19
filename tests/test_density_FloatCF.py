import numpy as np
import numpy.testing as npt
import freud
import unittest


class TestFloatCF(unittest.TestCase):
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

        ocf = freud.density.FloatCF(rmax, dr)

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
        ang = np.random.random_sample((num_points)).astype(np.float64) - 0.5
        ocf = freud.density.FloatCF(rmax, dr)

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
        ang = np.random.random_sample((num_points)).astype(np.float64) - 0.5
        ocf = freud.density.FloatCF(rmax, dr)
        correct = np.zeros(int(rmax/dr), dtype=np.float64)
        absolute_tolerance = 0.1
        # first bin is bad
        ocf.accumulate(box, points, ang)
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
        ocf.compute(box, points, ang)
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
        ocf.reset()
        ocf.accumulate(box, points, ang, points,
                       ang, query_args={'exclude_ii': True})
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
        ocf.reset()
        ocf.accumulate(box, points, ang, values=ang)
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
        ocf.compute(box, points, ang)
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
        ocf = freud.density.FloatCF(rmax, dr)
        ocf.accumulate(freud.box.Box.square(box_size), points, ang)

        correct = np.zeros(int(rmax/dr), dtype=np.float32)
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

        vectors = points[np.newaxis, :, :] - points[:, np.newaxis, :]
        vector_lengths = np.array(
            [[np.linalg.norm(box.wrap(vectors[i][j]))
              for i in range(num_points)]
             for j in range(num_points)])

        # Subtract len(points) to exclude the zero i-i distances
        correct = np.sum(vector_lengths < rmax) - len(points)

        ocf = freud.density.FloatCF(rmax, dr)
        ocf.compute(freud.box.Box.square(box_size), points, ang)
        self.assertEqual(np.sum(ocf.counts), correct)

    def test_repr(self):
        ocf = freud.density.FloatCF(1000, 40)
        self.assertEqual(str(ocf), str(eval(repr(ocf))))

    def test_repr_png(self):
        rmax = 10.0
        dr = 1.0
        num_points = 1000
        box_size = rmax*3.1
        box = freud.box.Box.square(box_size)
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        ang = np.random.random_sample((num_points)).astype(np.float64) - 0.5
        ocf = freud.density.FloatCF(rmax, dr)

        with self.assertRaises(AttributeError):
            ocf.plot()
        self.assertEqual(ocf._repr_png_(), None)

        ocf.accumulate(box, points, ang)
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

        cf = freud.density.FloatCF(rmax, dr)
        cf.compute(box, ref_points, ref_values, points, values,
                   query_args={'mode': 'nearest', 'nn': 1})
        npt.assert_array_equal(cf.RDF, [1, 1, 1])
        npt.assert_array_equal(cf.counts, [2, 2, 2])

        cf.compute(box, points, values, ref_points, ref_values,
                   query_args={'mode': 'nearest', 'nn': 1})
        npt.assert_array_equal(cf.RDF, [1, 0, 0])
        npt.assert_array_equal(cf.counts, [1, 0, 0])


if __name__ == '__main__':
    unittest.main()
