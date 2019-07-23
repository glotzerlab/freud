import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


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
        box, points = util.makeBoxAndRandomPoints(box_size, num_points, True)
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
        box, points = util.makeBoxAndRandomPoints(box_size, num_points, True)
        ang = np.random.random_sample((num_points)).astype(np.float64) - 0.5
        correct = np.zeros(int(rmax/dr), dtype=np.float64)
        absolute_tolerance = 0.1
        # first bin is bad

        test_set = util.makeRawQueryNlistTestSet(
            box, points, points, 'ball', rmax, 0, True)
        for ts in test_set:
            ocf = freud.density.FloatCF(rmax, dr)
            ocf.accumulate(box, ts[0], ang, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.compute(box, ts[0], ang, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.reset()
            ocf.accumulate(box, ts[0], ang, points,
                           ang, qargs={'exclude_ii': True}, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.reset()
            ocf.accumulate(box, ts[0], ang, values=ang, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.compute(box, ts[0], ang, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            self.assertEqual(freud.box.Box.square(box_size), ocf.box)

    def test_zero_points(self):
        rmax = 10.0
        dr = 1.0
        num_points = 1000
        box_size = rmax*3.1
        box, points = util.makeBoxAndRandomPoints(box_size, num_points, True)
        ang = np.zeros(int(num_points), dtype=np.float64)
        ocf = freud.density.FloatCF(rmax, dr)
        ocf.accumulate(box, points, ang)

        correct = np.zeros(int(rmax/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)

    def test_counts(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10
        box_size = rmax*2.1
        box, points = util.makeBoxAndRandomPoints(box_size, num_points, True)
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
        box, points = util.makeBoxAndRandomPoints(box_size, num_points, True)
        ang = np.random.random_sample((num_points)).astype(np.float64) - 0.5
        ocf = freud.density.FloatCF(rmax, dr)

        with self.assertRaises(AttributeError):
            ocf.plot()
        self.assertEqual(ocf._repr_png_(), None)

        ocf.accumulate(box, points, ang)
        ocf._repr_png_()

    def test_ref_points_ne_points(self):
        def value_func(_r):
            return np.sin(_r)

        rmax = 10.0
        dr = 0.1
        box_size = rmax*5
        box = freud.box.Box.square(box_size)

        ocf = freud.density.FloatCF(rmax, dr)

        points = []
        values = []
        supposed_RDF = []
        N = 300

        # We are generating the values so that they are sine wave from 0 to 2pi
        # rotated around z axis.
        # Therefore, the RDF should be a scalar multiple sin if we set our
        # ref_point to be in the origin.
        for r in ocf.R:
            for k in range(N):
                points.append([r * np.cos(2*np.pi*k/N),
                               r * np.sin(2*np.pi*k/N), 0])
                values.append(value_func(r))
            supposed_RDF.append(value_func(r))

        supposed_RDF = np.array(supposed_RDF)

        # ref_points are within distances closer than dr, so their impact on
        # the result should be minimal.
        ref_points = [[dr/4, 0, 0], [-dr/4, 0, 0], [0, dr/4, 0], [0, -dr/4, 0]]

        test_set = util.makeRawQueryNlistTestSet(
            box, ref_points, points, "ball", rmax, 0, False)
        for ts in test_set:
            ocf = freud.density.FloatCF(rmax, dr)
            # try for different scalar values.
            for rv in [0, 1, 2, 7]:
                ref_values = [rv] * 4

                ocf.compute(
                    box, ts[0], ref_values, points, values, nlist=ts[1])
                correct = supposed_RDF * rv

                npt.assert_allclose(ocf.RDF, correct, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
