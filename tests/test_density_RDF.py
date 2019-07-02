from __future__ import division

import numpy as np
import numpy.testing as npt
import freud
import unittest


class TestRDF(unittest.TestCase):
    def test_generateR(self):
        rmax = 51.23
        dr = 0.1
        for rmin in [0, 0.05, 0.1, 1.0, 3.0]:
            nbins = int((rmax - rmin) / dr)

            # make sure the radius for each bin is generated correctly
            r_list = np.zeros(nbins, dtype=np.float32)
            for i in range(nbins):
                r1 = i * dr + rmin
                r2 = r1 + dr
                r_list[i] = 2.0/3.0 * (r2**3.0 - r1**3.0) / (r2**2.0 - r1**2.0)
            rdf = freud.density.RDF(rmax, dr, rmin=rmin)
            npt.assert_allclose(rdf.R, r_list, rtol=1e-4, atol=1e-4)

    def test_attribute_access(self):
        rmax = 10.0
        dr = 1.0
        num_points = 100
        box_size = rmax*3.1
        box = freud.box.Box.square(box_size)
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        rdf = freud.density.RDF(rmax, dr)

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            rdf.RDF
        with self.assertRaises(AttributeError):
            rdf.box
        with self.assertRaises(AttributeError):
            rdf.n_r

        rdf.accumulate(box, points)

        # Test if accessible now
        rdf.RDF
        rdf.box
        rdf.n_r

        # reset
        rdf.reset()

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            rdf.RDF
        with self.assertRaises(AttributeError):
            rdf.box
        with self.assertRaises(AttributeError):
            rdf.n_r

        rdf.compute(box, points)

        # Test if accessible now
        rdf.RDF
        rdf.box
        rdf.n_r

    def test_invalid_rdf(self):
        # Make sure that invalid RDF objects raise errors
        with self.assertRaises(ValueError):
            freud.density.RDF(rmax=-1, dr=0.1)
        with self.assertRaises(ValueError):
            freud.density.RDF(rmax=1, dr=0)
        with self.assertRaises(ValueError):
            freud.density.RDF(rmax=1, dr=0.1, rmin=2)

    def test_random_point(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        tolerance = 0.1
        box_size = rmax*3.1
        np.random.seed(0)

        def ig_sphere(x, y, j):
            return 4/3*np.pi*np.trapz(y[:j]*(x[:j]+dr/2)**2, x[:j])

        for i, rmin in enumerate([0, 0.05, 0.1, 1.0, 3.0]):
            nbins = int((rmax - rmin) / dr)
            points = np.random.random_sample((num_points, 3)).astype(
                np.float32) * box_size - box_size/2

            points.flags['WRITEABLE'] = False
            rdf = freud.density.RDF(rmax, dr, rmin=rmin)
            box = freud.box.Box.cube(box_size)

            if i < 3:
                rdf.accumulate(box, points)
            else:
                rdf.compute(box, points)
            self.assertTrue(rdf.box == box)
            correct = np.ones(nbins, dtype=np.float32)
            correct[0] = 0.0
            npt.assert_allclose(rdf.RDF, correct, atol=tolerance)

            # Numerical integration to compute the running coordination number
            # will be highly inaccurate, so we can only test up to a limited
            # precision. Also, since dealing with nonzero rmin values requires
            # extrapolation, we only test when rmin=0.
            if rmin == 0:
                correct_cumulative = np.array(
                    [ig_sphere(rdf.R, rdf.RDF, j) for j in range(1, nbins+1)]
                )
                npt.assert_allclose(rdf.n_r, correct_cumulative,
                                    rtol=tolerance*5)

    def test_repr(self):
        rdf = freud.density.RDF(10, 0.1, rmin=0.5)
        self.assertEqual(str(rdf), str(eval(repr(rdf))))

    def test_repr_png(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10
        box_size = rmax*3.1
        np.random.seed(0)
        points = np.random.random_sample((num_points, 3)).astype(np.float32) \
            * box_size - box_size/2
        rdf = freud.density.RDF(rmax, dr)

        with self.assertRaises(AttributeError):
            rdf.plot()
        self.assertEqual(rdf._repr_png_(), None)

        box = freud.box.Box.cube(box_size)
        rdf.accumulate(box, points)
        rdf._repr_png_()

    def test_ref_points_ne_points(self):
        rmax = 100.0
        dr = 1
        box_size = rmax*5
        box = freud.box.Box.square(box_size)

        rdf = freud.density.RDF(rmax, dr)

        points = []
        supposed_RDF = [0]
        N = 100

        # With ref_points closely centered around the origin,
        # the cumulative average bin counts should be same as
        # having a single point at the origin.
        # Also, we can check for whether ref_points are not considered against
        # each other.
        ref_points = [[dr/4, 0, 0], [-dr/4, 0, 0], [0, dr/4, 0], [0, -dr/4, 0]]
        for r in rdf.R:
            for k in range(N):
                points.append([r * np.cos(2*np.pi*k/N),
                               r * np.sin(2*np.pi*k/N), 0])
            supposed_RDF.append(supposed_RDF[-1] + N)
        supposed_RDF = np.array(supposed_RDF[:-1])

        rdf.compute(box, ref_points, points)

        npt.assert_allclose(rdf.n_r, supposed_RDF, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
