from __future__ import division

import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestRDF(unittest.TestCase):
    def test_generateR(self):
        r_max = 5
        for r_min in [0, 0.05, 0.1, 1.0, 3.0]:
            bins = round((r_max-r_min)/0.1)
            dr = (r_max - r_min) / bins

            # make sure the radius for each bin is generated correctly
            r_list = np.array([r_min + dr*(i+1/2) for i in range(bins) if
                               r_min + dr*(i+1/2) < r_max])
            rdf = freud.density.RDF(bins, r_max, r_min=r_min)
            npt.assert_allclose(rdf.bin_centers, r_list, rtol=1e-4, atol=1e-4)
            npt.assert_allclose((rdf.bin_edges+dr/2)[:-1], r_list, rtol=1e-4,
                                atol=1e-4)

    def test_attribute_access(self):
        r_max = 10.0
        bins = 10
        num_points = 100
        box_size = r_max*3.1
        box, points = util.make_box_and_random_points(
            box_size, num_points, True)
        rdf = freud.density.RDF(r_max=r_max, bins=bins)

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            rdf.RDF
        with self.assertRaises(AttributeError):
            rdf.box
        with self.assertRaises(AttributeError):
            rdf.n_r

        rdf.accumulate((box, points))

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

        rdf.compute((box, points))

        # Test if accessible now
        rdf.RDF
        rdf.box
        rdf.n_r

    def test_invalid_rdf(self):
        # Make sure that invalid RDF objects raise errors
        with self.assertRaises(ValueError):
            freud.density.RDF(r_max=-1, bins=10)
        with self.assertRaises(ValueError):
            freud.density.RDF(r_max=1, bins=0)
        with self.assertRaises(ValueError):
            freud.density.RDF(r_max=1, bins=10, r_min=2)

    def test_random_point(self):
        r_max = 10.0
        bins = 10
        num_points = 10000
        tolerance = 0.1
        box_size = r_max*3.1

        for i, r_min in enumerate([0, 0.05, 0.1, 1.0, 3.0]):
            box, points = util.make_box_and_random_points(box_size, num_points)
            test_set = util.make_raw_query_nlist_test_set(
                box, points, points, "ball", r_max, 0, True)
            for nq, neighbors in test_set:
                rdf = freud.density.RDF(bins, r_max, r_min)

                if i < 3:
                    rdf.accumulate(nq, neighbors=neighbors)
                else:
                    rdf.compute(nq, neighbors=neighbors)
                self.assertTrue(rdf.box == box)
                correct = np.ones(bins, dtype=np.float32)
                npt.assert_allclose(rdf.RDF, correct, atol=tolerance)

                # Numerical integration to compute the running coordination
                # number will be highly inaccurate, so we can only test up to
                # a limited precision. Also, since dealing with nonzero r_min
                # values requires extrapolation, we only test when r_min=0.
                ndens = points.shape[0]/box.volume
                dr = (r_max - r_min) / bins
                bin_boundaries = np.array([r_min + dr*i for i in range(bins+1)
                                           if r_min + dr*i <= r_max])
                bin_volumes = 4/3*np.pi*np.diff(bin_boundaries**3)
                avg_counts = rdf.RDF*ndens*bin_volumes
                npt.assert_allclose(rdf.n_r, np.cumsum(avg_counts),
                                    rtol=tolerance)

    def test_repr(self):
        rdf = freud.density.RDF(r_max=10, bins=100, r_min=0.5)
        self.assertEqual(str(rdf), str(eval(repr(rdf))))

    def test_repr_png(self):
        r_max = 10.0
        bins = 10
        num_points = 10
        box_size = r_max*3.1
        box, points = util.make_box_and_random_points(box_size, num_points)
        rdf = freud.density.RDF(bins, r_max)

        with self.assertRaises(AttributeError):
            rdf.plot()
        self.assertEqual(rdf._repr_png_(), None)

        rdf.accumulate((box, points))
        rdf._repr_png_()

    def test_points_ne_query_points(self):
        r_max = 100.0
        bins = 100
        box_size = r_max*5
        box = freud.box.Box.square(box_size)

        rdf = freud.density.RDF(bins, r_max)

        query_points = []
        supposed_RDF = [0]
        N = 100

        # With points closely centered around the origin,
        # the cumulative average bin counts should be same as
        # having a single point at the origin.
        # Also, we can check for whether points are not considered against
        # each other.
        dr = r_max/bins
        points = [[dr/4, 0, 0], [-dr/4, 0, 0], [0, dr/4, 0], [0, -dr/4, 0]]
        for r in rdf.bin_centers:
            for k in range(N):
                query_points.append([r * np.cos(2*np.pi*k/N),
                                     r * np.sin(2*np.pi*k/N), 0])
            supposed_RDF.append(supposed_RDF[-1] + N)
        supposed_RDF = np.array(supposed_RDF[1:])

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "ball", r_max, 0, False)
        for nq, neighbors in test_set:
            rdf = freud.density.RDF(bins, r_max)
            rdf.compute(nq, query_points, neighbors=neighbors)

            npt.assert_allclose(rdf.n_r, supposed_RDF, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
