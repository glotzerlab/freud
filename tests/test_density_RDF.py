import numpy as np
import numpy.testing as npt
from freud import box, density
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
            rdf = density.RDF(rmax, dr, rmin=rmin)
            npt.assert_almost_equal(rdf.R, r_list, decimal=3)

    def test_invalid_rdf(self):
        # Make sure that invalid RDF objects raise errors
        with self.assertRaises(ValueError):
            density.RDF(rmax=-1, dr=0.1)
        with self.assertRaises(ValueError):
            density.RDF(rmax=1, dr=0)
        with self.assertRaises(ValueError):
            density.RDF(rmax=1, dr=0.1, rmin=2)

    def test_random_point(self):
        for rmin in [0, 0.05, 0.1, 1.0, 3.0]:
            rmax = 10.0
            dr = 1.0
            nbins = int((rmax - rmin) / dr)
            num_points = 10000
            box_size = rmax*3.1
            np.random.seed(0)
            points = np.random.random_sample((num_points, 3)).astype(
                np.float32) * box_size - box_size/2
            rdf = density.RDF(rmax, dr, rmin=rmin)
            fbox = box.Box.cube(box_size)
            rdf.accumulate(fbox, points)
            correct = np.ones(nbins, dtype=np.float32)
            correct[0] = 0.0
            absolute_tolerance = 0.1
            npt.assert_allclose(rdf.RDF, correct, atol=absolute_tolerance)


if __name__ == '__main__':
    unittest.main()
