import numpy as np
import numpy.testing as npt
from freud import box, density
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

        rdf = density.RDF(rmax, dr)

        npt.assert_almost_equal(rdf.getR(), r_list, decimal=3)

class TestRDF(unittest.TestCase):
    def test_random_point(self):
        rmax = 10.0
        dr = 1.0
        num_points = 10000
        box_size = rmax*3.1
        points = np.random.random_sample((num_points,3)).astype(np.float32)*box_size - box_size/2
        rdf = density.RDF(rmax, dr)
        rdf.accumulate(box.Box.cube(box_size), points, points)

        correct = np.ones(int(rmax/dr), dtype=np.float32)
        correct[0] = 0.0
        absolute_tolerance = 0.1
        npt.assert_allclose(rdf.getRDF(), correct, atol=absolute_tolerance)

if __name__ == '__main__':
    unittest.main()
