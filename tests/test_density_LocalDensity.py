import numpy as np
import numpy.testing as npt
from freud import box, density
import unittest


class TestLD(unittest.TestCase):
    """Test fixture for LocalDensity"""

    def setUp(self):
        """Initialize a box with randomly placed particles"""

        self.box = box.Box.cube(10)
        np.random.seed(0)
        self.pos = np.array(np.random.random(size=(10000, 3)),
                            dtype=np.float32) * 10 - 5
        self.ld = density.LocalDensity(3, 1, 1)

    @unittest.skip("Skip for CircleCI")
    def test_compute_api(self):
        # test 2 args, no keyword
        self.ld.compute(self.box, self.pos)
        # test 3 args, no keyword
        self.ld.compute(self.box, self.pos, self.pos)
        # test 2 args, keyword
        self.ld.compute(box=self.box, ref_points=self.pos)
        # test 3 args, keyword
        self.ld.compute(box=self.box, ref_points=self.pos, points=self.pos)

    def test_density(self):
        """Test that LocalDensity computes the correct density at each point"""

        self.ld.compute(self.box, self.pos, self.pos)
        density = self.ld.getDensity()

        npt.assert_array_less(np.fabs(density - 10.0), 1.5)

        neighbors = self.ld.num_neighbors
        npt.assert_array_less(np.fabs(neighbors - 1130.973355292), 200)

    @unittest.skip("Skip for CircleCI")
    def test_oldapi(self):
        """Test that LocalDensity can compute a correct density at each point
        using the old API"""

        self.ld.compute(self.box, self.pos)
        density = self.ld.getDensity()

        npt.assert_array_less(np.fabs(density - 10.0), 1.5)

        neighbors = self.ld.num_neighbors
        npt.assert_array_less(np.fabs(neighbors - 1130.973355292), 200)


if __name__ == '__main__':
    unittest.main()
