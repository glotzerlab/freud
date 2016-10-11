from freud import box, density
import numpy
import math
from nose.tools import assert_equal, assert_almost_equal, assert_less, raises

class TestLD:
    """Test fixture for LocalDensity"""

    def setup(self):
        """ Initialize a box with randomly placed particles"""

        self.box = box.Box.cube(10);
        self.pos = numpy.array(numpy.random.random(size=(10000,3)), dtype=numpy.float32)*10 - 5
        self.ld = density.LocalDensity(3, 1, 1);

    def test_density(self):
        """Test that LocalDensity can compute a correct density at each point"""

        self.ld.compute(self.box, self.pos, self.pos);
        density = self.ld.getDensity();

        for i in range(0,len(self.pos)):
            assert_less(math.fabs(density[i]-10.0), 1.5);

        neighbors = self.ld.getNumNeighbors();
        for i in range(0,len(neighbors)):
            assert_less(math.fabs(neighbors[i]-1130.973355292), 200);

    def test_oldapi(self):
        """Test that LocalDensity can compute a correct density at each point, using the old API"""

        self.ld.compute(self.box, self.pos);
        density = self.ld.getDensity();

        for i in range(0,len(self.pos)):
            assert_less(math.fabs(density[i]-10.0), 1.5);

        neighbors = self.ld.getNumNeighbors();
        for i in range(0,len(neighbors)):
            assert_less(math.fabs(neighbors[i]-1130.973355292), 200);
