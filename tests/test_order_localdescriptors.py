import numpy as np
import numpy.testing as npt
import freud
from freud.order import LocalDescriptors
import freud.parallel;freud.parallel.setNumThreads(1)
import unittest

class TestLocalDescriptors(unittest.TestCase):
    def test_shape(self):
        N = 1000
        Nneigh = 4
        lmax = 8

        box = freud.trajectory.Box(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = LocalDescriptors(box, Nneigh, lmax, .5, True)
        comp.compute(positions)

        sphs = comp.getSph()

        assert sphs.shape[0] == N
        assert sphs.shape[1] == Nneigh

if __name__ == '__main__':
    unittest.main()
