import unittest
import numpy.testing as npt
import numpy as np
import freud

class TestHexOrderParameter(unittest.TestCase):
    def test_getK(self):
        boxlen = 10
        N = 500
        rmax = 3

        box = freud.trajectory.Box(boxlen, is2D=True)

        hop = freud.order.HexOrderParameter(rmax)
        npt.assert_equal(hop.getK(), 6.0)

    def test_getK_pass(self):
        boxlen = 10
        N = 500
        rmax = 3
        k=3.0

        box = freud.trajectory.Box(boxlen, is2D=True)
        hop = freud.order.HexOrderParameter(rmax, k)
        npt.assert_equal(hop.getK(), 3.0)

    def test_getNP(self):
        boxlen = 10
        N = 500
        rmax = 3

        box = freud.trajectory.Box(boxlen, is2D=True)

        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N, 3)),
                            dtype=np.float32)
        points[:,2] = 0.0
        hop = freud.order.HexOrderParameter(rmax)
        hop.compute(box, points)
        npt.assert_equal(hop.getNP(), N)

    def test_compute_random(self):
        boxlen = 10
        N = 500
        rmax = 3

        box = freud.trajectory.Box(boxlen,  is2D=True)

        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N, 3)),
                            dtype=np.float32)
        points[:,2] = 0.0
        hop = freud.order.HexOrderParameter(rmax)
        hop.compute(box, points)
        npt.assert_almost_equal(np.mean(hop.getPsi()), 0. + 0.j, decimal=1)

    def test_compute(self):
        boxlen = 10
        N = 7
        rmax = 3

        box = freud.trajectory.Box(boxlen,  is2D=True)

        points = [[0.0, 0.0, 0.0]]

        for i in range(6):
            points.append([np.cos(float(i) * 2.0 * np.pi / 6.0),
                           np.sin(float(i) * 2.0 * np.pi / 6.0),
                           0.0])

        points = np.asarray(points, dtype=np.float32)
        points[:,2] = 0.0
        hop = freud.order.HexOrderParameter(rmax)
        hop.compute(box, points)
        npt.assert_almost_equal(hop.getPsi()[0], 1. + 0.j, decimal=1)

if __name__ == '__main__':
    unittest.main()
