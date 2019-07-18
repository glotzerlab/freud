import numpy.testing as npt
import numpy as np
import freud
import unittest
import util


class TestHexOrderParameter(unittest.TestCase):
    def test_getK(self):
        rmax = 3
        hop = freud.order.HexOrderParameter(rmax)
        npt.assert_equal(hop.K, 6)

    def test_getK_pass(self):
        rmax = 3
        k = 3
        hop = freud.order.HexOrderParameter(rmax, k)
        npt.assert_equal(hop.K, 3)

    def test_getNP(self):
        boxlen = 10
        N = 500
        rmax = 3
        box = freud.box.Box.square(boxlen)
        np.random.seed(0)
        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N, 3)),
                            dtype=np.float32)
        points[:, 2] = 0.0
        hop = freud.order.HexOrderParameter(rmax)
        hop.compute(box, points)
        npt.assert_equal(hop.num_particles, N)

    def test_compute_random(self):
        boxlen = 10
        N = 500
        rmax = 3
        box = freud.box.Box.square(boxlen)
        np.random.seed(0)
        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N, 3)),
                            dtype=np.float32)
        points[:, 2] = 0.0
        hop = freud.order.HexOrderParameter(rmax)
        hop.compute(box, points)
        npt.assert_allclose(np.mean(hop.psi), 0. + 0.j, atol=1e-1)

        self.assertTrue(hop.box == box)

    def test_compute(self):
        boxlen = 10
        rmax = 3
        box = freud.box.Box.square(boxlen)
        points = [[0.0, 0.0, 0.0]]

        for i in range(6):
            points.append([np.cos(float(i) * 2.0 * np.pi / 6.0),
                           np.sin(float(i) * 2.0 * np.pi / 6.0),
                           0.0])

        points = np.asarray(points, dtype=np.float32)
        points[:, 2] = 0.0
        hop = freud.order.HexOrderParameter(rmax)

        # Test access
        with self.assertRaises(AttributeError):
            hop.num_particles
        with self.assertRaises(AttributeError):
            hop.box
        with self.assertRaises(AttributeError):
            hop.psi

        test_set = util.makeRawQueryNlistTestSet(
            box, points, points, 'nearest', rmax, 6, True)
        for ts in test_set:
            hop.compute(box, ts[0], nlist=ts[1])
            # Test access
            hop.num_particles
            hop.box
            hop.psi

            npt.assert_allclose(hop.psi[0], 1. + 0.j, atol=1e-1)

    def test_repr(self):
        hop = freud.order.HexOrderParameter(3.0, 6, 7)
        self.assertEqual(str(hop), str(eval(repr(hop))))


if __name__ == '__main__':
    unittest.main()
