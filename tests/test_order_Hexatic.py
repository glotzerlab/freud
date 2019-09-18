import numpy.testing as npt
import numpy as np
import freud
import unittest
import util


class TestHexatic(unittest.TestCase):
    def test_getK(self):
        hop = freud.order.Hexatic()
        npt.assert_equal(hop.k, 6)

    def test_getK_pass(self):
        k = 3
        hop = freud.order.Hexatic(k)
        npt.assert_equal(hop.k, 3)

    def test_order_size(self):
        boxlen = 10
        N = 500
        box, points = util.make_box_and_random_points(boxlen, N, is2D=True)
        hop = freud.order.Hexatic()
        hop.compute(box, points)
        npt.assert_equal(len(hop.order), N)

    def test_compute_random(self):
        boxlen = 10
        N = 500
        r_max = 3
        box, points = util.make_box_and_random_points(boxlen, N, is2D=True)
        hop = freud.order.Hexatic()
        hop.compute(box, points)
        npt.assert_allclose(np.mean(hop.order), 0. + 0.j, atol=1e-1)

    def test_compute(self):
        boxlen = 10
        r_max = 3
        box = freud.box.Box.square(boxlen)
        points = [[0.0, 0.0, 0.0]]

        for i in range(6):
            points.append([np.cos(float(i) * 2.0 * np.pi / 6.0),
                           np.sin(float(i) * 2.0 * np.pi / 6.0),
                           0.0])

        points = np.asarray(points, dtype=np.float32)
        points[:, 2] = 0.0
        hop = freud.order.Hexatic()

        # Test access
        hop.k
        with self.assertRaises(AttributeError):
            hop.order

        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'nearest', r_max, 6, True)
        for ts in test_set:
            hop.compute(box, ts[0], nlist=ts[1])
            # Test access
            hop.k
            hop.order

            npt.assert_allclose(hop.order[0], 1. + 0.j, atol=1e-1)

    def test_repr(self):
        hop = freud.order.Hexatic(3)
        self.assertEqual(str(hop), str(eval(repr(hop))))


if __name__ == '__main__':
    unittest.main()
