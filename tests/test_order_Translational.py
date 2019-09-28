import numpy as np
import numpy.testing as npt
import freud
import unittest
import itertools
import util


class TestTranslational(unittest.TestCase):
    def test_simple(self):
        box = freud.box.Box.square(10)

        # make a square grid
        xs = np.linspace(-box.Lx/2, box.Lx/2, 10, endpoint=False)
        positions = np.zeros((len(xs)**2, 3), dtype=np.float32)
        positions[:, :2] = np.array(list(itertools.product(xs, xs)),
                                    dtype=np.float32)

        r_max = 1.1
        n = 4
        trans = freud.order.Translational(4)
        # Test access
        with self.assertRaises(AttributeError):
            trans.order

        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, 'nearest', r_max, n, True)
        for nq, neighbors in test_set:
            trans.compute(nq, neighbors=neighbors)
            # Test access
            trans.order

            npt.assert_allclose(trans.order, 0, atol=1e-6)

    def test_repr(self):
        trans = freud.order.Translational(4)
        self.assertEqual(str(trans), str(eval(repr(trans))))


if __name__ == '__main__':
    unittest.main()
