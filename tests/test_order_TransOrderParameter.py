import numpy as np
import freud
import unittest
import itertools


class TestTransOrder(unittest.TestCase):
    def test_simple(self):
        box = freud.box.Box.square(10)

        # make a square grid
        xs = np.linspace(-box.getLx()/2, box.getLx()/2, 10, endpoint=False)
        positions = np.zeros((len(xs)**2, 3), dtype=np.float32)
        positions[:, :2] = np.array(list(itertools.product(xs, xs)),
                                    dtype=np.float32)

        trans = freud.order.TransOrderParameter(1.1, 4, 4)
        trans.compute(box, positions)

        self.assertTrue(np.allclose(trans.d_r, 0, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
