import numpy as np
import numpy.testing as npt
import freud
import unittest
import itertools


class TestTransOrder(unittest.TestCase):
    def test_simple(self):
        box = freud.box.Box.square(10)

        # make a square grid
        xs = np.linspace(-box.Lx/2, box.Lx/2, 10, endpoint=False)
        positions = np.zeros((len(xs)**2, 3), dtype=np.float32)
        positions[:, :2] = np.array(list(itertools.product(xs, xs)),
                                    dtype=np.float32)

        trans = freud.order.TransOrderParameter(1.1, 4, 4)
        # Test access
        with self.assertRaises(AttributeError):
            trans.num_particles
        with self.assertRaises(AttributeError):
            trans.box
        with self.assertRaises(AttributeError):
            trans.d_r
        trans.compute(box, positions)

        # Test access
        trans.num_particles
        trans.box
        trans.d_r

        npt.assert_allclose(trans.d_r, 0, atol=1e-6)

        self.assertEqual(box, trans.box)
        self.assertEqual(len(positions), trans.num_particles)

    def test_repr(self):
        trans = freud.order.TransOrderParameter(1.1, 4, 4)
        self.assertEqual(str(trans), str(eval(repr(trans))))


if __name__ == '__main__':
    unittest.main()
