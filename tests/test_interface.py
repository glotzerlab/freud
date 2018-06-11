import numpy as np
import freud
import unittest
import util


class TestInterface(unittest.TestCase):
    def test_take_one(self):
        """Test that there is exactly 1 or 12 particles at the interface when
        one particle is removed from an FCC structure"""
        np.random.seed(0)
        (box, positions) = util.make_fcc(4, 4, 4, noise=1e-2)

        index = np.random.randint(0, len(positions))

        point = positions[index].reshape((1, 3))
        others = np.concatenate([positions[:index], positions[index + 1:]])

        inter = freud.interface.InterfaceMeasure(box, 1.5)

        self.assertEqual(inter.compute(point, others), 1)
        self.assertEqual(inter.compute(others, point), 12)


if __name__ == '__main__':
    unittest.main()
