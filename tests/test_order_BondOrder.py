import numpy as np
import numpy.testing as npt
import freud
import unittest
import internal

class TestBondOrder(unittest.TestCase):
    def test_nonzero_bins(self):
        """Test that there are exactly 12 non-zero bins for a perfect FCC structure"""
        (box, positions) = internal.make_fcc(4, 4, 4)
        quats = np.zeros((len(positions), 4), dtype=np.float32)
        quats[:, 0] = 1

        bo = freud.order.BondOrder(1.5, 0, 12, 6, 6)
        bo.compute(box, positions, quats, positions, quats)

        self.assertEqual(np.sum(bo.getBondOrder() > 0), 12)

if __name__ == '__main__':
    unittest.main()
