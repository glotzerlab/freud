import numpy as np
import freud
import unittest
from freud.errors import FreudDeprecationWarning
import warnings
import util


class TestBondOrder(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_nonzero_bins(self):
        """Test that there are exactly 12 non-zero bins for a perfect
        FCC structure"""
        (box, positions) = util.make_fcc(4, 4, 4)
        quats = np.zeros((len(positions), 4), dtype=np.float32)
        quats[:, 0] = 1

        num_neighbors = 12
        npt = npp = 6
        bo = freud.environment.BondOrder(1.5, 0, num_neighbors, npt, npp)

        bo.compute(box, positions, quats, positions, quats)
        self.assertEqual(np.sum(bo.bond_order > 0), 12)
        self.assertEqual(bo.n_bins_theta, npt)
        self.assertEqual(bo.n_bins_phi, npp)

        self.assertTrue(np.allclose(
            bo.theta, np.array([(2*i+1)*np.pi/6 for i in range(npt)])))
        self.assertTrue(np.allclose(
            bo.phi, np.array([(2*i+1)*np.pi/12 for i in range(npp)])))

        bo.reset()
        bo.accumulate(box, positions, quats, positions, quats, mode='lbod')
        self.assertEqual(np.sum(bo.getBondOrder() > 0), 12)
        self.assertEqual(bo.getNBinsTheta(), npt)
        self.assertEqual(bo.getNBinsPhi(), npp)

        self.assertTrue(np.allclose(
            bo.getTheta(), np.array([(2*i+1)*np.pi/6 for i in range(npt)])))
        self.assertTrue(np.allclose(
            bo.getPhi(), np.array([(2*i+1)*np.pi/12 for i in range(npp)])))

        bo.compute(box, positions, quats, positions, quats, mode='obcd')
        self.assertEqual(np.sum(bo.bond_order > 0), 12)
        self.assertEqual(bo.n_bins_theta, npt)
        self.assertEqual(bo.n_bins_phi, npp)

        self.assertTrue(np.allclose(
            bo.theta, np.array([(2*i+1)*np.pi/6 for i in range(npt)])))
        self.assertTrue(np.allclose(
            bo.phi, np.array([(2*i+1)*np.pi/12 for i in range(npp)])))

        bo.resetBondOrder()
        bo.accumulate(box, positions, quats, positions, quats, mode='oocd')
        self.assertEqual(np.sum(bo.bond_order > 0), 12)
        self.assertEqual(bo.n_bins_theta, npt)
        self.assertEqual(bo.n_bins_phi, npp)

        self.assertTrue(np.allclose(
            bo.theta, np.array([(2*i+1)*np.pi/6 for i in range(npt)])))
        self.assertTrue(np.allclose(
            bo.phi, np.array([(2*i+1)*np.pi/12 for i in range(npp)])))


if __name__ == '__main__':
    unittest.main()
