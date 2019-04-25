import numpy as np
import freud
import unittest
import warnings
import util


class TestBondOrder(unittest.TestCase):
    def test_bond_order(self):
        """Test the bond order diagram calculation."""
        (box, positions) = util.make_fcc(4, 4, 4)
        quats = np.zeros((len(positions), 4), dtype=np.float32)
        quats[:, 0] = 1

        r_cut = 1.5
        num_neighbors = 12
        npt = npp = 6
        bo = freud.environment.BondOrder(r_cut, 0, num_neighbors, npt, npp)

        # Test that there are exactly 12 non-zero bins for a perfect FCC
        # structure.
        bo.compute(box, positions, quats, positions, quats)
        op_value = bo.bond_order.copy()
        self.assertEqual(np.sum(op_value > 0), 12)

        # Test all the basic attributes.
        self.assertEqual(bo.n_bins_theta, npt)
        self.assertEqual(bo.n_bins_phi, npp)
        self.assertEqual(bo.box, box)
        self.assertTrue(np.allclose(
            bo.theta, np.array([(2*i+1)*np.pi/6 for i in range(npt)])))
        self.assertTrue(np.allclose(
            bo.phi, np.array([(2*i+1)*np.pi/12 for i in range(npp)])))

        # Test that reset works.
        bo.reset()
        self.assertTrue(np.all(bo.bond_order == 0))

        # Test that lbod gives identical results when orientations are the
        # same.
        #TODO: Find a way to test a rotated system to ensure that lbod gives  # noqa
        # the desired results.
        bo.accumulate(box, positions, quats, positions, quats, mode='lbod')
        self.assertTrue(np.allclose(bo.bond_order, op_value))

        # Test that obcd gives identical results when orientations are the
        # same.
        bo.compute(box, positions, quats, positions, quats, mode='obcd')
        self.assertTrue(np.allclose(bo.bond_order, op_value))

        # Test that normal bod looks ordered for randomized orientations.
        np.random.seed(10893)
        random_quats = np.random.rand(len(positions), 4)
        random_quats /= np.linalg.norm(random_quats, axis=1)[:, np.newaxis]
        bo.compute(box, positions, random_quats, positions, random_quats)
        self.assertTrue(np.allclose(bo.bond_order, op_value))

        # Ensure that obcd looks random for the randomized orientations.
        bo.compute(box, positions, random_quats, positions, random_quats,
                   mode='obcd')
        self.assertTrue(not np.allclose(bo.bond_order, op_value))
        self.assertEqual(np.sum(bo.bond_order > 0), bo.bond_order.size)

        # Test that oocd shows exactly one peak when all orientations are the
        # same.
        bo.reset()
        bo.accumulate(box, positions, quats, positions, quats, mode='oocd')
        self.assertEqual(np.sum(bo.bond_order > 0), 1)
        self.assertTrue(bo.bond_order[0, 0] > 0)

        # Test that oocd is highly disordered with random quaternions. In
        # practice, the edge bins may still not get any values, so just check
        # that we get a lot of values.
        bo.compute(box, positions, random_quats, positions, random_quats,
                   mode='oocd')
        self.assertGreater(np.sum(bo.bond_order > 0), 30)

    def test_repr(self):
        bo = freud.environment.BondOrder(1.5, 0, 12, 6, 6)
        print(repr(bo))
        self.assertEqual(str(bo), str(eval(repr(bo))))


if __name__ == '__main__':
    unittest.main()
