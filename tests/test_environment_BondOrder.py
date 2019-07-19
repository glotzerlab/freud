import numpy as np
import freud
import unittest
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
        bo = freud.environment.BondOrder(r_cut, num_neighbors, npt, npp)

        # Test access
        with self.assertRaises(AttributeError):
            bo.box
        with self.assertRaises(AttributeError):
            bo.bond_order

        # Test that there are exactly 12 non-zero bins for a perfect FCC
        # structure.
        bo.compute(box, positions, quats)
        op_value = bo.bond_order.copy()
        self.assertEqual(np.sum(op_value > 0), 12)

        # Test access
        bo.box
        bo.bond_order

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
        # Test access
        with self.assertRaises(AttributeError):
            bo.box
        with self.assertRaises(AttributeError):
            bo.bond_order

        test_set = util.makeRawQueryNlistTestSet(
            box, positions, positions, "nearest", r_cut, num_neighbors, True)
        for ts in test_set:
            bo.reset()
            # Test that lbod gives identical results when orientations are the
            # same.
            #TODO: Find a way to test a rotated system to ensure that lbod gives  # noqa
            # the desired results.
            bo.accumulate(box, ts[0], quats, mode='lbod', nlist=ts[1])
            self.assertTrue(np.allclose(bo.bond_order, op_value))

            # Test access
            bo.box
            bo.bond_order

            # Test that obcd gives identical results when orientations are the
            # same.
            bo.compute(box, ts[0], quats, mode='obcd', nlist=ts[1])
            self.assertTrue(np.allclose(bo.bond_order, op_value))

            # Test that normal bod looks ordered for randomized orientations.
            np.random.seed(10893)
            random_quats = np.random.rand(len(positions), 4)
            random_quats /= np.linalg.norm(random_quats, axis=1)[:, np.newaxis]
            bo.compute(box, ts[0], random_quats, nlist=ts[1])
            self.assertTrue(np.allclose(bo.bond_order, op_value))

            # Ensure that obcd looks random for the randomized orientations.
            bo.compute(box, ts[0], random_quats, mode='obcd', nlist=ts[1])
            self.assertTrue(not np.allclose(bo.bond_order, op_value))
            self.assertEqual(np.sum(bo.bond_order > 0), bo.bond_order.size)

            # Test that oocd shows exactly one peak when all orientations
            # are the same.
            bo.reset()
            bo.accumulate(box, ts[0], quats, mode='oocd', nlist=ts[1])
            self.assertEqual(np.sum(bo.bond_order > 0), 1)
            self.assertTrue(bo.bond_order[0, 0] > 0)

            # Test that oocd is highly disordered with random quaternions. In
            # practice, the edge bins may still not get any values, so just
            # check that we get a lot of values.
            bo.compute(box, ts[0], random_quats, mode='oocd', nlist=ts[1])
            self.assertGreater(np.sum(bo.bond_order > 0), 30)

    def test_repr(self):
        bo = freud.environment.BondOrder(1.5, 12, 6, 6)
        self.assertEqual(str(bo), str(eval(repr(bo))))


if __name__ == '__main__':
    unittest.main()
