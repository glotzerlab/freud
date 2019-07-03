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
        bo = freud.environment.BondOrder(r_cut, 0, num_neighbors, npt, npp)

        # Test access
        with self.assertRaises(AttributeError):
            bo.box
        with self.assertRaises(AttributeError):
            bo.bond_order

        # Test that there are exactly 12 non-zero bins for a perfect FCC
        # structure.
        bo.compute(box, positions, quats, positions, quats)
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

        # Test that lbod gives identical results when orientations are the
        # same.
        #TODO: Find a way to test a rotated system to ensure that lbod gives  # noqa
        # the desired results.
        bo.accumulate(box, positions, quats, positions, quats, mode='lbod')
        self.assertTrue(np.allclose(bo.bond_order, op_value))

        # Test access
        bo.box
        bo.bond_order

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
        self.assertEqual(str(bo), str(eval(repr(bo))))

    def test_ref_points_ne_points(self):
        lattice_size = 10
        # big box to ignore periodicity
        box = freud.box.Box.square(lattice_size*5)
        angle = np.pi/30
        ref_points, points = util.make_alternating_lattice(lattice_size, angle)

        # actually not used
        rmax = 1.6
        k = 0

        # to make sure what we skip ref_points ref_points neighbors
        n = 12
        n_bins_t = 30
        n_bins_p = 2
        bod = freud.environment.BondOrder(rmax=rmax, k=k, n=n,
                                          n_bins_t=n_bins_t, n_bins_p=n_bins_p)

        # orientations are not used in bod mode
        ref_orientations = np.array([[1, 0, 0, 0]]*len(ref_points))
        orientations = np.array([[1, 0, 0, 0]]*len(points))

        bod.compute(box, ref_points, ref_orientations, points, orientations)

        # we want to make sure that we get 12 nonzero places, so we can test
        # whether we are not considering neighbors between ref_points
        self.assertEqual(np.count_nonzero(bod.bond_order), 12)
        self.assertEqual(len(np.unique(bod.bond_order)), 2)


if __name__ == '__main__':
    unittest.main()
