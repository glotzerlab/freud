import numpy as np
import numpy.testing as npt
import freud
import rowan
import unittest
import util

# Validated against manual calculation and pyboo
PERFECT_FCC_Q6 = 0.57452416
PERFECT_FCC_W6 = -0.00262604


class TestSteinhardt(unittest.TestCase):
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = util.make_box_and_random_points(L, N)

        comp = freud.order.Steinhardt(1.5, 6)
        comp.compute(box, positions)

        npt.assert_equal(comp.order.shape[0], N)

    def test_identical_environments_Ql(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.Steinhardt(1.5, 6)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), PERFECT_FCC_Q6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, PERFECT_FCC_Q6, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, average=True)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), PERFECT_FCC_Q6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, PERFECT_FCC_Q6, delta=1e-5)

    def test_identical_environments_Ql_near(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        # Perturb one position
        perturbed_positions = positions.copy()
        perturbed_positions[-1] += [0.1, 0, 0]

        # Use a really small cutoff to ensure that it is used as a soft cutoff
        comp = freud.order.Steinhardt(0.1, 6, num_neigh=12)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), PERFECT_FCC_Q6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, PERFECT_FCC_Q6, delta=1e-5)

        # Ensure exactly 13 values change for the perturbed system
        comp.compute(box, perturbed_positions)
        self.assertEqual(
            sum(~np.isclose(comp.Ql, PERFECT_FCC_Q6, rtol=1e-6)), 13)

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12, average=True)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), PERFECT_FCC_Q6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, PERFECT_FCC_Q6, delta=1e-5)

        # More than 13 particles should change for Ql averaged over neighbors
        comp.compute(box, perturbed_positions)
        self.assertGreater(
            sum(~np.isclose(comp.order, PERFECT_FCC_Q6, rtol=1e-6)), 13)

    def test_identical_environments_Wl(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.Steinhardt(1.5, 6, Wl=True)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), PERFECT_FCC_W6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, PERFECT_FCC_W6, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, Wl=True, average=True)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), PERFECT_FCC_W6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, PERFECT_FCC_W6, delta=1e-5)

        self.assertEqual(len(positions), comp.num_particles)

    def test_identical_environments_Wl_near(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        # Use a really small cutoff to ensure that it is used as a soft cutoff
        comp = freud.order.Steinhardt(0.1, 6, num_neigh=12, Wl=True)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), PERFECT_FCC_W6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, PERFECT_FCC_W6, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12, Wl=True,
                                      average=True)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), PERFECT_FCC_W6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, PERFECT_FCC_W6, delta=1e-5)

        self.assertEqual(len(positions), comp.num_particles)

    def test_rotational_invariance(self):
        box = freud.box.Box.cube(10)
        positions = np.array([[0, 0, 0],
                              [-1, -1, 0],
                              [-1, 1, 0],
                              [1, -1, 0],
                              [1, 1, 0],
                              [-1, 0, -1],
                              [-1, 0, 1],
                              [1, 0, -1],
                              [1, 0, 1],
                              [0, -1, -1],
                              [0, -1, 1],
                              [0, 1, -1],
                              [0, 1, 1]])
        index_i = np.zeros(12)
        index_j = np.arange(1, 13)
        nlist = freud.locality.NeighborList.from_arrays(
            13, 13, index_i, index_j)

        q6 = freud.order.Steinhardt(1.5, 6)
        w6 = freud.order.Steinhardt(1.5, 6, Wl=True)

        q6.compute(box, positions, nlist=nlist)
        q6_unrotated_order = q6.order[0]
        w6.compute(box, positions, nlist=nlist)
        w6_unrotated_order = w6.order[0]

        for i in range(10):
            np.random.seed(i)
            quat = rowan.random.rand()
            positions_rotated = rowan.rotate(quat, positions)

            # Ensure Q6 is rotationally invariant
            q6.compute(box, positions_rotated, nlist=nlist)
            npt.assert_allclose(q6.order[0], q6_unrotated_order, rtol=1e-5)
            npt.assert_allclose(q6.order[0], PERFECT_FCC_Q6, rtol=1e-5)

            # Ensure W6 is rotationally invariant
            w6.compute(box, positions_rotated, nlist=nlist)
            npt.assert_allclose(w6.order[0], w6_unrotated_order, rtol=1e-5)
            npt.assert_allclose(w6.order[0], PERFECT_FCC_W6, rtol=1e-5)

    def test_repr(self):
        comp = freud.order.Steinhardt(1.5, 6)
        self.assertEqual(str(comp), str(eval(repr(comp))))


if __name__ == '__main__':
    unittest.main()
