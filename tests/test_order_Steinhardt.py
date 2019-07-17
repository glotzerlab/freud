import numpy as np
import numpy.testing as npt
import freud
import unittest
import util

PERFECT_FCC_Q6 = 0.57452416
PERFECT_FCC_W6 = -0.00262604

class TestSteinhardt(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

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

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12)
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
        npt.assert_allclose(
            np.real(np.average(comp.order)), PERFECT_FCC_W6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(np.real(comp.norm), PERFECT_FCC_W6, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, Wl=True, average=True)
        comp.compute(box, positions)
        npt.assert_allclose(
            np.real(np.average(comp.order)), PERFECT_FCC_W6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(np.real(comp.norm), PERFECT_FCC_W6, delta=1e-5)

        self.assertEqual(len(positions), comp.num_particles)

    def test_identical_environments_Wl_near(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12, Wl=True)
        comp.compute(box, positions)
        npt.assert_allclose(
            np.real(np.average(comp.order)), PERFECT_FCC_W6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(np.real(comp.norm), PERFECT_FCC_W6, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12, Wl=True,
                                      average=True)
        comp.compute(box, positions)
        npt.assert_allclose(
            np.real(np.average(comp.order)), PERFECT_FCC_W6, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(np.real(comp.norm), PERFECT_FCC_W6, delta=1e-5)

        self.assertEqual(len(positions), comp.num_particles)

    def test_repr(self):
        comp = freud.order.Steinhardt(1.5, 6)
        self.assertEqual(str(comp), str(eval(repr(comp))))


if __name__ == '__main__':
    unittest.main()
