import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestLocalWl(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalWl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.Wl.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalWl(box, 1.5, 6)

        comp.compute(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), -0.0026260, atol=1e-5))
        self.assertTrue(np.allclose(comp.Wl, comp.Wl[0]))

        comp.computeAve(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), -0.0026260, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_Wl, comp.ave_Wl[0]))

        comp.computeNorm(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), -0.0026260, atol=1e-5))
        self.assertTrue(np.allclose(comp.norm_Wl, comp.norm_Wl[0]))

        comp.computeAveNorm(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), -0.0026260, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_norm_Wl, comp.ave_norm_Wl[0]))

        self.assertEqual(box, comp.box)

        self.assertEqual(len(positions), comp.num_particles)


class TestLocalWlNear(unittest.TestCase):
    def test_init_kwargs(self):
        """Ensure that keyword arguments are correctly accepted"""
        box = freud.box.Box.cube(10)
        comp = freud.order.LocalWlNear(box, 1.5, 6, kn=12)  # noqa: F841

    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalWlNear(box, 1.5, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.Wl.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalWlNear(box, 1.5, 6, 12)

        comp.compute(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), -0.0026260, atol=1e-5))
        self.assertTrue(np.allclose(comp.Wl, comp.Wl[0]))

        comp.computeAve(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), -0.0026260, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_Wl, comp.ave_Wl[0]))

        comp.computeNorm(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), -0.0026260, atol=1e-5))
        self.assertTrue(np.allclose(comp.norm_Wl, comp.norm_Wl[0]))

        comp.computeAveNorm(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), -0.0026260, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_norm_Wl, comp.ave_norm_Wl[0]))

        self.assertEqual(box, comp.box)

        self.assertEqual(len(positions), comp.num_particles)


if __name__ == '__main__':
    unittest.main()
