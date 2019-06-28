import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


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
        npt.assert_allclose(np.average(comp.order), 0.57452422, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, 0.57452422, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, average=True)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), 0.57452422, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, 0.57452422, delta=1e-5)

    def test_identical_environments_Ql_near(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), 0.57452422, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, 0.57452422, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12, average=True)
        comp.compute(box, positions)
        npt.assert_allclose(np.average(comp.order), 0.57452422, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(comp.norm, 0.57452422, delta=1e-5)

    def test_identical_environments_Wl(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.Steinhardt(1.5, 6, Wl=True)
        comp.compute(box, positions)
        npt.assert_allclose(
            np.real(np.average(comp.order)), -0.002626035, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(np.real(comp.norm), -0.002626035, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, Wl=True, average=True)
        comp.compute(box, positions)
        npt.assert_allclose(
            np.real(np.average(comp.order)), -0.002626035, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(np.real(comp.norm), -0.002626035, delta=1e-5)

        self.assertEqual(len(positions), comp.num_particles)

    def test_identical_environments_Wl_near(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12, Wl=True)
        comp.compute(box, positions)
        npt.assert_allclose(
            np.real(np.average(comp.order)), -0.002626035, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(np.real(comp.norm), -0.002626035, delta=1e-5)

        comp = freud.order.Steinhardt(1.5, 6, num_neigh=12, Wl=True,
                                      average=True)
        comp.compute(box, positions)
        npt.assert_allclose(
            np.real(np.average(comp.order)), -0.002626035, atol=1e-5)
        npt.assert_allclose(comp.order, comp.order[0], atol=1e-5)
        self.assertAlmostEqual(np.real(comp.norm), -0.002626035, delta=1e-5)

        self.assertEqual(len(positions), comp.num_particles)

    def test_repr(self):
        comp = freud.order.Steinhardt(1.5, 6)
        self.assertEqual(str(comp), str(eval(repr(comp))))


if __name__ == '__main__':
    unittest.main()
