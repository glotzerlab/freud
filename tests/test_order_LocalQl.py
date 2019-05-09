import numpy as np
import numpy.testing as npt
import freud
import unittest
import warnings
import util


class TestLocalQl(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)
        positions.flags['WRITEABLE'] = False

        comp = freud.order.LocalQl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.Ql.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalQl(box, 1.5, 6)

        comp.compute(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452422, rtol=1e-6)
        npt.assert_allclose(comp.Ql, comp.Ql[0], rtol=1e-6)

        comp.computeAve(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452422, rtol=1e-6)
        npt.assert_allclose(comp.ave_Ql, comp.ave_Ql[0], rtol=1e-6)

        comp.computeNorm(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452422, rtol=1e-6)
        npt.assert_allclose(comp.norm_Ql, comp.norm_Ql[0], rtol=1e-6)

        comp.computeAveNorm(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452422, rtol=1e-6)
        npt.assert_allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0], rtol=1e-6)

        self.assertEqual(box, comp.box)

        self.assertEqual(len(positions), comp.num_particles)

    def test_repr(self):
        box = freud.box.Box.cube(10)
        comp = freud.order.LocalQl(box, 1.5, 6)
        self.assertEqual(str(comp), str(eval(repr(comp))))


class TestLocalQlNear(unittest.TestCase):
    def test_init_kwargs(self):
        """Ensure that keyword arguments are correctly accepted"""
        box = freud.box.Box.cube(10)
        comp = freud.order.LocalQlNear(box, 1.5, 6, kn=12)  # noqa: F841

    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.Ql.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)

        comp.compute(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452416, rtol=1e-6)
        npt.assert_allclose(comp.Ql, comp.Ql[0], rtol=1e-6)

        comp.computeAve(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452416, rtol=1e-6)
        npt.assert_allclose(comp.ave_Ql, comp.ave_Ql[0], rtol=1e-6)

        comp.computeNorm(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452416, rtol=1e-6)
        npt.assert_allclose(comp.norm_Ql, comp.norm_Ql[0], rtol=1e-6)

        comp.computeAveNorm(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452416, rtol=1e-6)
        npt.assert_allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0], rtol=1e-6)

        self.assertEqual(box, comp.box)

        self.assertEqual(len(positions), comp.num_particles)

    def test_repr(self):
        box = freud.box.Box.cube(10)
        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)
        self.assertEqual(str(comp), str(eval(repr(comp))))


if __name__ == '__main__':
    unittest.main()
