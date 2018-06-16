import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestLocalQl(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalQl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.Ql.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalQl(box, 1.5, 6)

        comp.compute(positions)
        self.assertAlmostEqual(np.average(comp.Ql), 0.57452422)
        assert np.allclose(comp.Ql, comp.Ql[0])

        comp.computeAve(positions)
        self.assertAlmostEqual(np.average(comp.Ql), 0.57452422)
        assert np.allclose(comp.ave_Ql, comp.ave_Ql[0])

        comp.computeNorm(positions)
        self.assertAlmostEqual(np.average(comp.Ql), 0.57452422)
        assert np.allclose(comp.norm_Ql, comp.norm_Ql[0])

        comp.computeAveNorm(positions)
        self.assertAlmostEqual(np.average(comp.Ql), 0.57452422)
        assert np.allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0])


class TestLocalQlNear(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.Ql.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)

        comp.compute(positions)
        self.assertAlmostEqual(np.average(comp.Ql), 0.57452416)
        assert np.allclose(comp.Ql, comp.Ql[0])

        comp.computeAve(positions)
        self.assertAlmostEqual(np.average(comp.Ql), 0.57452416)
        assert np.allclose(comp.ave_Ql, comp.ave_Ql[0])

        comp.computeNorm(positions)
        self.assertAlmostEqual(np.average(comp.Ql), 0.57452416)
        assert np.allclose(comp.norm_Ql, comp.norm_Ql[0])

        comp.computeAveNorm(positions)
        self.assertAlmostEqual(np.average(comp.Ql), 0.57452416)
        assert np.allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0])


if __name__ == '__main__':
    unittest.main()
