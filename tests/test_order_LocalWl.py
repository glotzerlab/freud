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
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalWl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.Wl.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalWl(box, 1.5, 6)

        comp.compute(positions)
        assert np.allclose(comp.Wl, comp.Wl[0])
        self.assertAlmostEqual(np.real(np.average(comp.Wl)), -0.0026260)

        comp.computeAve(positions)
        assert np.allclose(comp.ave_Wl, comp.ave_Wl[0])
        self.assertAlmostEqual(np.real(np.average(comp.Wl)), -0.0026260)

        comp.computeNorm(positions)
        assert np.allclose(comp.norm_Wl, comp.norm_Wl[0])
        self.assertAlmostEqual(np.real(np.average(comp.Wl)), -0.0026260)

        comp.computeAveNorm(positions)
        assert np.allclose(comp.ave_norm_Wl, comp.ave_norm_Wl[0])
        self.assertAlmostEqual(np.real(np.average(comp.Wl)), -0.0026260)

class TestLocalWlNear(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalWlNear(box, 1.5, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.Wl.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalWlNear(box, 1.5, 6, 12)

        comp.compute(positions)
        assert np.allclose(comp.Wl, comp.Wl[0])
        self.assertAlmostEqual(np.real(np.average(comp.Wl)), -0.0026260)

        comp.computeAve(positions)
        assert np.allclose(comp.ave_Wl, comp.ave_Wl[0])
        self.assertAlmostEqual(np.real(np.average(comp.Wl)), -0.0026260)

        comp.computeNorm(positions)
        assert np.allclose(comp.norm_Wl, comp.norm_Wl[0])
        self.assertAlmostEqual(np.real(np.average(comp.Wl)), -0.0026260)

        comp.computeAveNorm(positions)
        assert np.allclose(comp.ave_norm_Wl, comp.ave_norm_Wl[0])
        self.assertAlmostEqual(np.real(np.average(comp.Wl)), -0.0026260)

if __name__ == '__main__':
    unittest.main()
