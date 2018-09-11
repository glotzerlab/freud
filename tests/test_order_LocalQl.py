import numpy as np
import numpy.testing as npt
import freud
import unittest
from freud.errors import FreudDeprecationWarning
import warnings
import util


class TestLocalQl(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalQl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.Ql.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalQl(box, 1.5, 6)

        comp.compute(positions)
        npt.assert_almost_equal(np.average(comp.Ql), 0.57452422, decimal=5)
        npt.assert_almost_equal(comp.Ql, comp.Ql[0])
        npt.assert_almost_equal(comp.getQl(), comp.Ql[0])

        comp.computeAve(positions)
        npt.assert_almost_equal(np.average(comp.Ql), 0.57452422, decimal=5)
        npt.assert_almost_equal(comp.ave_Ql, comp.ave_Ql[0])
        npt.assert_almost_equal(comp.getAveQl(), comp.ave_Ql[0])

        comp.computeNorm(positions)
        npt.assert_almost_equal(np.average(comp.Ql), 0.57452422, decimal=5)
        npt.assert_almost_equal(comp.norm_Ql, comp.norm_Ql[0])
        npt.assert_almost_equal(comp.getQlNorm(), comp.norm_Ql[0])

        comp.computeAveNorm(positions)
        npt.assert_almost_equal(np.average(comp.Ql), 0.57452422, decimal=5)
        npt.assert_almost_equal(comp.ave_norm_Ql, comp.ave_norm_Ql[0])
        npt.assert_almost_equal(comp.getQlAveNorm(), comp.ave_norm_Ql[0])

        self.assertEqual(box, comp.box)
        self.assertEqual(box, comp.getBox())

        self.assertEqual(len(positions), comp.num_particles)
        self.assertEqual(len(positions), comp.getNP())


class TestLocalQlNear(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

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
        npt.assert_almost_equal(np.average(comp.Ql), 0.57452416, decimal=5)
        npt.assert_almost_equal(comp.Ql, comp.Ql[0])
        npt.assert_almost_equal(comp.getQl(), comp.Ql[0])

        comp.computeAve(positions)
        npt.assert_almost_equal(np.average(comp.Ql), 0.57452416, decimal=5)
        npt.assert_almost_equal(comp.ave_Ql, comp.ave_Ql[0])
        npt.assert_almost_equal(comp.getAveQl(), comp.ave_Ql[0])

        comp.computeNorm(positions)
        npt.assert_almost_equal(np.average(comp.Ql), 0.57452416, decimal=5)
        npt.assert_almost_equal(comp.norm_Ql, comp.norm_Ql[0])
        npt.assert_almost_equal(comp.getQlNorm(), comp.norm_Ql[0])

        comp.computeAveNorm(positions)
        npt.assert_almost_equal(np.average(comp.Ql), 0.57452416, decimal=5)
        npt.assert_almost_equal(comp.ave_norm_Ql, comp.ave_norm_Ql[0])
        npt.assert_almost_equal(comp.getQlAveNorm(), comp.ave_norm_Ql[0])

        self.assertEqual(box, comp.box)
        self.assertEqual(box, comp.getBox())

        self.assertEqual(len(positions), comp.num_particles)
        self.assertEqual(len(positions), comp.getNP())


if __name__ == '__main__':
    unittest.main()
