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
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalQl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.Ql.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalQl(box, 1.5, 6)

        comp.compute(positions)
        self.assertTrue(np.isclose(np.average(comp.Ql), 0.57452422, atol=1e-5))
        self.assertTrue(np.allclose(comp.Ql, comp.Ql[0]))
        self.assertTrue(np.allclose(comp.getQl(), comp.Ql[0]))

        comp.computeAve(positions)
        self.assertTrue(np.isclose(np.average(comp.Ql), 0.57452422, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_Ql, comp.ave_Ql[0]))
        self.assertTrue(np.allclose(comp.getAveQl(), comp.ave_Ql[0]))

        comp.computeNorm(positions)
        self.assertTrue(np.isclose(np.average(comp.Ql), 0.57452422, atol=1e-5))
        self.assertTrue(np.allclose(comp.norm_Ql, comp.norm_Ql[0]))
        self.assertTrue(np.allclose(comp.getQlNorm(), comp.norm_Ql[0]))

        comp.computeAveNorm(positions)
        self.assertTrue(np.isclose(np.average(comp.Ql), 0.57452422, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0]))
        self.assertTrue(np.allclose(comp.getQlAveNorm(), comp.ave_norm_Ql[0]))

        self.assertEqual(box, comp.box)
        self.assertEqual(box, comp.getBox())

        self.assertEqual(len(positions), comp.num_particles)
        self.assertEqual(len(positions), comp.getNP())


class TestLocalQlNear(unittest.TestCase):
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
        self.assertTrue(np.isclose(np.average(comp.Ql), 0.57452416, atol=1e-5))
        self.assertTrue(np.allclose(comp.Ql, comp.Ql[0]))
        self.assertTrue(np.allclose(comp.getQl(), comp.Ql[0]))

        comp.computeAve(positions)
        self.assertTrue(np.isclose(np.average(comp.Ql), 0.57452416, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_Ql, comp.ave_Ql[0]))
        self.assertTrue(np.allclose(comp.getAveQl(), comp.ave_Ql[0]))

        comp.computeNorm(positions)
        self.assertTrue(np.isclose(np.average(comp.Ql), 0.57452416, atol=1e-5))
        self.assertTrue(np.allclose(comp.norm_Ql, comp.norm_Ql[0]))
        self.assertTrue(np.allclose(comp.getQlNorm(), comp.norm_Ql[0]))

        comp.computeAveNorm(positions)
        self.assertTrue(np.isclose(np.average(comp.Ql), 0.57452416, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0]))
        self.assertTrue(np.allclose(comp.getQlAveNorm(), comp.ave_norm_Ql[0]))

        self.assertEqual(box, comp.box)
        self.assertEqual(box, comp.getBox())

        self.assertEqual(len(positions), comp.num_particles)
        self.assertEqual(len(positions), comp.getNP())


if __name__ == '__main__':
    unittest.main()
