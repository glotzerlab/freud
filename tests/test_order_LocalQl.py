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
        positions.flags['WRITEABLE'] = False

        comp = freud.order.LocalQl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.Ql.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalQl(box, 1.5, 6)

        with self.assertRaises(AttributeError):
            comp.num_particles
        with self.assertRaises(AttributeError):
            comp.Ql
        with self.assertRaises(AttributeError):
            comp.ave_Ql
        with self.assertRaises(AttributeError):
            comp.norm_Ql
        with self.assertRaises(AttributeError):
            comp.ave_norm_Ql

        comp.compute(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452422, rtol=1e-6)
        npt.assert_allclose(comp.Ql, comp.Ql[0], rtol=1e-6)

        with self.assertRaises(AttributeError):
            comp.ave_Ql
        with self.assertRaises(AttributeError):
            comp.norm_Ql
        with self.assertRaises(AttributeError):
            comp.ave_norm_Ql

        comp.computeAve(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452422, rtol=1e-6)
        npt.assert_allclose(comp.ave_Ql, comp.ave_Ql[0], rtol=1e-6)

        with self.assertRaises(AttributeError):
            comp.norm_Ql
        with self.assertRaises(AttributeError):
            comp.ave_norm_Ql

        comp.computeNorm(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452422, rtol=1e-6)
        npt.assert_allclose(comp.norm_Ql, comp.norm_Ql[0], rtol=1e-6)

        with self.assertRaises(AttributeError):
            comp.ave_norm_Ql

        comp.computeAveNorm(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452422, rtol=1e-6)
        npt.assert_allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0], rtol=1e-6)

        self.assertEqual(box, comp.box)

        self.assertEqual(len(positions), comp.num_particles)

    def test_repr(self):
        box = freud.box.Box.cube(10)
        comp = freud.order.LocalQl(box, 1.5, 6)
        self.assertEqual(str(comp), str(eval(repr(comp))))

    def test_repr_png(self):
        (box, positions) = util.make_fcc(4, 4, 4)
        comp = freud.order.LocalQl(box, 1.5, 6)
        with self.assertRaises(AttributeError):
            comp.plot(mode="Ql")
        with self.assertRaises(AttributeError):
            comp.plot(mode="ave_Ql")
        with self.assertRaises(AttributeError):
            comp.plot(mode="ave_norm_Ql")
        with self.assertRaises(AttributeError):
            comp.plot(mode="norm_Ql")
        self.assertEqual(comp._repr_png_(), None)
        comp.compute(positions)
        comp.plot(mode="Ql")
        comp.computeAve(positions)
        comp.plot(mode="ave_Ql")
        comp.computeAveNorm(positions)
        comp.plot(mode="ave_norm_Ql")
        comp.computeNorm(positions)
        comp.plot(mode="norm_Ql")


class TestLocalQlNear(unittest.TestCase):
    def test_init_kwargs(self):
        """Ensure that keyword arguments are correctly accepted"""
        box = freud.box.Box.cube(10)
        comp = freud.order.LocalQlNear(box, 1.5, 6, num_neighbors=12)  # noqa

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

        with self.assertRaises(AttributeError):
            comp.num_particles
        with self.assertRaises(AttributeError):
            comp.Ql
        with self.assertRaises(AttributeError):
            comp.ave_Ql
        with self.assertRaises(AttributeError):
            comp.norm_Ql
        with self.assertRaises(AttributeError):
            comp.ave_norm_Ql

        comp.compute(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452416, rtol=1e-6)
        npt.assert_allclose(comp.Ql, comp.Ql[0], rtol=1e-6)

        with self.assertRaises(AttributeError):
            comp.ave_Ql
        with self.assertRaises(AttributeError):
            comp.norm_Ql
        with self.assertRaises(AttributeError):
            comp.ave_norm_Ql

        comp.computeAve(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452416, rtol=1e-6)
        npt.assert_allclose(comp.ave_Ql, comp.ave_Ql[0], rtol=1e-6)

        with self.assertRaises(AttributeError):
            comp.norm_Ql
        with self.assertRaises(AttributeError):
            comp.ave_norm_Ql

        comp.computeNorm(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452416, rtol=1e-6)
        npt.assert_allclose(comp.norm_Ql, comp.norm_Ql[0], rtol=1e-6)

        with self.assertRaises(AttributeError):
            comp.ave_norm_Ql

        comp.computeAveNorm(positions)
        npt.assert_allclose(np.average(comp.Ql), 0.57452416, rtol=1e-6)
        npt.assert_allclose(comp.ave_norm_Ql, comp.ave_norm_Ql[0], rtol=1e-6)

        self.assertEqual(box, comp.box)

        self.assertEqual(len(positions), comp.num_particles)

    def test_repr(self):
        box = freud.box.Box.cube(10)
        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)
        self.assertEqual(str(comp), str(eval(repr(comp))))

    def test_repr_png(self):
        (box, positions) = util.make_fcc(4, 4, 4)
        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)
        with self.assertRaises(AttributeError):
            comp.plot(mode="Ql")
        with self.assertRaises(AttributeError):
            comp.plot(mode="ave_Ql")
        with self.assertRaises(AttributeError):
            comp.plot(mode="ave_norm_Ql")
        with self.assertRaises(AttributeError):
            comp.plot(mode="norm_Ql")
        self.assertEqual(comp._repr_png_(), None)
        comp.compute(positions)
        comp.plot(mode="Ql")
        comp.computeAve(positions)
        comp.plot(mode="ave_Ql")
        comp.computeAveNorm(positions)
        comp.plot(mode="ave_norm_Ql")
        comp.computeNorm(positions)
        comp.plot(mode="norm_Ql")


if __name__ == '__main__':
    unittest.main()
