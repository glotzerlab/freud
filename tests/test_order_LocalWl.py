import numpy as np
import numpy.testing as npt
import freud
import unittest
import util

# Validated against manual calculation and pyboo
PERFECT_FCC_W6 = -0.0026260383340077588


class TestLocalWl(unittest.TestCase):
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = util.make_box_and_random_points(L, N)

        comp = freud.order.LocalWl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.Wl.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.LocalWl(box, 1.5, 6)

        with self.assertRaises(AttributeError):
            comp.num_particles
        with self.assertRaises(AttributeError):
            comp.Wl
        with self.assertRaises(AttributeError):
            comp.ave_Wl
        with self.assertRaises(AttributeError):
            comp.norm_Wl
        with self.assertRaises(AttributeError):
            comp.ave_norm_Wl

        comp.compute(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), PERFECT_FCC_W6, atol=1e-5))
        self.assertTrue(np.allclose(comp.Wl, comp.Wl[0]))

        with self.assertRaises(AttributeError):
            comp.ave_Wl
        with self.assertRaises(AttributeError):
            comp.norm_Wl
        with self.assertRaises(AttributeError):
            comp.ave_norm_Wl

        comp.computeAve(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), PERFECT_FCC_W6, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_Wl, comp.ave_Wl[0]))

        # Perturb one position to ensure exactly 13 particles' values change
        perturbed_positions = positions.copy()
        perturbed_positions[-1] += [0.1, 0, 0]
        comp.computeAve(perturbed_positions)
        self.assertEqual(
            sum(~np.isclose(np.real(comp.Wl), PERFECT_FCC_W6, rtol=1e-6)), 13)

        # More than 13 particles should change for Wl averaged over neighbors
        self.assertGreater(
            sum(~np.isclose(np.real(comp.ave_Wl), PERFECT_FCC_W6, rtol=1e-6)),
            13)

        with self.assertRaises(AttributeError):
            comp.norm_Wl
        with self.assertRaises(AttributeError):
            comp.ave_norm_Wl

        comp.computeNorm(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), PERFECT_FCC_W6, atol=1e-5))
        self.assertTrue(np.allclose(comp.norm_Wl, comp.norm_Wl[0]))

        with self.assertRaises(AttributeError):
            comp.ave_norm_Wl

        comp.computeAveNorm(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), PERFECT_FCC_W6, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_norm_Wl, comp.ave_norm_Wl[0]))

        self.assertEqual(box, comp.box)

        self.assertEqual(len(positions), comp.num_particles)

    def test_repr(self):
        box = freud.box.Box.cube(10)
        comp = freud.order.LocalWl(box, 1.5, 6)
        self.assertEqual(str(comp), str(eval(repr(comp))))

    def test_repr_png(self):
        (box, positions) = util.make_fcc(4, 4, 4)
        comp = freud.order.LocalWl(box, 1.5, 6)

        with self.assertRaises(AttributeError):
            comp.plot(mode="Wl")
        with self.assertRaises(AttributeError):
            comp.plot(mode="ave_Wl")
        with self.assertRaises(AttributeError):
            comp.plot(mode="ave_norm_Wl")
        with self.assertRaises(AttributeError):
            comp.plot(mode="norm_Wl")
        self.assertEqual(comp._repr_png_(), None)
        comp.compute(positions)
        comp.plot(mode="Wl")
        comp.computeAve(positions)
        comp.plot(mode="ave_Wl")
        comp.computeAveNorm(positions)
        comp.plot(mode="ave_norm_Wl")
        comp.computeNorm(positions)
        comp.plot(mode="norm_Wl")


class TestLocalWlNear(unittest.TestCase):
    def test_init_kwargs(self):
        """Ensure that keyword arguments are correctly accepted"""
        box = freud.box.Box.cube(10)
        # Use a really small cutoff to ensure that it is used as a soft cutoff
        comp = freud.order.LocalWlNear(box, 0.1, 6, kn=12)  # noqa: F841

    def test_shape(self):
        N = 1000
        L = 10

        box, positions = util.make_box_and_random_points(L, N)

        # Use a really small cutoff to ensure that it is used as a soft cutoff
        comp = freud.order.LocalWlNear(box, 0.1, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.Wl.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        # Use a really small cutoff to ensure that it is used as a soft cutoff
        comp = freud.order.LocalWlNear(box, 0.1, 6, 12)

        with self.assertRaises(AttributeError):
            comp.num_particles
        with self.assertRaises(AttributeError):
            comp.Wl
        with self.assertRaises(AttributeError):
            comp.ave_Wl
        with self.assertRaises(AttributeError):
            comp.norm_Wl
        with self.assertRaises(AttributeError):
            comp.ave_norm_Wl

        comp.compute(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), PERFECT_FCC_W6, atol=1e-5))
        self.assertTrue(np.allclose(comp.Wl, comp.Wl[0]))

        with self.assertRaises(AttributeError):
            comp.ave_Wl
        with self.assertRaises(AttributeError):
            comp.norm_Wl
        with self.assertRaises(AttributeError):
            comp.ave_norm_Wl

        comp.computeAve(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), PERFECT_FCC_W6, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_Wl, comp.ave_Wl[0]))

        with self.assertRaises(AttributeError):
            comp.norm_Wl
        with self.assertRaises(AttributeError):
            comp.ave_norm_Wl

        comp.computeNorm(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), PERFECT_FCC_W6, atol=1e-5))
        self.assertTrue(np.allclose(comp.norm_Wl, comp.norm_Wl[0]))

        with self.assertRaises(AttributeError):
            comp.ave_norm_Wl

        comp.computeAveNorm(positions)
        self.assertTrue(np.isclose(
            np.real(np.average(comp.Wl)), PERFECT_FCC_W6, atol=1e-5))
        self.assertTrue(np.allclose(comp.ave_norm_Wl, comp.ave_norm_Wl[0]))

        self.assertEqual(box, comp.box)

        self.assertEqual(len(positions), comp.num_particles)

    def test_repr(self):
        box = freud.box.Box.cube(10)
        # Use a really small cutoff to ensure that it is used as a soft cutoff
        comp = freud.order.LocalWlNear(box, 0.1, 6, 12)
        self.assertEqual(str(comp), str(eval(repr(comp))))

    def test_repr_png(self):
        (box, positions) = util.make_fcc(4, 4, 4)
        # Use a really small cutoff to ensure that it is used as a soft cutoff
        comp = freud.order.LocalWlNear(box, 0.1, 6, 12)

        with self.assertRaises(AttributeError):
            comp.plot(mode="Wl")
        with self.assertRaises(AttributeError):
            comp.plot(mode="ave_Wl")
        with self.assertRaises(AttributeError):
            comp.plot(mode="ave_norm_Wl")
        with self.assertRaises(AttributeError):
            comp.plot(mode="norm_Wl")
        self.assertEqual(comp._repr_png_(), None)
        comp.compute(positions)
        comp.plot(mode="Wl")
        comp.computeAve(positions)
        comp.plot(mode="ave_Wl")
        comp.computeAveNorm(positions)
        comp.plot(mode="ave_norm_Wl")
        comp.computeNorm(positions)
        comp.plot(mode="norm_Wl")


if __name__ == '__main__':
    unittest.main()
