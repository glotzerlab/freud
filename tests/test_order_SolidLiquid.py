import numpy.testing as npt
import freud
import unittest
import util

# Validated against manual calculation and pyboo
PERFECT_FCC_Q6 = 0.57452416


class TestSolidLiquid(unittest.TestCase):
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = util.make_box_and_random_points(L, N)

        comp = freud.order.SolidLiquid(6, Q_threshold=.7, S_threshold=6)
        comp.compute(box, positions, query_args=dict(r_max=2.0))

        npt.assert_equal(comp.cluster_idx.shape, (N,))

    def test_identical_environments(self):
        box, positions = util.make_fcc(4, 4, 4)

        comp_default = freud.order.SolidLiquid(
            6, Q_threshold=.7, S_threshold=6)
        comp_no_norm = freud.order.SolidLiquid(
            6, Q_threshold=.3, S_threshold=6, normalize_Q=False)

        for comp in (comp_default, comp_no_norm):
            for query_args in (dict(r_max=2.0), dict(num_neighbors=12)):
                comp.compute(box, positions, query_args=query_args)
                self.assertEqual(comp.largest_cluster_size, len(positions))
                self.assertEqual(len(comp.cluster_sizes), 1)
                self.assertEqual(comp.cluster_sizes[0], len(positions))
                npt.assert_array_equal(comp.num_connections, 12)

    def test_attribute_access(self):
        box, positions = util.make_fcc(4, 4, 4)
        sph_l = 6
        Q_threshold = 0.7
        S_threshold = 4
        normalize_Q = False

        comp = freud.order.SolidLiquid(
            sph_l, Q_threshold=Q_threshold, S_threshold=S_threshold,
            normalize_Q=normalize_Q)

        self.assertEqual(comp.l, sph_l)
        npt.assert_allclose(comp.Q_threshold, Q_threshold)
        npt.assert_allclose(comp.S_threshold, S_threshold)
        self.assertEqual(comp.normalize_Q, normalize_Q)

        with self.assertRaises(AttributeError):
            comp.largest_cluster_size
        with self.assertRaises(AttributeError):
            comp.cluster_sizes
        with self.assertRaises(AttributeError):
            comp.cluster_idx
        with self.assertRaises(AttributeError):
            comp.num_connections
        with self.assertRaises(AttributeError):
            comp.plot()

        comp.compute(box, positions, query_args=dict(r_max=2.0))

        comp.largest_cluster_size
        comp.cluster_sizes
        comp.cluster_idx
        comp.num_connections
        comp._repr_png_()

    def test_repr(self):
        comp = freud.order.SolidLiquid(6, Q_threshold=.7, S_threshold=6)
        self.assertEqual(str(comp), str(eval(repr(comp))))


if __name__ == '__main__':
    unittest.main()
