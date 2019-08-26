import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestSolidLiquid(unittest.TestCase):
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = util.make_box_and_random_points(L, N)

        comp = freud.order.SolidLiquid(6, Qthreshold=.7, Sthreshold=6)
        comp.compute(positions)

        npt.assert_equal(comp.clusters.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.SolidLiquid(6, Qthreshold=.7, Sthreshold=6)

        comp.compute(positions)
        # TODO: use both r_max=1.5 and num_neighbors=12 to test this
        self.assertTrue(np.allclose(comp.largest_cluster_size, len(positions)))
        self.assertEqual(len(comp.cluster_sizes), 1)

        comp.computeSolidLiquidNoNorm(positions)
        self.assertTrue(np.allclose(comp.largest_cluster_size, len(positions)))
        self.assertEqual(len(comp.cluster_sizes), 1)

        comp.computeSolidLiquidVariant(positions)
        self.assertEqual(comp.largest_cluster_size, 1)

    def test_attribute_access(self):
        (box, positions) = util.make_fcc(4, 4, 4)
        comp = freud.order.SolidLiquid(6, Qthreshold=.7, Sthreshold=6)

        with self.assertRaises(AttributeError):
            comp.largest_cluster_size
        with self.assertRaises(AttributeError):
            comp.cluster_sizes
        with self.assertRaises(AttributeError):
            comp.Ql_mi
        with self.assertRaises(AttributeError):
            comp.clusters
        with self.assertRaises(AttributeError):
            comp.num_connections
        with self.assertRaises(AttributeError):
            comp.Ql_dot_ij
        with self.assertRaises(AttributeError):
            comp.num_particles

        comp.compute(box, positions)

        comp.largest_cluster_size
        comp.cluster_sizes
        comp.Ql_mi
        comp.clusters
        comp.num_connections
        comp.Ql_dot_ij
        comp.num_particles

    def test_repr(self):
        comp = freud.order.SolidLiquid(6, Qthreshold=.7, Sthreshold=6)
        self.assertEqual(str(comp), str(eval(repr(comp))))


if __name__ == '__main__':
    unittest.main()
