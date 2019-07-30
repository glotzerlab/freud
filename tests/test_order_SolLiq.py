import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestSolLiq(unittest.TestCase):
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = util.make_box_and_random_points(L, N)

        comp = freud.order.SolLiq(box, 2, .7, 6, 6)
        comp.compute(positions)

        npt.assert_equal(comp.clusters.shape[0], N)

        self.assertEqual(box, comp.box)

        box2 = freud.box.Box.cube(20)
        comp.box = box2
        self.assertNotEqual(box, comp.box)
        self.assertEqual(box2, comp.box)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.SolLiq(box, 2, .7, 6, 6)

        comp.compute(positions)
        self.assertTrue(np.allclose(comp.largest_cluster_size, len(positions)))
        self.assertEqual(len(comp.cluster_sizes), 1)

        comp.computeSolLiqNoNorm(positions)
        self.assertTrue(np.allclose(comp.largest_cluster_size, len(positions)))
        self.assertEqual(len(comp.cluster_sizes), 1)

        comp.computeSolLiqVariant(positions)
        self.assertEqual(comp.largest_cluster_size, 1)

    def test_attribute_access(self):
        (box, positions) = util.make_fcc(4, 4, 4)
        func_names = ["compute", "computeSolLiqVariant", "computeSolLiqNoNorm"]
        for f in func_names:
            comp = freud.order.SolLiq(box, 2, .7, 6, 6)
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

            func = getattr(comp, f)
            func(positions)

            comp.largest_cluster_size
            comp.cluster_sizes
            comp.Ql_mi
            comp.clusters
            comp.num_connections
            comp.Ql_dot_ij
            comp.num_particles

    def test_repr(self):
        box = freud.box.Box.cube(10)
        comp = freud.order.SolLiq(box, 2, .7, 6, 6)
        self.assertEqual(str(comp), str(eval(repr(comp))))


class TestSolLiqNear(unittest.TestCase):
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = util.make_box_and_random_points(L, N)

        comp = freud.order.SolLiqNear(box, 2, .7, 6, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.clusters.shape[0], N)

        self.assertEqual(box, comp.box)

        box2 = freud.box.Box.cube(20)
        comp.box = box2
        self.assertNotEqual(box, comp.box)
        self.assertEqual(box2, comp.box)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.SolLiqNear(box, 2, .7, 6, 6, 12)

        comp.compute(positions)
        self.assertTrue(np.allclose(comp.largest_cluster_size, len(positions)))
        self.assertEqual(len(comp.cluster_sizes), 1)

        comp.computeSolLiqNoNorm(positions)
        self.assertTrue(np.allclose(comp.largest_cluster_size, len(positions)))
        self.assertEqual(len(comp.cluster_sizes), 1)

        comp.computeSolLiqVariant(positions)
        self.assertEqual(comp.largest_cluster_size, 1)

    def test_attribute_access(self):
        (box, positions) = util.make_fcc(4, 4, 4)
        func_names = ["compute", "computeSolLiqVariant", "computeSolLiqNoNorm"]
        for f in func_names:
            comp = freud.order.SolLiqNear(box, 2, .7, 6, 6, 12)
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

            func = getattr(comp, f)
            func(positions)

            comp.largest_cluster_size
            comp.cluster_sizes
            comp.Ql_mi
            comp.clusters
            comp.num_connections
            comp.Ql_dot_ij
            comp.num_particles

    def test_repr(self):
        box = freud.box.Box.cube(10)
        comp = freud.order.SolLiqNear(box, 2, .7, 6, 6, 12)
        self.assertEqual(str(comp), str(eval(repr(comp))))


if __name__ == '__main__':
    unittest.main()
