import numpy as np
import numpy.testing as npt
import freud
import unittest
from freud.errors import FreudDeprecationWarning
import warnings
import util


class TestSolLiq(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.SolLiq(box, 2, .7, 6, 6)
        comp.compute(positions)

        npt.assert_equal(comp.clusters.shape[0], N)

        self.assertEqual(box, comp.box)
        self.assertEqual(box, comp.getBox())

        box2 = freud.box.Box.cube(20)
        comp.setBox(box2)
        self.assertNotEqual(box, comp.box)
        self.assertEqual(box2, comp.box)

        comp.box = box
        self.assertNotEqual(box2, comp.box)
        self.assertEqual(box, comp.box)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.SolLiq(box, 2, .7, 6, 6)

        comp.compute(positions)
        self.assertTrue(np.allclose(comp.largest_cluster_size, len(positions)))
        self.assertEqual(len(comp.cluster_sizes), 1)

        comp.computeSolLiqNoNorm(positions)
        self.assertTrue(np.allclose(
            comp.getLargestClusterSize(), len(positions)))
        self.assertEqual(len(comp.getClusterSizes()), 1)

        comp.computeSolLiqVariant(positions)
        self.assertEqual(comp.largest_cluster_size, 1)


class TestSolLiqNear(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        np.random.seed(0)
        positions = np.random.uniform(-box.Lx/2, box.Lx/2,
                                      size=(N, 3)).astype(np.float32)

        comp = freud.order.SolLiqNear(box, 2, .7, 6, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.clusters.shape[0], N)

        self.assertEqual(box, comp.box)
        self.assertEqual(box, comp.getBox())

        box2 = freud.box.Box.cube(20)
        comp.setBox(box2)
        self.assertNotEqual(box, comp.box)
        self.assertEqual(box2, comp.box)

        comp.box = box
        self.assertNotEqual(box2, comp.box)
        self.assertEqual(box, comp.box)

    def test_identical_environments(self):
        (box, positions) = util.make_fcc(4, 4, 4)

        comp = freud.order.SolLiqNear(box, 2, .7, 6, 6, 12)

        comp.compute(positions)
        self.assertTrue(np.allclose(comp.largest_cluster_size, len(positions)))
        self.assertEqual(len(comp.cluster_sizes), 1)

        comp.computeSolLiqNoNorm(positions)
        self.assertTrue(np.allclose(
            comp.getLargestClusterSize(), len(positions)))
        self.assertEqual(len(comp.getClusterSizes()), 1)

        comp.computeSolLiqVariant(positions)
        self.assertEqual(comp.largest_cluster_size, 1)


if __name__ == '__main__':
    unittest.main()
