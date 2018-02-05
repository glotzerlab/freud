import numpy as np
import numpy.testing as npt
import freud
import unittest
import internal

class TestSolLiq(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = freud.order.SolLiq(box, 2, .7, 6, 6)
        comp.compute(positions)

        npt.assert_equal(comp.clusters.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = internal.make_fcc(4, 4, 4)

        comp = freud.order.SolLiq(box, 2, .7, 6, 6)

        comp.compute(positions)
        assert np.allclose(comp.largest_cluster_size, len(positions))

        comp.computeSolLiqNoNorm(positions)
        assert np.allclose(comp.largest_cluster_size, len(positions))

class TestSolLiqNear(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = freud.order.SolLiqNear(box, 2, .7, 6, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.clusters.shape[0], N)

    def test_identical_environments(self):
        (box, positions) = internal.make_fcc(4, 4, 4)

        comp = freud.order.SolLiqNear(box, 2, .7, 6, 6, 12)

        comp.compute(positions)
        assert np.allclose(comp.largest_cluster_size, len(positions))

        comp.computeSolLiqNoNorm(positions)
        assert np.allclose(comp.largest_cluster_size, len(positions))

if __name__ == '__main__':
    unittest.main()
