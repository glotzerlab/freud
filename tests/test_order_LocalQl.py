import numpy as np
import numpy.testing as npt
import freud
import unittest
import internal

class TestLocalQl(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalQl(box, 1.5, 6)
        comp.compute(positions)

        npt.assert_equal(comp.getQl().shape[0], N)

    def test_identical_environments(self):
        (box, positions) = internal.make_fcc(4, 4, 4)

        comp = freud.order.LocalQl(box, 1.5, 6)

        comp.compute(positions)
        assert np.allclose(comp.getQl(), comp.getQl()[0])

        comp.computeAve(positions)
        assert np.allclose(comp.getAveQl(), comp.getAveQl()[0])

        comp.computeNorm(positions)
        assert np.allclose(comp.getQlNorm(), comp.getQlNorm()[0])

        comp.computeAveNorm(positions)
        assert np.allclose(comp.getQlAveNorm(), comp.getQlAveNorm()[0])

class TestLocalQlNear(unittest.TestCase):
    def test_shape(self):
        N = 1000

        box = freud.box.Box.cube(10)
        positions = np.random.uniform(-box.getLx()/2, box.getLx()/2, size=(N, 3)).astype(np.float32)

        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)
        comp.compute(positions)

        npt.assert_equal(comp.getQl().shape[0], N)

    def test_identical_environments(self):
        (box, positions) = internal.make_fcc(4, 4, 4)

        comp = freud.order.LocalQlNear(box, 1.5, 6, 12)

        comp.compute(positions)
        assert np.allclose(comp.getQl(), comp.getQl()[0])

        comp.computeAve(positions)
        assert np.allclose(comp.getAveQl(), comp.getAveQl()[0])

        comp.computeNorm(positions)
        assert np.allclose(comp.getQlNorm(), comp.getQlNorm()[0])

        comp.computeAveNorm(positions)
        assert np.allclose(comp.getQlAveNorm(), comp.getQlAveNorm()[0])

if __name__ == '__main__':
    unittest.main()
