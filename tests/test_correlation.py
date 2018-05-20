import unittest
import numpy as np
import freud


class TestCorrelationFunction(unittest.TestCase):
    def test_type_check(self):
        boxlen = 10
        N = 500
        rmax, dr = 3, 0.1

        box = freud.box.Box.cube(boxlen)

        np.random.seed(0)
        points = np.asarray(np.random.uniform(-boxlen/2, boxlen/2, (N, 3)),
                            dtype=np.float32)
        values = np.ones((N,)) + 0j
        corrfun = freud.density.ComplexCF(rmax, dr)

        values = np.asarray(values, dtype=np.complex128)
        corrfun.compute(box, points, values, points, values.conj())
        assert True


if __name__ == '__main__':
    unittest.main()
