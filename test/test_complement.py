import numpy
import numpy.testing as npt
from freud import trajectory, complement
import unittest

class TestSameSide(unittest.TestCase):
    def test_sameside(self):
        a = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)
        b = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        r = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float32)
        p = numpy.array([-1.0, 10.0, 0.0], dtype=numpy.float32)
        comp = complement.complement(trajectory.Box(10.0), 1.0, 0.1)
        test = comp._sameSide(a, b, r, p)
        npt.assert_equal(test, True)
        
class TestIsInside(unittest.TestCase):
    def test_isinside(self):
        a = numpy.array([-1.0, -1.0], dtype=numpy.float32)
        b = numpy.array([1.0, -1.0], dtype=numpy.float32)
        c = numpy.array([0.0, 1.0], dtype=numpy.float32)
        t = numpy.array([a, b, c], dtype=numpy.float32)
        p = numpy.array([0.0, 0.0], dtype=numpy.float32)
        comp = complement.complement(trajectory.Box(10.0), 1.0, 0.1)
        test = comp._isInside(t, p)
        npt.assert_equal(test, True)

if __name__ == '__main__':
    unittest.main()
