import numpy
import numpy.testing as npt
from freud import trajectory, pairing
import unittest

class TestSameSide(unittest.TestCase):
    def test_sameside(self):
        a = numpy.array([-1.0, 0.0, 0.0], dtype=numpy.float32)
        b = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        r = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float32)
        p = numpy.array([-1.0, 10.0, 0.0], dtype=numpy.float32)
        comp = pairing.pairing(trajectory.Box(10.0), 1.0, 0.1)
        test = comp._sameSide(a, b, r, p)
        npt.assert_equal(test, True)
        r = numpy.array([0.0, -1.0, 0.0], dtype=numpy.float32)
        test = comp._sameSide(a, b, r, p)
        npt.assert_equal(test, False)
        r = numpy.array([-2.0, 0.0, 0.0], dtype=numpy.float32)
        test = comp._sameSide(a, b, r, p)
        npt.assert_equal(test, True)
        r = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32)
        test = comp._sameSide(a, b, r, p)
        npt.assert_equal(test, True)

class TestIsInside(unittest.TestCase):
    def test_isinside(self):
        a = numpy.array([-1.0, -1.0], dtype=numpy.float32)
        b = numpy.array([1.0, -1.0], dtype=numpy.float32)
        c = numpy.array([0.0, 1.0], dtype=numpy.float32)
        t = numpy.array([a, b, c], dtype=numpy.float32)
        p = numpy.array([0.0, 0.0], dtype=numpy.float32)
        comp = pairing.pairing(trajectory.Box(10.0), 1.0, 0.1)
        test = comp._isInside(t, p)
        npt.assert_equal(test, True)
        p = numpy.array([-1.0, -1.0], dtype=numpy.float32)
        test = comp._isInside(t, p)
        npt.assert_equal(test, True)
        p = numpy.array([-10.0, -10.0], dtype=numpy.float32)
        test = comp._isInside(t, p)
        npt.assert_equal(test, False)

class TestCross(unittest.TestCase):
    def test_cross(self):
        v1 = numpy.array([1.0, 1.0, 0.0], dtype=numpy.float32)
        v2 = numpy.array([1.0, 1.0, 0.0], dtype=numpy.float32)
        comp = pairing.pairing(trajectory.Box(10.0), 1.0, 0.1)
        v = comp._dot3(v1, v2)
        npt.assert_array_equal(v, 2.0)

class TestDot(unittest.TestCase):
    def test_dot(self):
        v1 = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        v2 = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float32)
        v = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32)
        ans = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)
        comp = pairing.pairing(trajectory.Box(10.0), 1.0, 0.1)
        comp._cross(v, v1, v2)
        npt.assert_array_equal(v, ans)

class TestMatRot(unittest.TestCase):
    def test_mat_rot(self):
        p = numpy.array([1.0, 0.0], dtype=numpy.float32)
        p_rot = numpy.array([0.0, 0.0], dtype=numpy.float32)
        ans = numpy.array([0.0, 1.0], dtype=numpy.float32)
        angle = float(numpy.pi/2.0)
        comp = pairing.pairing(trajectory.Box(10.0), 1.0, 0.1)
        comp._mat_rot(p_rot, p, angle)
        npt.assert_array_almost_equal(p_rot, ans, decimal = 3)

class TestIntoLocal(unittest.TestCase):
    def test_into_local(self):
        verts = numpy.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=numpy.float32)
        local = numpy.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=numpy.float32)
        p1 = numpy.array([1.0, 1.0], dtype=numpy.float32)
        p2 = numpy.array([1.0, -1.0], dtype=numpy.float32)
        a1 = float(numpy.pi/8.0)
        a2 = float(-numpy.pi/8.0)
        comp = pairing.pairing(trajectory.Box(10.0), 1.0, 0.1)
        nv = len(verts)
        for i in range(nv):
            comp._into_local(local[i], p1, p2, verts[i], a1, a2)
        ans = numpy.array([[-2.17958045, -1.84775901], [-0.76536685, -3.26197267], [0.64884681, -1.84775901], [-0.76536685, -0.43354535]], dtype=numpy.float32)
        # ans = numpy.array([[ -1.41421366e+00,  -2.00000000e+00], [ -1.38695135e-08,  -3.41421366e+00], [  1.41421366e+00,  -2.00000000e+00], [  1.38695135e-08,  -5.85786343e-01]], dtype=numpy.float32)
        npt.assert_array_almost_equal(local, ans, decimal = 3)

if __name__ == '__main__':
    unittest.main()
