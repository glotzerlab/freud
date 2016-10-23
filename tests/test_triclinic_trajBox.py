import freud
import numpy as np
import numpy.testing as npt
import unittest

class TestBox(unittest.TestCase):
    def test_BoxLength(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)

        Lx = box.getLx()
        Ly = box.getLy()
        Lz = box.getLz()

        npt.assert_almost_equal(Lx, 2, decimal=2, err_msg="LxFail")
        npt.assert_almost_equal(Ly, 2, decimal=2, err_msg="LyFail")
        npt.assert_almost_equal(Lz, 2, decimal=2, err_msg="LzFail")

    def test_TiltFactor(self):
        box = freud.box.Box(2,	2, 2, 1, 0, 0);

        tiltxy = box.getTiltFactorXY()
        tiltxz = box.getTiltFactorXZ()
        tiltyz = box.getTiltFactorYZ()

        npt.assert_almost_equal(tiltxy, 1, decimal=2, err_msg="TiltXYfail")
        npt.assert_almost_equal(tiltxz, 0, decimal=2, err_msg="TiltXZfail")
        npt.assert_almost_equal(tiltyz, 0, decimal=2, err_msg="TiltYZfail")

    def test_BoxVolume(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)

        volume = box.getVolume()

        npt.assert_almost_equal(volume, 8, decimal=2, err_msg="VolumnFail")

    def test_WrapSingleParticle(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([0, -1, -1], dtype=np.float32)
        box.wrap(testpoints)

        npt.assert_almost_equal(testpoints[0], -2, decimal=2, err_msg="WrapFail")

    def test_WrapMultipleParticles(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([[0, -1, -1],
                               [0, 0.5, 0]], dtype=np.float32)
        box.wrap(testpoints)

        npt.assert_almost_equal(testpoints[0,0], -2, decimal=2, err_msg="WrapFail")

    def test_WrapMultipleImages(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([[10, -5, -5],
                               [0, 0.5, 0]], dtype=np.float32)
        box.wrap(testpoints)

        npt.assert_almost_equal(testpoints[0,0], -2, decimal=2, err_msg="WrapFail")

    def test_unwrap(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([[0, -1, -1],
                               [0, 0.5, 0]], dtype=np.float32)
        imgs = np.array([[1,0,0],
                         [1,1,0]], dtype=np.int32)
        box.unwrap(testpoints, imgs)

        npt.assert_almost_equal(testpoints[0,0], 2, decimal=2, err_msg="WrapFail")

if __name__ == '__main__':
    unittest.main()
