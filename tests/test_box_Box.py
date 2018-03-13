from freud import box as bx
import numpy as np
import numpy.testing as npt
import warnings
import unittest

class TestBox(unittest.TestCase):

    def setUp(self):
        # We ignore warnings for test_2_dimensional
        warnings.simplefilter("ignore")

    def test_getLength(self):
        box = bx.Box(2, 4, 5, 1, 0, 0)

        npt.assert_almost_equal(box.Lx, 2, decimal=2, err_msg="LxFail")
        npt.assert_almost_equal(box.Ly, 4, decimal=2, err_msg="LyFail")
        npt.assert_almost_equal(box.Lz, 5, decimal=2, err_msg="LzFail")
        npt.assert_almost_equal(box.L, [2, 4, 5], decimal=2, err_msg="LFail")
        npt.assert_almost_equal(box.Linv, [0.5, 0.25, 0.2], decimal=2,
                                err_msg="LinvFail")

    def test_setLength(self):
        # Make sure we can change the lengths of the box after its creation
        box = bx.Box(1, 2, 3, 1, 0, 0)

        box.Lx = 4
        box.Ly = 5
        box.Lz = 6

        npt.assert_almost_equal(box.Lx, 4, decimal=2, err_msg="SetLxFail")
        npt.assert_almost_equal(box.Ly, 5, decimal=2, err_msg="SetLyFail")
        npt.assert_almost_equal(box.Lz, 6, decimal=2, err_msg="SetLzFail")

        box.L = [7, 8, 9]
        npt.assert_almost_equal(box.L, [7, 8, 9], decimal=2,
                                err_msg="SetLFail")

    def test_TiltFactor(self):
        box = bx.Box(2, 2, 2, 1, 2, 3)

        npt.assert_almost_equal(box.xy, 1, decimal=2, err_msg="TiltXYFail")
        npt.assert_almost_equal(box.xz, 2, decimal=2, err_msg="TiltXZFail")
        npt.assert_almost_equal(box.yz, 3, decimal=2, err_msg="TiltYZFail")

    def test_BoxVolume(self):
        box3d = bx.Box(2, 2, 2, 1, 0, 0)
        box2d = bx.Box(2, 2, 0, 0, 0, 0, is2D=True)

        npt.assert_almost_equal(box3d.volume, 8, decimal=2,
                                err_msg="Volume3DFail")
        npt.assert_almost_equal(box2d.volume, 4, decimal=2,
                                err_msg="Volume2DFail")

    def test_WrapSingleParticle(self):
        box = bx.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([0, -1, -1], dtype=np.float32)
        box.wrap(testpoints)

        npt.assert_almost_equal(testpoints[0], -2, decimal=2,
                                err_msg="WrapFail")

    def test_WrapMultipleParticles(self):
        box = bx.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([[0,  -1, -1],
                               [0, 0.5,  0]], dtype=np.float32)
        box.wrap(testpoints)

        npt.assert_almost_equal(testpoints[0,0], -2, decimal=2,
                                err_msg="WrapFail")

    def test_WrapMultipleImages(self):
        box = bx.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([[10, -5, -5],
                               [0, 0.5, 0]], dtype=np.float32)
        box.wrap(testpoints)

        npt.assert_almost_equal(testpoints[0,0], -2, decimal=2,
                                err_msg="WrapFail")

    def test_unwrap(self):
        box = bx.Box(2, 2, 2, 1, 0, 0)
        testpoints = np.array([[0,  -1, -1],
                               [0, 0.5,  0]], dtype=np.float32)
        imgs = np.array([[1,0,0],
                         [1,1,0]], dtype=np.int32)
        box.unwrap(testpoints, imgs)

        npt.assert_almost_equal(testpoints[0,0], 2, decimal=2,
                                err_msg="WrapFail")

    def test_coordinates(self):
        box = bx.Box(2, 4, 6, 0, 0, 0)
        rel_coords = [0.5, 0.5, 0.5]
        abs_coords = box.getCoordinates(rel_coords)
        self.assertTrue(np.isclose(abs_coords, [0, 0, 0]).all())
        rel_coords = [1, 1, 1]
        abs_coords = box.getCoordinates(rel_coords)
        self.assertTrue(np.isclose(abs_coords, [1, 2, 3]).all())

    def test_periodic(self):
        box = bx.Box(1, 2, 3, 0, 0, 0)
        assert box.periodic == [True, True, True]
        box.periodic = [False, False, False]
        assert box.periodic == [False, False, False]

    def test_equal(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box(2, 2, 2, 1, 0, 0)
        self.assertEqual(box, box)
        self.assertNotEqual(box, box2)

    def test_str(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        self.assertEqual(str(box), str(box2))

    def test_dict(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)

        class BoxTuple(object):
            def __init__(self, box_dict):
                self.__dict__.update(box_dict)
        box2 = bx.Box.from_box(BoxTuple(box.to_dict()))
        self.assertEqual(box, box2)

    def test_tuple(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box.from_box(box.to_tuple())
        self.assertEqual(box, box2)

    def test_from_box(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box.from_box(box)
        self.assertEqual(box, box2)

    def test_matrix(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box.from_matrix(box.to_matrix())
        self.assertTrue(np.isclose(box.to_matrix(), box2.to_matrix()).all())

    def test_2_dimensional(self):
        box = bx.Box.square(L=1)
        # Setting Lz for a 2D box throws a warning that we hide with setUp()
        box.Lz = 1.0
        self.assertEqual(box.Lz, 0.0)
        self.assertTrue(box.dimensions == 2)
        box.dimensions = 3
        self.assertEqual(box.Lz, 0.0)
        self.assertTrue(box.dimensions == 3)
        box.Lz = 1.0
        self.assertEqual(box.Lz, 1.0)

    def test_cube(self):
        L = 10.0
        cube = bx.Box.cube(L=L)
        self.assertEqual(cube.Lx, L)
        self.assertEqual(cube.Ly, L)
        self.assertEqual(cube.Lz, L)
        self.assertEqual(cube.xy, 0)
        self.assertEqual(cube.xz, 0)
        self.assertEqual(cube.yz, 0)
        self.assertEqual(cube.dimensions, 3)

    def test_square(self):
        L = 10.0
        square = bx.Box.square(L=L)
        self.assertEqual(square.Lx, L)
        self.assertEqual(square.Ly, L)
        self.assertEqual(square.Lz, 0)
        self.assertEqual(square.xy, 0)
        self.assertEqual(square.xz, 0)
        self.assertEqual(square.yz, 0)
        self.assertEqual(square.dimensions, 2)



if __name__ == '__main__':
    unittest.main()
