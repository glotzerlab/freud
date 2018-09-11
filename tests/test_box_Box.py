import numpy as np
import numpy.testing as npt
from freud import box as bx
from collections import namedtuple
import unittest
import warnings


class TestBox(unittest.TestCase):

    def setUp(self):
        # We ignore warnings for test_2_dimensional
        warnings.simplefilter("ignore")

    def test_construct(self):
        """Test correct behavior for various constructor signatures"""
        with self.assertRaises(ValueError):
            bx.Box()

        with self.assertRaises(ValueError):
            bx.Box(0, 0)

        with self.assertRaises(ValueError):
            bx.Box(1, 2, is2D=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bx.Box(1, 2, 3, is2D=True)
            self.assertTrue(len(w) == 1)

        box = bx.Box(1, 2)
        self.assertTrue(box.dimensions == 2)

    def test_get_length(self):
        box = bx.Box(2, 4, 5, 1, 0, 0)

        npt.assert_almost_equal(box.Lx, 2, decimal=2, err_msg="LxFail")
        npt.assert_almost_equal(box.Ly, 4, decimal=2, err_msg="LyFail")
        npt.assert_almost_equal(box.Lz, 5, decimal=2, err_msg="LzFail")
        npt.assert_almost_equal(box.L, [2, 4, 5], decimal=2, err_msg="LFail")
        npt.assert_almost_equal(box.Linv, [0.5, 0.25, 0.2], decimal=2,
                                err_msg="LinvFail")

        npt.assert_equal(box.Lx, box.getLx())
        npt.assert_equal(box.Ly, box.getLy())
        npt.assert_equal(box.Lz, box.getLz())
        npt.assert_equal(box.L, box.getL())
        npt.assert_equal(box.Linv, box.getLinv())

    def test_set_length(self):
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

        box.setL(10)
        npt.assert_almost_equal(box.L, [10, 10, 10], decimal=2,
                                err_msg="SetLFail")

        box.setL([1, 2, 3])
        npt.assert_almost_equal(box.L, [1, 2, 3], decimal=2,
                                err_msg="SetLFail")

        with self.assertRaises(ValueError):
            box.setL([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            box.setL([1, 2])

    def test_get_tilt_factor(self):
        box = bx.Box(2, 2, 2, 1, 2, 3)

        npt.assert_almost_equal(box.xy, 1, decimal=2, err_msg="TiltXYFail")
        npt.assert_almost_equal(box.xz, 2, decimal=2, err_msg="TiltXZFail")
        npt.assert_almost_equal(box.yz, 3, decimal=2, err_msg="TiltYZFail")
        npt.assert_almost_equal(box.getTiltFactorXY(), 1, decimal=2,
                                err_msg="TiltXYFail")
        npt.assert_almost_equal(box.getTiltFactorXZ(), 2, decimal=2,
                                err_msg="TiltXZFail")
        npt.assert_almost_equal(box.getTiltFactorYZ(), 3, decimal=2,
                                err_msg="TiltYZFail")

    def test_box_volume(self):
        box3d = bx.Box(2, 2, 2, 1, 0, 0)
        box2d = bx.Box(2, 2, 0, 0, 0, 0, is2D=True)

        npt.assert_almost_equal(box3d.volume, 8, decimal=2,
                                err_msg="Volume3DFail")
        npt.assert_almost_equal(box3d.getVolume(), 8, decimal=2,
                                err_msg="Volume3DFail")
        npt.assert_almost_equal(box2d.volume, 4, decimal=2,
                                err_msg="Volume2DFail")
        npt.assert_almost_equal(box2d.getVolume(), 4, decimal=2,
                                err_msg="Volume2DFail")

    def test_wrap_single_particle(self):
        box = bx.Box(2, 2, 2, 1, 0, 0)

        testpoints = [0, -1, -1]
        npt.assert_almost_equal(box.wrap(testpoints)[0], -2, decimal=2,
                                err_msg="WrapFail")

        testpoints = np.array(testpoints)
        npt.assert_almost_equal(box.wrap(testpoints)[0], -2, decimal=2,
                                err_msg="WrapFail")

        with self.assertRaises(ValueError):
            box.wrap([1, 2])

    def test_wrap_multiple_particles(self):
        box = bx.Box(2, 2, 2, 1, 0, 0)

        testpoints = [[0, -1, -1], [0, 0.5, 0]]
        npt.assert_almost_equal(box.wrap(testpoints)[0, 0], -2, decimal=2,
                                err_msg="WrapFail")

        testpoints = np.array(testpoints)
        npt.assert_almost_equal(box.wrap(testpoints)[0, 0], -2, decimal=2,
                                err_msg="WrapFail")

    def test_wrap_multiple_images(self):
        box = bx.Box(2, 2, 2, 1, 0, 0)

        testpoints = [[10, -5, -5], [0, 0.5, 0]]
        npt.assert_almost_equal(box.wrap(testpoints)[0, 0], -2, decimal=2,
                                err_msg="WrapFail")

        testpoints = np.array(testpoints)
        npt.assert_almost_equal(box.wrap(testpoints)[0, 0], -2, decimal=2,
                                err_msg="WrapFail")

    def test_unwrap(self):
        box = bx.Box(2, 2, 2, 1, 0, 0)

        testpoints = [0, -1, -1]
        imgs = [1, 0, 0]
        npt.assert_almost_equal(box.unwrap(testpoints, imgs), [2, -1, -1],
                                decimal=2, err_msg="WrapFail")

        testpoints = [[0, -1, -1], [0, 0.5, 0]]
        imgs = [[1, 0, 0], [1, 1, 0]]
        npt.assert_almost_equal(box.unwrap(testpoints, imgs)[0, 0], 2,
                                decimal=2, err_msg="WrapFail")

        testpoints = np.array(testpoints)
        imgs = np.array(imgs)
        npt.assert_almost_equal(box.unwrap(testpoints, imgs)[0, 0], 2,
                                decimal=2, err_msg="WrapFail")

        with self.assertRaises(ValueError):
            box.unwrap(testpoints, imgs[..., np.newaxis])

        with self.assertRaises(ValueError):
            box.unwrap(testpoints[:, :2], imgs)

    def test_images(self):
        box = bx.Box(2, 2, 2, 0, 0, 0)
        testpoints = np.array([[50, 40, 30],
                               [-10, 0, 0]])
        testimages = np.array([box.getImage(vec) for vec in testpoints])
        npt.assert_equal(testimages,
                         np.array([[25, 20, 15],
                                   [-5, 0, 0]]),
                         err_msg="ImageFail")

    def test_coordinates(self):
        box = bx.Box(2, 2, 2)
        f_point = np.array([0.5, 0.25, 0.75])
        point = np.array([0, -0.5, 0.5])

        self.assertTrue(box.getCoordinates(f_point),
                        box.makeCoordinates(f_point))
        npt.assert_equal(box.makeCoordinates(f_point),
                         point)
        npt.assert_equal(box.makeFraction(point),
                         f_point)

        dims = np.array([2, 2, 2])
        for i in range(10):
            npt.assert_array_equal(box.getImage(dims*i), [i, i, i])

    def test_vectors(self):
        """Test getting lattice vectors"""
        b_list = [1, 2, 3, 0.1, 0.2, 0.3]
        Lx, Ly, Lz, xy, xz, yz = b_list
        box = bx.Box.from_box(b_list)
        self.assertEqual(
            box.getLatticeVector(0),
            [Lx, 0, 0]
        )
        npt.assert_array_almost_equal(
            box.getLatticeVector(1),
            [xy*Ly, Ly, 0]
        )
        npt.assert_array_almost_equal(
            box.getLatticeVector(2),
            [xz*Lz, yz*Lz, Lz]
        )

    def test_periodic(self):
        box = bx.Box(1, 2, 3, 0, 0, 0)
        false = [False, False, False]
        true = [True, True, True]
        self.assertEqual(box.periodic, true)
        self.assertEqual(box.getPeriodic(), true)
        self.assertTrue(box.periodic_x)
        self.assertTrue(box.periodic_y)
        self.assertTrue(box.periodic_z)
        self.assertTrue(box.getPeriodicX())
        self.assertTrue(box.getPeriodicY())
        self.assertTrue(box.getPeriodicZ())

        box.periodic = false
        self.assertEqual(box.periodic, false)
        self.assertEqual(box.getPeriodic(), false)
        self.assertFalse(box.periodic_x)
        self.assertFalse(box.periodic_y)
        self.assertFalse(box.periodic_z)
        self.assertFalse(box.getPeriodicX())
        self.assertFalse(box.getPeriodicY())
        self.assertFalse(box.getPeriodicZ())

        box.setPeriodic(*true)
        self.assertEqual(box.periodic, true)

        box.periodic_x = False
        box.periodic_y = False
        box.periodic_z = False
        self.assertEqual(box.periodic_x, False)
        self.assertEqual(box.periodic_y, False)
        self.assertEqual(box.periodic_z, False)

        box.periodic = True
        self.assertEqual(box.periodic, true)

        box.setPeriodicX(False)
        box.setPeriodicY(False)
        box.setPeriodicZ(False)
        self.assertEqual(box.periodic_x, False)
        self.assertEqual(box.periodic_y, False)
        self.assertEqual(box.periodic_z, False)

    def test_equal(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box(2, 2, 2, 1, 0, 0)
        self.assertEqual(box, box)
        self.assertNotEqual(box, box2)

    def test_str(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        self.assertEqual(str(box), str(box2))

    def test_to_dict(self):
        """Test converting box to dict"""
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = box.to_dict()
        box_dict = {'Lx': 2, 'Ly': 2, 'Lz': 2, 'xy': 1, 'xz': 0.5, 'yz': 0.1}
        for k in box_dict:
            npt.assert_almost_equal(box_dict[k], box2[k])

    def test_tuple(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box.from_box(box.to_tuple())
        self.assertEqual(box, box2)

    def test_from_box(self):
        """Test various methods of initializing a box"""
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box.from_box(box)
        self.assertEqual(box, box2)

        box_dict = {'Lx': 2, 'Ly': 2, 'Lz': 2, 'xy': 1, 'xz': 0.5, 'yz': 0.1}
        box3 = bx.Box.from_box(box_dict)
        self.assertEqual(box, box3)

        with self.assertRaises(ValueError):
            box_dict['dimensions'] = 3
            bx.Box.from_box(box_dict, 2)

        BoxTuple = namedtuple('BoxTuple',
                              ['Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz',
                               'dimensions'])
        box4 = bx.Box.from_box(BoxTuple(2, 2, 2, 1, 0.5, 0.1, 3))
        self.assertEqual(box, box4)

        with self.assertRaises(ValueError):
            bx.Box.from_box(BoxTuple(2, 2, 2, 1, 0.5, 0.1, 2), 3)

        box5 = bx.Box.from_box([2, 2, 2, 1, 0.5, 0.1])
        self.assertEqual(box, box5)

        box6 = bx.Box.from_box(np.array([2, 2, 2, 1, 0.5, 0.1]))
        self.assertEqual(box, box6)

        with self.assertRaises(ValueError):
            bx.Box.from_box([2, 2, 2, 1, 0.5])

        box7 = bx.Box.from_matrix(box.to_matrix())
        self.assertTrue(np.isclose(box.to_matrix(), box7.to_matrix()).all())

    def test_matrix(self):
        box = bx.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = bx.Box.from_matrix(box.to_matrix())
        self.assertTrue(np.isclose(box.to_matrix(), box2.to_matrix()).all())

        box3 = bx.Box(2, 2, 0, 0.5, 0, 0)
        box4 = bx.Box.from_matrix(box3.to_matrix())
        self.assertTrue(np.isclose(box3.to_matrix(), box4.to_matrix()).all())

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

    def test_multiply(self):
        box = bx.Box(2, 3, 4, 1, 0.5, 0.1)
        box2 = box * 2
        self.assertTrue(np.isclose(box2.Lx, 4))
        self.assertTrue(np.isclose(box2.Ly, 6))
        self.assertTrue(np.isclose(box2.Lz, 8))
        self.assertTrue(np.isclose(box2.xy, 1))
        self.assertTrue(np.isclose(box2.xz, 0.5))
        self.assertTrue(np.isclose(box2.yz, 0.1))
        box3 = 2 * box
        self.assertEqual(box2, box3)


if __name__ == '__main__':
    unittest.main()
