import numpy as np
import numpy.testing as npt
from freud import common
import unittest
from collections import namedtuple


class TestCommon(unittest.TestCase):
    def test_convert_array(self):
        # create array
        x = np.arange(100)
        # create a non-contiguous array
        y = x.reshape(10, 10).T
        # run through convert
        # first check to make sure it passes with default
        z = common.convert_array(y, 2)
        npt.assert_equal(y.dtype, x.dtype)
        # now change type
        z = common.convert_array(y, 2, dtype=np.float32)
        npt.assert_equal(z.dtype, np.float32)
        # now make contiguous
        npt.assert_equal(y.flags.contiguous, False)
        z = common.convert_array(y, 2, contiguous=True)
        npt.assert_equal(z.flags.contiguous, True)
        # test the dim_message
        try:
            z = common.convert_array(
                y, 1, dtype=np.float32, contiguous=True,
                dim_message="ref_points must be a 2 dimensional array")
        except TypeError:
            npt.assert_equal(True, True)

    def test_convert_matrix_box(self):
        matrix_box = np.array([[1, 2, 3],
                               [0, 2, 3],
                               [0, 0, 3]])
        box = common.convert_box(matrix_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 1, rtol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 1, rtol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 1, rtol=1e-6, err_msg="TiltYZFail")
        self.assertTrue(box.dimensions == 3)

    def test_convert_tuple_box(self):
        TupleBox = namedtuple('TupleBox', ['Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz'])
        tuple_box = TupleBox(1, 2, 3, 4, 5, 6)
        box = common.convert_box(tuple_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 4, rtol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 5, rtol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 6, rtol=1e-6, err_msg="TiltYZFail")
        self.assertTrue(box.dimensions == 3)

    def test_convert_dict_box(self):
        dict_box = dict(Lx=1, Ly=2, Lz=3, xy=4, xz=5, yz=6)
        box = common.convert_box(dict_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 4, rtol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 5, rtol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 6, rtol=1e-6, err_msg="TiltYZFail")
        self.assertTrue(box.dimensions == 3)

    def test_convert_array_len_2_box(self):
        array_box = [1, 2]
        box = common.convert_box(array_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 0, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 0, atol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 0, atol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 0, atol=1e-6, err_msg="TiltYZFail")
        self.assertTrue(box.dimensions == 2)

    def test_convert_array_len_3_box(self):
        array_box = [1, 2, 3]
        box = common.convert_box(array_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 0, atol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 0, atol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 0, atol=1e-6, err_msg="TiltYZFail")
        self.assertTrue(box.dimensions == 3)

    def test_convert_array_len_6_box(self):
        array_box = [1, 2, 3, 4, 5, 6]
        box = common.convert_box(array_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 4, rtol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 5, rtol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 6, rtol=1e-6, err_msg="TiltYZFail")
        self.assertTrue(box.dimensions == 3)


if __name__ == '__main__':
    unittest.main()
