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
        except TypeError as e:
            npt.assert_equal(True, True)

    def test_convert_tuple_box(self):
        TupleBox = namedtuple('TupleBox', ['Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz'])
        fake_box = TupleBox(1, 2, 3, 4, 5, 6)
        box = common.convert_box(fake_box)
        npt.assert_almost_equal(box.Lx, 1, decimal=2, err_msg="LxFail")
        npt.assert_almost_equal(box.Ly, 2, decimal=2, err_msg="LyFail")
        npt.assert_almost_equal(box.Lz, 3, decimal=2, err_msg="LzFail")
        npt.assert_almost_equal(box.xy, 4, decimal=2, err_msg="TiltXYFail")
        npt.assert_almost_equal(box.xz, 5, decimal=2, err_msg="TiltXZFail")
        npt.assert_almost_equal(box.yz, 6, decimal=2, err_msg="TiltYZFail")

    def test_convert_dict_box(self):
        dict_box = dict(Lx=1, Ly=2, Lz=3, xy=4, xz=5, yz=6)
        box = common.convert_box(dict_box)
        npt.assert_almost_equal(box.Lx, 1, decimal=2, err_msg="LxFail")
        npt.assert_almost_equal(box.Ly, 2, decimal=2, err_msg="LyFail")
        npt.assert_almost_equal(box.Lz, 3, decimal=2, err_msg="LzFail")
        npt.assert_almost_equal(box.xy, 4, decimal=2, err_msg="TiltXYFail")
        npt.assert_almost_equal(box.xz, 5, decimal=2, err_msg="TiltXZFail")
        npt.assert_almost_equal(box.yz, 6, decimal=2, err_msg="TiltYZFail")


if __name__ == '__main__':
    unittest.main()
