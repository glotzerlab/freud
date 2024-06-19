# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from collections import namedtuple

import numpy as np
import numpy.testing as npt
import pytest

import freud


class TestCommon:
    def test_convert_array(self):
        # create array
        x = np.arange(100)
        # create a non-contiguous array
        y = x.reshape(10, 10).T
        # run through convert
        # first check to make sure it passes with default
        z = freud.util._convert_array(y, (None, None))
        npt.assert_equal(y.dtype, x.dtype)
        # now change type
        z = freud.util._convert_array(y, (None, None), dtype=np.float32)
        npt.assert_equal(z.dtype, np.float32)
        # now make contiguous
        npt.assert_equal(y.flags.contiguous, False)
        z = freud.util._convert_array(y, (None, None))
        npt.assert_equal(z.flags.contiguous, True)

        # test dimension checking
        with pytest.raises(ValueError):
            z = freud.util._convert_array(y, (None,), dtype=np.float32)

        # test for non-default dtype
        z = freud.util._convert_array(y, dtype=np.float64)
        npt.assert_equal(z.dtype, np.float64)

        # test for list of list input
        yl = [list(r) for r in y]
        zl = freud.util._convert_array(yl, (None, None))
        z = freud.util._convert_array(y, (None, None))
        npt.assert_equal(z, zl)

        # test for dimensions default argument
        zd = freud.util._convert_array(y)
        z = freud.util._convert_array(y, (None, None))
        npt.assert_equal(z, zd)

        # test dimension checking
        with pytest.raises(ValueError):
            z = freud.util._convert_array(y, shape=(1,), dtype=np.float32)

        with pytest.raises(ValueError):
            freud.util._convert_array(z, shape=(None, 9))

    def test_convert_matrix_box(self):
        matrix_box = np.array([[1, 2, 3], [0, 2, 3], [0, 0, 3]])
        box = freud.util._convert_box(matrix_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 1, rtol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 1, rtol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 1, rtol=1e-6, err_msg="TiltYZFail")
        assert box.dimensions == 3

    def test_convert_tuple_box(self):
        TupleBox = namedtuple("TupleBox", ["Lx", "Ly", "Lz", "xy", "xz", "yz"])
        tuple_box = TupleBox(1, 2, 3, 4, 5, 6)
        box = freud.util._convert_box(tuple_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 4, rtol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 5, rtol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 6, rtol=1e-6, err_msg="TiltYZFail")
        assert box.dimensions == 3

    def test_convert_dict_box(self):
        dict_box = dict(Lx=1, Ly=2, Lz=3, xy=4, xz=5, yz=6)
        box = freud.util._convert_box(dict_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 4, rtol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 5, rtol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 6, rtol=1e-6, err_msg="TiltYZFail")
        assert box.dimensions == 3

    def test_convert_array_len_2_box(self):
        array_box = [1, 2]
        box = freud.util._convert_box(array_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 0, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 0, atol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 0, atol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 0, atol=1e-6, err_msg="TiltYZFail")
        assert box.dimensions == 2

    def test_convert_array_len_3_box(self):
        array_box = [1, 2, 3]
        box = freud.util._convert_box(array_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 0, atol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 0, atol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 0, atol=1e-6, err_msg="TiltYZFail")
        assert box.dimensions == 3

    def test_convert_array_len_6_box(self):
        array_box = [1, 2, 3, 4, 5, 6]
        box = freud.util._convert_box(array_box)
        npt.assert_allclose(box.Lx, 1, rtol=1e-6, err_msg="LxFail")
        npt.assert_allclose(box.Ly, 2, rtol=1e-6, err_msg="LyFail")
        npt.assert_allclose(box.Lz, 3, rtol=1e-6, err_msg="LzFail")
        npt.assert_allclose(box.xy, 4, rtol=1e-6, err_msg="TiltXYFail")
        npt.assert_allclose(box.xz, 5, rtol=1e-6, err_msg="TiltXZFail")
        npt.assert_allclose(box.yz, 6, rtol=1e-6, err_msg="TiltYZFail")
        assert box.dimensions == 3
