import numpy as np
import numpy.testing as npt
from freud import common
import unittest

class TestCommon(unittest.TestCase):
    def test_convert_array(self):
        # create array
        x = np.arange(100)
        # create a non-contiguous array
        y = x.reshape(10,10).T
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

if __name__ == '__main__':
    unittest.main()
