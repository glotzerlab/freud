import numpy as np
import numpy.testing as npt
from freud.util import pos
import unittest

class TestPOS(unittest.TestCase):
    # The pos_test.pos file has cases that will test the gamut of the pos.py
    def test_read(self):
        pos_file = pos.file("pos_test.pos")
        pos_file.grabData()
        pos_file.grabBox()
        npt.assert_equal(pos_file.isData, True)
        npt.assert_equal(pos_file.isBox, True)
        npt.assert_equal(pos_file.isDefs, True)

if __name__ == '__main__':
    unittest.main()
