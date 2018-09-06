import numpy as np
import numpy.testing as npt
from freud.environment import Pairing2D
import warnings
from freud import box
from freud.errors import FreudDeprecationWarning
import unittest


class TestPairing(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=FreudDeprecationWarning)

    # by Eric
    def test_pairing(self):
        fbox = box.Box(Lx=10, Ly=10, is2D=True)
        pos = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
        ang = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        myPair = Pairing2D(rmax=1.1, k=2, compDotTol=0.1)
        c_ang = np.zeros(shape=(pos.shape[0], 2), dtype=np.float32)
        c_ang[:, 1] = np.pi
        myPair.compute(fbox, pos, ang, c_ang)
        match_list = np.copy(myPair.match)
        pair_list = np.copy(myPair.pair)
        npt.assert_equal(match_list, [1, 1, 1],
                         err_msg="Incorrect matches reported")
        npt.assert_equal(pair_list, [1, 0, 1],
                         err_msg="Incorrect pairs reported")
