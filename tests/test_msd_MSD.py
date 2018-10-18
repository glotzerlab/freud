from __future__ import division

import numpy as np
import numpy.testing as npt
from freud import msd
import unittest


class TestBox(unittest.TestCase):

    def test_msd(self):
        """Test correct behavior for various constructor signatures"""
        positions = np.array([[[1, 0, 0]]])
        self.assertTrue(msd.MSD(positions) == [0])

        positions = positions.repeat(10, axis=0)
        npt.assert_array_almost_equal(msd.MSD(positions), 0)

        positions[:, 0, 0] = np.arange(10)
        npt.assert_array_almost_equal(msd.MSD(positions), np.arange(10)**2)

        positions = positions.repeat(2, axis=1)
        positions[:, 1, :] = 0
        npt.assert_array_almost_equal(msd.MSD(positions), np.arange(10)**2/2)

        def simple_msd(positions):
            """A naive MSD calculation, used to test."""
            msds = []

            for m in np.arange(positions.shape[0]):
                if m:
                    diffs = positions[:-m, :, :] - positions[m:, :, :]
                else:
                    diffs = np.zeros_like(positions)
                sqdist = np.square(diffs).sum(axis=2)
                msds.append(sqdist.mean(axis=0))

            return np.array(msds).mean(axis=1)

        num_tests = 5
        for _ in range(num_tests):
            positions = np.random.rand(10, 10, 3)
            simple = simple_msd(positions)
            solution = msd.MSD(positions)
            npt.assert_array_almost_equal(solution, simple)


if __name__ == '__main__':
    unittest.main()
