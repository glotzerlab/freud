import unittest
import os

import freud

import numpy as np
import numpy.testing as npt


class TestRotationalAutocorrelation(unittest.TestCase):
    """Test the rotational autocorrelation order parameter"""
    def test_equality(self):
        """Ensure that autocorrelation against identical values is 1"""
        np.random.seed(24)
        orientations = np.random.rand(4, 4)
        orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]

        ra = freud.order.RotationalAutocorrelation(2)
        ra.compute(orientations, orientations)

        npt.assert_allclose(ra.autocorrelation, 1, rtol=1e-6)
        npt.assert_allclose(ra.ra_array, 1, rtol=1e-6)

    def test_attributes(self):
        """Check that all attributes are sensible."""
        np.random.seed(24)
        orientations = np.random.rand(4, 4)
        orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]

        ra = freud.order.RotationalAutocorrelation(2)
        ra.compute(orientations, orientations)

        self.assertEqual(ra.azimuthal, 2)
        self.assertEqual(ra.num_orientations, 4)

    def test_data(self):
        """Regression test against known outputs."""
        fn = os.path.join(os.path.dirname(__file__),
                          'numpy_test_files',
                          'rotational_autocorrelation_orientations.npz')

        with np.load(fn) as data:
            orientations = data['orientations']

            ra2 = freud.order.RotationalAutocorrelation(2)
            l2 = []
            for i in range(orientations.shape[0]):
                ra2.compute(orientations[0, :, :], orientations[i, :, :])
                l2.append(ra2.autocorrelation)
            npt.assert_allclose(l2, data['l2auto'], atol=1e-6, rtol=1e-6)

            ra6 = freud.order.RotationalAutocorrelation(6)
            l6 = []
            for i in range(orientations.shape[0]):
                ra6.compute(orientations[0, :, :], orientations[i, :, :])
                l6.append(ra6.autocorrelation)
            npt.assert_allclose(l6, data['l6auto'], atol=1e-6, rtol=1e-6)

        # As a sanity check, make sure computing with the same object works on
        # new data.
        np.random.seed(42)
        orientations = np.random.rand(4, 4)
        orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]

        npt.assert_allclose(
            ra2.compute(orientations, orientations).autocorrelation,
            1, rtol=1e-6)
        npt.assert_allclose(
            ra6.compute(orientations, orientations).autocorrelation,
            1, rtol=1e-6)

    def test_repr(self):
        ra2 = freud.order.RotationalAutocorrelation(2)
        self.assertEqual(str(ra2), str(eval(repr(ra2))))


if __name__ == '__main__':
    unittest.main()
