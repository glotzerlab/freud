import freud
import pytest
import numpy as np
import numpy.testing as npt


class TestAutocorrelation:

    def test_basic_inputs(self):
        data = np.ones((6, 4, 3))

        corr = freud.correlation.Autocorrelation()
        corr.compute(data)

        npt.assert_allclose(corr.autocorrelation, np.ones(6))
        npt.assert_allclose(corr.particle_autocorrelation, np.ones((6, 4)))
