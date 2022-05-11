r"""
The `freud.correlation` module provides classes for computing correlations of
particles in periodic systems.
"""

from freud.util cimport _Compute
import logging

import numpy as np

import freud.parallel

cimport numpy as np

logger = logging.getLogger(__name__)

# Use fastest available fft library
try:
    import pyfftw
    logger.info("Using PyFFTW for FFTs")

    pyfftw.config.NUM_THREADS = min(1, freud.parallel.get_num_threads())
    logger.info("Setting number of threads to {}".format(
        freud.parallel.get_num_threads()))

    # Note that currently these functions are defined to match only the parts
    # of the numpy/scipy API that are actually used below. There is no promise
    # that other aspects of the API will be preserved.
    def fft(x, n, axis):
        a = pyfftw.empty_aligned(x.shape, 'complex64')
        a[:] = x
        fft_object = pyfftw.builders.fft(a, n=n, axis=axis)
        return fft_object()

    def ifft(x, axis):
        a = pyfftw.empty_aligned(x.shape, 'complex64')
        a[:] = x
        fft_object = pyfftw.builders.ifft(a, axis=axis)
        return fft_object()
except ImportError:
    try:
        from scipy.fftpack import fft, ifft
        logger.info("Using SciPy's fftpack for FFTs")
    except ImportError:
        from numpy.fft import fft, ifft
        logger.info("Using NumPy for FFTs")


cdef class Autocorrelation(_Compute):
    r"""Compute the autocorrelation function.

    The autocorrelation function measures how correlated data is across time
    scales. The implementation in this class can compute the autocorrelation
    function for quantities defined in an arbitrary number of dimensions. We
    associate with each point in the simulation a :math:`d`-dimensional
    vector quantity :math:`v`. The autocorrelation is defined as

    .. math::

        R(\Delta t) = \frac{1}{N_{particles}} \sum_{i=1}^{N_{particles}} \frac
        { \sum_{t=0}^{N_{frames} - \Delta t - 1} \left(\vec{v_i}(t) - \vec{\bar{v}_i}
        \right) \dot \left( \vec{v_i}(t + \Delta t) - \vec{\bar{v}_i} \right) }
        { \sum_{t=0}^{N_{frames} - 1} |\left( \vec{v_i} - \vec{\bar{v}_i} \right)|^2 }

    where the bar refers to an average taken over all time windows in each of
    the :math:`d` dimensions separately. To be explicit

    .. math::

        \vec{\bar{v}_i} = \left( \sum_{t=0}^{N_{frames}} v_{i,0}(t),
        \sum_{t=0}^{N_{frames}} v_{i,1}(t), ... \right)

    Note:
        Similar to the ``window`` method of `freud.msd.MSD` module, this
        calculation is implemented using the fastest available FFT library. See
        `freud.msd.MSD` for more information.
    """
    # TODO put the full docs for FFT choices here, and have MSD refer to this
    # module when describing FFT choices and accumulating over points rather than frames

    def __cinit__(self):
        cdef _autocorr
        self._autocorr = []

    def compute(self, data, reset=False):
        """Compute the Autocorrelation function for the provided data.

        Note:
            If computing the positional autocorrelation (i.e. ``data`` is
            position data), we assume all positions are already unwrapped.

        Note:
            Similar to the MSD module, accumulation for the autocorrelation
            function is split over points rather than frames of a simulation.
            See the ``compute`` method of `freud.msd.MSD` for more information.

        Args:
            data (:math:`N_{frames}`, :math:`N_{particles}`, :math:`d`):
                Data to use for computing the autocorrelation function.
            reset (bool):
                Whether to erase the previously computed values before adding
                the new computation; if False, will accumulate data (Default
                value: True).
        """

        if reset:
            self._autocorr = []

        corrs = []
        for i in range(data.shape[2]):
            x = data[:, :, i]
            N = x.shape[0]
            F = fft(x, n=2*N, axis=0)
            PSD = F * F.conjugate()
            res = ifft(PSD, axis=0)
            res = (res[:N]).real
            n = np.arange(1, N+1)[::-1]
            corrs.append(res/n[:, np.newaxis])
        self._autocorr.append(np.sum(corrs, axis=0))

    @_Compute._computed_property
    def autocorrelation(self):
        """:math:`\\left(N_{frames} \\right)`: The autocorrelation function
        averaged over all particles."""
        return np.concatenate(self._autocorr, axis=1).mean(axis=-1)

    @_Compute._computed_property
    def particle_autocorrelation(self):
        """:math:`\\left(N_{frames}, N_{particles} \\right)`: The
        autocorrelation function of each particle."""
        return np.concatenate(self._autocorr, axis=1)

    # TODO add a plot function

    def __repr__(self):
        return "freud.correlation.Autocorrelation()"
