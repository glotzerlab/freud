# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The MSD module provides functions for computing the mean-squared-displacement
of particles in periodic systems.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import freud.common

cimport freud.box
cimport numpy as np

# SciPy's FFT appears faster, so use it if available.
try:
    from scipy.fftpack import fft, ifft
except ImportError:
    from numpy.fft import fft, ifft


def _autocorrelation(x):
    R"""Compute the autocorrelation of a sequence"""
    N = x.shape[0]
    F = fft(x, n=2*N, axis=0)
    PSD = F * F.conjugate()
    res = ifft(PSD, axis=0)
    res = (res[:N]).real
    n = np.arange(1, N+1)[::-1]  # N to 1
    return res/n[:, np.newaxis]


cdef class MSD:
    R"""Compute the mean squared displacement.

    The mean squared displacement (MSD) measures how much particles move over
    time. The MSD plays an important role in characterizing Brownian motion,
    since it provides a measure of whether particles are moving according to
    diffusion alone or if there are other forces contributing. There are a
    number of definitions for the mean squared displacement. This function
    provides access to the two most common definitions through the mode
    argument.

    * :code:`'window'` (*default*):
      This mode calculates the most common form of the MSD, which is defined as

      .. math::

          MSD(m) = \left\langle\frac{1}{N-m} \sum_{k=0}^{N-m-1} (\vec{r}(k+m) - \vec{r}(k))^2\right\rangle_{particles}

      According to this definition, the mean squared displacement is the
      average displacement over all windows of length :math:`m` over the course
      of the simulation. Therefore, for any :math:`m`, :math:`MSD(m)` is
      averaged over all windows of length :math:`m` and over all particles.
      This calculation can be accessed using the 'window' mode of this
      function.

      The windowed calculation can be quite computationally intensive. To
      perform this calculation efficiently, we use the algorithm described in
      [Calandrini2011]_ as described in `this StackOverflow thread
      <https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft>`_.

    * :code:`'direct'`:
      Under some circumstances, however, we may be more interested in
      calculating a different quantity described by

      .. math::
          :nowrap:

          \begin{eqnarray*}
              MSD(t) = &\langle (\vec{r}-\vec{r}_0)^2 \rangle_{particles} \\
                     = & \dfrac{1}{N} \sum_{n=1}^N (x_n(t) - x_n(0))^2 \\
          \end{eqnarray*}

      In this case, we simply compute how much particles have moved from their
      initial position, averaged over all particles. For more information on
      this calculation, see `the Wikipedia page
      <https://en.wikipedia.org/wiki/Mean_squared_displacement>`_.

    .. note::
        The MSD is only well-defined when the box is constant over the
        course of the simulation. Additionally, the number of particles must be
        constant over the course of the simulation.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:class:`freud.box.Box`, optional):
            If not provided, the class will assume that all positions provided
            in calls to :meth:`~compute` or :meth:`~accumulate` are already
            unwrapped.
        mode (str, optional):
            Mode of calculation. Options are :code:`'window'` and
            :code:`'direct'`.  (Default value = :code:`'window'`).

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        msd (:math:`\left(N_{frames}, \right)` :class:`numpy.ndarray`):
            The mean squared displacement.
    """   # noqa: E501
    cdef freud.box.Box box
    cdef particle_msd
    cdef str mode

    def __cinit__(self, box=None, mode='window'):
        if box is not None:
            self.box = freud.common.convert_box(box)
        else:
            self.box = None

        self.particle_msd = []

        if mode not in ['window', 'direct']:
            raise ValueError("Invalid mode")
        self.mode = mode

    def accumulate(self, positions, images=None):
        """Calculate the MSD for the positions provided and add to the existing
        per-particle data.

        .. note::
            Unlike most methods in freud, accumulation for the MSD is split
            over particles rather than frames of a simulation. The reason for
            this choice is that efficient computation of the MSD requires using
            the entire trajectory for a given particle. As a result, this
            accumulation is primarily useful when the trajectory is so large
            that computing an MSD on all particles at once is prohibitive.

        Args:
            positions ((:math:`N_{frames}`, :math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                The particle positions over a trajectory. If neither box nor images
                are provided, the positions are assumed to be unwrapped already.
            images ((:math:`N_{frames}`, :math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                The particle images to unwrap with if provided. Must be provided
                along with a simulation box (in the constructor) if particle
                positions need to be unwrapped. If neither are provided,
                positions are assumed to be unwrapped already.
        """  # noqa: E501

        positions = freud.common.convert_array(
            positions, 3, dtype=np.float32, contiguous=True,
            array_name="positions")
        if positions.shape[2] != 3:
            raise TypeError('positions should be an MxNx3 array')

        if images is not None:
            images = freud.common.convert_array(
                images, 3, dtype=np.float32, contiguous=True,
                array_name="images")
            if images.shape[2] != 3:
                raise TypeError('images should be an MxNx3 array')

            if not positions.shape == images.shape:
                raise TypeError(
                    'The positions and images must have the same shape')

        # Make sure we aren't modifying the provided array
        unwrapped_positions = positions.copy()
        if self.box is not None and images is not None:
            for i in range(positions.shape[0]):
                unwrapped_positions[i, :, :] = self.box.unwrap(
                    unwrapped_positions[i, :, :], images[i, :, :])
        positions = unwrapped_positions

        if self.mode == 'window':
            # First compute the first term r^2(k+m) - r^2(k)
            N = positions.shape[0]
            D = np.square(positions).sum(axis=2)
            D = np.append(D, np.zeros(positions.shape[:2]), axis=0)
            Q = 2*D.sum(axis=0)
            S1 = np.zeros(positions.shape[:2])
            for m in range(N):
                Q -= (D[m-1, :] + D[N-m, :])
                S1[m, :] = Q/(N-m)

            # The second term can be computed via autocorrelation
            corrs = []
            for i in range(positions.shape[2]):
                corrs.append(_autocorrelation(positions[:, :, i]))
            S2 = np.sum(corrs, axis=0)

            self.particle_msd.append(S1 - 2*S2)
        elif self.mode == 'direct':
            self.particle_msd.append(
                np.linalg.norm(
                    positions - positions[[0], :, :], axis=-1)**2)

        return self

    @property
    def box(self):
        return self.box

    @property
    def msd(self):
        return np.concatenate(self.particle_msd, axis=1).mean(axis=-1)

    def reset(self):
        R"""Clears the stored MSD values from previous calls to accumulate (or
        the last call to compute)."""
        self.particle_msd = []

    def compute(self, positions, images=None):
        """Calculate the MSD for the positions provided.

        Args:
            positions ((:math:`N_{frames}`, :math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                The particle positions over a trajectory. If neither box nor images
                are provided, the positions are assumed to be unwrapped already.
            images ((:math:`N_{frames}`, :math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                The particle images to unwrap with if provided. Must be provided
                along with a simulation box (in the constructor) if particle
                positions need to be unwrapped. If neither are provided,
                positions are assumed to be unwrapped already.
        """  # noqa: E501
        self.reset()
        self.accumulate(positions, images)
        return self
