# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.diffraction` module provides functions for computing the
diffraction pattern of particles in periodic systems.

.. rubric:: Stability

:mod:`freud.diffraction` is **unstable**. When upgrading from version 2.x to
2.y (y > x), existing freud scripts may need to be updated. The API will be
finalized in a future release.
"""

import freud.locality
import logging
import numpy as np
import scipy.interpolate
import scipy.ndimage
import rowan
import time

from freud.util cimport _Compute
cimport numpy as np


logger = logging.getLogger(__name__)


class DiffractionPattern(_Compute):
    R"""Computes a 2D diffraction pattern.

    The diffraction image represents the scattering of incident radiation,
    and is useful for identifying translational order present in the system.
    This class computes the static `structure factor
    <https://en.wikipedia.org/wiki/Structure_factor>`_ :math:`S(\vec{q})` for
    a plane of wavevectors :math:`\vec{q}` orthogonal to a view plane. The
    view orientation :math:`(1, 0, 0, 0)` defaults to looking down the
    :math:`z` axis (at the :math:`xy` plane). The points in the system are
    converted to fractional coordinates, then binned into a grid whose
    resolution is given by ``grid_size``. The points are convolved with a
    Gaussian of width :math:`\sigma`, given by ``peak_width``. This
    convolution is performed as a multiplication in Fourier space.

    Args:
        grid_size (unsigned int):
            Size of the diffraction grid (Default value = 512).
        zoom (float):
            Scaling factor for incident wavevectors (Default value = 4).
        peak_width (float):
            Width of Gaussian convolved with points, in system length units
            (Default value = 1).
    """

    def __init__(self, grid_size=512, zoom=4, peak_width=1):
        self.grid_size = grid_size
        self.zoom = zoom
        self.peak_width = peak_width

    def _calc_proj(self, view_orientation, box):
        """Calculate the inverse shear matrix from finding the projected box
        vectors whose area of parallogram is the largest.

        Args:
            view_orientation ((:math:`4`) :class:`numpy.ndarray`):
                View orientation.
            box (:class:`~.box.Box`):
                Simulation box.

        Returns:
            (2, 2) :class:`numpy.ndarray`:
                Inverse shear matrix.
        """
        # Rotate the box matrix by the view orientation.
        box_matrix = rowan.rotate(view_orientation, box.to_matrix())

        # Compute normals for each box face.
        # The area of the face is the length of the vector.
        box_face_normals = np.cross(
            np.roll(box_matrix, 1, axis=-1),
            np.roll(box_matrix, -1, axis=-1),
            axis=0)

        # Compute view axis projections.
        projections = np.abs(box_face_normals.T @ np.array([0., 0., 1.]))

        # Determine the largest projection area along the view axis and use
        # that face for the projection into 2D.
        best_projection_axis = np.argmax(projections)
        secondary_axes = np.array([
            best_projection_axis + 1, best_projection_axis + 2]) % 3

        # Figure out appropriate shear matrix
        shear = box_matrix[np.ix_([0, 1], secondary_axes)]

        # Return the inverse shear matrix
        inv_shear = np.linalg.inv(shear)
        return inv_shear

    def _scale_and_shear(self, img, box, inv_shear):
        """Scales and shears interpolated matrix.

        Args:
            img ((:math:`N`, :math:`N`) :class:`numpy.ndarray`):
                Array of diffraction intensities.
            box (:class:`~.box.Box`):
                Simulation box.
            inv_shear ((2, 2) :class:`numpy.ndarray`):
                Inverse shear matrix.

        Returns:
            (:math:`N`, :math:`N`) :class:`numpy.ndarray`:
                Transformed array of diffraction intensities.
        """
        roll = img.shape[0] / 2 - 1
        box_matrix = box.to_matrix()
        ss = np.max(box_matrix) * inv_shear

        shift_matrix = np.array(
            [[1, 0, -roll],
             [0, 1, -roll],
             [0, 0, 1]])

        shear_matrix = np.array(
            [[ss[1, 0], ss[0, 0], roll],
             [ss[1, 1], ss[0, 1], roll],
             [0, 0, 1]])

        zoom_matrix = np.diag((self.zoom, self.zoom, 1))

        # This matrix uses homogeneous coordinates. It is a 3x3 matrix that
        # transforms 2D points and adds an offset.
        inverse_transform = np.linalg.inv(
            zoom_matrix @ shear_matrix @ shift_matrix)

        img = scipy.ndimage.affine_transform(
            input=img,
            matrix=inverse_transform,
            output_shape=(self.grid_size, self.grid_size),
            order=1,
            mode="constant")
        return img

    def compute(self, system, view_orientation=None):
        R"""Computes diffraction pattern.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            view_orientation ((:math:`4`) :class:`numpy.ndarray`, optional):
                View orientation. Uses :math:`(1, 0, 0, 0)` if not provided
                or :code:`None` (Default value = :code:`None`).
        """
        system = freud.locality.NeighborQuery.from_system(system)

        if view_orientation is None:
            view_orientation = np.array([1., 0., 0., 0.])

        grid_size = int(self.grid_size / self.zoom)
        inv_shear = self._calc_proj(view_orientation, system.box)
        xy = rowan.rotate(view_orientation, system.points)[:, 0:2]
        xy = xy @ inv_shear.T

        # Map positions to [0, 1] and compute the histogram
        xy += 0.5
        xy %= 1
        im, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=np.linspace(0, 1, grid_size))

        self._diffraction = np.fft.fft2(im)
        self._diffraction = scipy.ndimage.fourier.fourier_gaussian(
            self._diffraction, self.peak_width / self.zoom)
        self._diffraction = np.fft.fftshift(self._diffraction)
        self._diffraction = np.absolute(self._diffraction)
        self._diffraction *= self._diffraction
        start = time.time()
        self._diffraction = self._scale_and_shear(
            self._diffraction, system.box, inv_shear)

        # Normalize by N^2
        N = len(system.points)
        self._diffraction /= N*N

        # TODO: FIXME
        self._k_vectors = np.zeros((int(self._diffraction.shape[0]),
                                    int(self._diffraction.shape[1]), 3))

        return self

    @_Compute._computed_property
    def diffraction(self):
        """:class:`numpy.ndarray`: diffraction pattern."""
        return self._diffraction

    @_Compute._computed_property
    def k_vectors(self):
        """(:math:`N`, :math:`N`, 3) :class:`numpy.ndarray`: k-vectors."""
        return self._k_vectors

    def __repr__(self):
        return ("freud.diffraction.{cls}(grid_size={grid_size}, "
                "zoom={zoom}, peak_width={peak_width})").format(
                    cls=type(self).__name__,
                    grid_size=self.grid_size,
                    zoom=self.zoom,
                    peak_width=self.peak_width)

    def plot(self, ax=None, cmap='afmhot', vmin=4e-6, vmax=0.7):
        """Plot Diffraction Pattern.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)
            cmap (str):
                Colormap name to use (Default value = :code:`'afmhot'`).
            vmin (float):
                Minimum of the color scale (Default value = 4e-6).
            vmax (float):
                Maximum of the color scale (Default value = 0.7).

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        return freud.plot.diffraction_plot(
            self.diffraction, ax, cmap, vmin, vmax)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None
