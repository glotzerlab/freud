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


cdef class DiffractionPattern(_Compute):
    R"""Computes a 2D diffraction pattern.

    The diffraction image represents the scattering of incident radiation,
    and is useful for identifying translational order present in the system.
    This class computes the static `structure factor
    <https://en.wikipedia.org/wiki/Structure_factor>`_ :math:`S(\vec{q})` for
    a plane of wavevectors :math:`\vec{q}` orthogonal to a view axis. The
    view orientation :math:`(1, 0, 0, 0)` defaults to looking down the
    :math:`z` axis (at the :math:`xy` plane). The points in the system are
    converted to fractional coordinates, then binned into a grid whose
    resolution is given by ``grid_size``. A higher ``grid_size`` will lead to
    a higher resolution. The points are convolved with a Gaussian of width
    :math:`\sigma`, given by ``peak_width``. This convolution is performed
    as a multiplication in Fourier space. The computed diffraction pattern
    is returned as a square array of shape ``(output_size, output_size)`.

    Args:
        grid_size (unsigned int):
            Resolution of the diffraction grid. (Default value = 512).
        output_size (unsigned int):
            Size of the output diffraction image (Default value = 512).
        zoom (float):
            Scaling factor for incident wavevectors (Default value = 4).
        peak_width (float):
            Width of Gaussian convolved with points, in system length units
            (Default value = 1).
    """
    cdef int grid_size
    cdef int output_size
    cdef float zoom
    cdef float peak_width
    cdef np.ndarray _k_values_orig
    cdef np.ndarray _k_vectors_orig
    cdef np.ndarray _k_values
    cdef np.ndarray _k_vectors
    cdef np.ndarray _diffraction
    cdef bint has_computed

    def __init__(self, grid_size=512, output_size=512, zoom=4, peak_width=1):
        self.grid_size = int(grid_size)
        self.output_size = int(output_size)
        self.zoom = float(zoom)
        self.peak_width = float(peak_width)
        self.has_computed = False

        # Cache these because they can be computed independent of system.
        self._k_values_orig = np.zeros(grid_size)
        self._k_vectors_orig = np.zeros((grid_size, grid_size))

        # Cache these because they are exposed as properties.
        self._k_values = np.zeros(grid_size)
        self._k_vectors = np.zeros((grid_size, grid_size))
        self._diffraction = np.zeros((grid_size, grid_size))

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

    def _transform(self, img, box, inv_shear):
        """Zoom, shear, and scale diffraction intensities.

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
        roll = img.shape[0] / 2
        if img.shape[0] % 2 == 0:
            roll -= 0.5

        roll_shift = self.output_size / self.zoom / 2
        if (self.output_size / self.zoom) % 2 == 0:
            roll_shift -= 0.5

        box_matrix = box.to_matrix()
        ss = np.max(box_matrix) * inv_shear

        shift_matrix = np.array(
            [[1, 0, -roll],
             [0, 1, -roll],
             [0, 0, 1]])

        # Translation for [roll_shift, roll_shift]
        # Then shift using ss
        shear_matrix = np.array(
            [[ss[1, 0], ss[0, 0], roll_shift],
             [ss[1, 1], ss[0, 1], roll_shift],
             [0, 0, 1]])

        zoom_matrix = np.diag((self.zoom, self.zoom, 1))

        # This matrix uses homogeneous coordinates. It is a 3x3 matrix that
        # transforms 2D points and adds an offset.
        inverse_transform = np.linalg.inv(
            zoom_matrix @ shear_matrix @ shift_matrix)

        img = scipy.ndimage.affine_transform(
            input=img,
            matrix=inverse_transform,
            output_shape=(self.output_size, self.output_size),
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

        # Compute the box projection matrix
        inv_shear = self._calc_proj(view_orientation, system.box)

        # Rotate points by the view quaternion and shear by the box projection
        xy = rowan.rotate(view_orientation, system.points)[:, 0:2]
        xy = xy @ inv_shear.T

        # Map positions to [0, 1] and compute a histogram "image"
        xy += 0.5
        xy %= 1
        im, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=np.linspace(0, 1, grid_size))

        # Compute FFT and convolve with Gaussian
        self._diffraction = np.fft.fft2(im)
        self._diffraction = scipy.ndimage.fourier.fourier_gaussian(
            self._diffraction, self.peak_width / self.zoom)
        self._diffraction = np.fft.fftshift(self._diffraction)

        # Compute the squared modulus of the FFT, which is S(q)
        self._diffraction = np.real(
            self._diffraction * np.conjugate(self._diffraction))

        # Transform the image (scale, shear, and zoom)
        self._diffraction = self._transform(
            self._diffraction, system.box, inv_shear)

        # Normalize S(q) by N^2
        N = len(system.points)
        self._diffraction /= N*N

        # Compute a cached array of k-vectors that can be rotated and scaled
        if not self.has_computed:
            # Create a 1D axis of k-vector magnitudes
            self._k_values_orig = np.fft.fftshift(np.fft.fftfreq(
                n=self.output_size,
                d=1/self.output_size))

            # Create a 3D meshgrid of k-vectors, shape (N, N, 3)
            self._k_vectors_orig = np.asarray(np.meshgrid(
                self._k_values_orig, self._k_values_orig, [0])).T
            self._k_vectors_orig = self._k_vectors_orig.reshape(-1, 3)
            self.has_computed = True

        # Compute the rotated and scaled k-values and k-vectors
        self._k_values = self._k_values_orig / np.max(system.box.to_matrix())
        self._k_vectors = rowan.rotate(view_orientation, self._k_vectors_orig)
        self._k_vectors /= np.max(system.box.to_matrix())

        return self

    @_Compute._computed_property
    def diffraction(self):
        """:class:`numpy.ndarray`: diffraction pattern."""
        return self._diffraction

    @_Compute._computed_property
    def k_values(self):
        """(:math:`N`, ) :class:`numpy.ndarray`: k-values."""
        return self._k_values

    @_Compute._computed_property
    def k_vectors(self):
        """(:math:`N`, :math:`N`, 3) :class:`numpy.ndarray`: k-vectors."""
        return self._k_vectors

    def __repr__(self):
        return ("freud.diffraction.{cls}(grid_size={grid_size}, "
                "output_size={output_size}, zoom={zoom}, "
                "peak_width={peak_width})").format(
                    cls=type(self).__name__,
                    grid_size=self.grid_size,
                    output_size=self.output_size,
                    zoom=self.zoom,
                    peak_width=self.peak_width)

    def to_image(self, cmap='afmhot', vmin=4e-6, vmax=0.7):
        """Generates image of diffraction pattern.

        Args:
            cmap (str):
                Colormap name to use (Default value = :code:`'afmhot'`).
            vmin (float):
                Minimum of the color scale (Default value = 4e-6).
            vmax (float):
                Maximum of the color scale (Default value = 0.7).

        Returns:
            ((grid_size, grid_size, 4) :class:`numpy.ndarray`):
                RGBA array of pixels.
        """
        import matplotlib.cm
        import matplotlib.colors
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.get_cmap(cmap)
        image = cmap(norm(np.clip(self.diffraction, vmin, vmax)))
        return (image * 255).astype(np.uint8)

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
            self.diffraction, self.k_values, ax, cmap, vmin, vmax)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None
