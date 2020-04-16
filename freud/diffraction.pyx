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
        bot (float):
            Plotting quantity -- should be removed (Default value = 4e-6).
        top (float):
            Plotting quantity -- should be removed (Default value = 0.7).
    """

    def __init__(self, grid_size=512, zoom=4, peak_width=1,
                 bot=4e-6, top=0.7):
        self.grid_size = grid_size
        self.zoom = zoom
        self.peak_width = peak_width
        self.bin_w = 2.0
        self.bot = bot
        self.top = top

    def _pbc_2d(self, xy, grid_size):
        """Reasonably fast periodic boundary conditions in two dimensions.
           Normalizes xy coordinates to the grid size.

        Args:
            xy ((:math:`N_{bins}`, 2) :class:`numpy.ndarray`):
                Cartesian coordinates from [-0.5, 0.5) to be mapped to
                [0, grid_size).
            grid_size (unsigned int) :
                Size of the diffraction grid.

        Returns:
            (:math:`N_{bins}`, 2) :class:`numpy.ndarray`:
                Particle bins indices in the x and y directions.
        """
        xy -= np.rint(xy) - 0.5
        xy *= grid_size
        xy %= grid_size
        return xy.astype(int)

    def _bin(self, xy, grid_size):
        """Quickly counts intensities for particles on 2D grid.

        Args:
            xy ((:math:`N_{bins}`, 2) :class:`numpy.ndarray`):
                Array of bin indices.
            grid_size (unsigned int):
                Size of the diffraction grid.

        Returns:
            im ((grid_size, grid_size) :class:`numpy.ndarray`):
                Grid of intensities.
        """
        t = xy.view(np.dtype((np.void, xy.dtype.itemsize * xy.shape[1])))
        _, ids, counts = np.unique(t, return_index=True, return_counts=True)
        unique_xy = xy[ids]
        grid_size = int(grid_size)
        im = np.zeros((grid_size, grid_size))
        for x, c in zip(unique_xy, counts):
            im[x[1], x[0]] = c
        return im

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

    def _circle_cutout(self, p):
        """Find pixel indices in diffraction intensity array outside of the circle
           Note: taken from Diffractometer.prep_sq()

        Args:
            p ((:math:`N`, :math:`N`) :class:`numpy.ndarray`):
                The array of diffraction intensity.

        Returns:
            (:math:`N`,) :class:`numpy.ndarray`:
                Indices of particles outside the circle.
                note: N != to N in p.shape
        """
        y, x = np.indices(p.shape)
        rmax = len(x) / 2 - 1
        center = np.array([rmax, rmax])
        # radii, constant for a single zoom
        r = np.hypot(x - center[1], y - center[0]).flatten()
        # array index into p corresponding to r
        i = np.argsort(r.flat)
        # sorted radius indices
        r_sort = r.flat[i]
        return i[r_sort > rmax]

    def _scale(self, a):
        """Scales up a matrix around middle particle.
            Note: Doesn't handle atoms on periodic boundaries perfectly --
            intensity only on one half of boundary.

        Args:
            a ((:math:`N`, :math:`N`) :class:`numpy.ndarray`): Input array.

        Returns:
            (:math:`N`, :math:`N`) :class:`numpy.ndarray`: Scaled array
        """
        ny, nx = np.shape(a)
        y = np.array([list(range(ny))])
        x = np.array([list(range(nx))])
        d = scipy.interpolate.RectBivariateSpline(x, y, a, kx=1, ky=1)
        x = np.linspace(0, nx, self.grid_size)
        y = np.linspace(0, ny, self.grid_size)
        d = d(x, y)
        return d

    def _shear_back(self, img, box, inv_shear):
        """Transform the inverse shear matrix back to the sheared matrix

        Args:
            img ((:math:`N`, :math:`N`) :class:`numpy.ndarray`):
                Array of diffraction intensities.
            box (:class:`~.box.Box`):
                Simulation box.
            inv_shear ((2, 2) :class:`numpy.ndarray`):
                Inverse shear matrix.

        Returns
            (:math:`N`, :math:`N`) :class:`numpy.ndarray`:
                Sheared array of diffraction intensities
        """
        roll = img.shape[0] / 2 - 1
        box_matrix = box.to_matrix()
        ss = np.max(box_matrix) * inv_shear
        A1 = np.array([[1, 0, -roll],
                       [0, 1, -roll],
                       [0, 0, 1]])

        A2 = np.array([[ss[1, 0], ss[0, 0], roll],
                       [ss[1, 1], ss[0, 1], roll],
                       [0, 0, 1]])

        A3 = np.linalg.inv(np.dot(A2, A1))
        A4 = A3[0:2, 0:2]
        A5 = A3[0:2, 2]
        img = scipy.ndimage.interpolation.affine_transform(
            img, A4, A5, mode="constant")
        return img

    def compute(self, system, view_orientation=None, cutout=True):
        R"""2D FFT to get diffraction pattern from intensity matrix.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            view_orientation ((:math:`4`) :class:`numpy.ndarray`, optional):
                View orientation. Uses :math:`(1, 0, 0, 0)` if not provided
                or :code:`None` (Default value = :code:`None`).
            cutout (bool, optional):
                diffraction pattern with circle cutout
                (Default value = :code:`True`).
        """
        system = freud.locality.NeighborQuery.from_system(system)

        if view_orientation is None:
            view_orientation = np.array([1., 0., 0., 0.])

        grid_size = self.grid_size / self.zoom
        inv_shear = self._calc_proj(view_orientation, system.box)
        xy = np.copy(rowan.rotate(view_orientation, system.points)[:, 0:2])
        xy = np.dot(xy, inv_shear.T)
        xy = self._pbc_2d(xy, grid_size)
        im = self._bin(xy, grid_size)

        self._diffraction = np.fft.fft2(im)
        self._diffraction = scipy.ndimage.fourier.fourier_gaussian(
            self._diffraction, self.peak_width / self.zoom)
        self._diffraction = np.fft.fftshift(self._diffraction)
        self._diffraction = np.absolute(self._diffraction)
        self._diffraction *= self._diffraction

        self._diffraction = self._scale(self._diffraction)
        self._diffraction = self._shear_back(
            self._diffraction, system.box, inv_shear)
        self._diffraction /= self._diffraction.max()
        self._diffraction[self._diffraction < self.bot] = self.bot
        self._diffraction[self._diffraction > self.top] = self.top
        self._diffraction = np.log10(self._diffraction)

        """
        NOTE: cut into a circle, not sure if needed-YJ
        """
        # if not cutout:
        #     return dp

        # idbig = self.circle_cutout(dp)
        # dp[np.unravel_index(idbig, (self.grid_size, self.grid_size))] = np.log(self.bot)
        return self

    @_Compute._computed_property
    def diffraction(self):
        """(:class:`numpy.ndarray`): diffraction pattern. """
        return self._diffraction

    def __repr__(self):
        return f"freud.diffraction.{type(self).__name__}, (N={self.N}, \
                 zoom={self.zoom}, peak_width={self.peak_width}, \
                 bot={self.bot}, top={self.top})"

    def plot(self, ax=None, cmap='afmhot'):
        """Plot Diffraction Pattern.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)
            cmap (str):
                Colormap name to use (Default value = :code:`'afmhot'`).

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        return freud.plot.diffraction_plot(self.diffraction, ax, cmap)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None
