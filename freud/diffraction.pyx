# Copyright (c) 2010-2023 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

r"""
The :class:`freud.diffraction` module provides functions for computing the
diffraction pattern of particles in systems with long range order.

.. rubric:: Stability

:mod:`freud.diffraction` is **unstable**. When upgrading from version 2.x to
2.y (y > x), existing freud scripts may need to be updated. The API will be
finalized in a future release.
"""

from libcpp cimport bool as cbool
from libcpp.vector cimport vector

from freud.util cimport _Compute, vec3

import logging

import numpy as np
import rowan
import scipy.ndimage

import freud.locality

cimport numpy as np

cimport freud._diffraction
cimport freud.locality
cimport freud.util

logger = logging.getLogger(__name__)


cdef class _StaticStructureFactor(_Compute):

    cdef freud._diffraction.StaticStructureFactor * ssfptr

    def __dealloc__(self):
        if type(self) is _StaticStructureFactor:
            del self.ssfptr

    @_Compute._computed_property
    def S_k(self):
        """(:math:`N_{bins}`,) :class:`numpy.ndarray`: Static
        structure factor :math:`S(k)` values."""
        return freud.util.make_managed_numpy_array(
            &self.ssfptr.getStructureFactor(),
            freud.util.arr_type_t.FLOAT)

    @property
    def k_max(self):
        """float: Maximum value of k at which to calculate the structure
        factor."""
        return self.bounds[1]

    @property
    def k_min(self):
        """float: Minimum value of k at which to calculate the structure
        factor."""
        return self.bounds[0]

    @_Compute._computed_property
    def min_valid_k(self):
        return self.ssfptr.getMinValidK()

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


cdef class StaticStructureFactorDebye(_StaticStructureFactor):
    r"""Computes a 1D static structure factor using the
    Debye scattering equation.

    This computes the static `structure factor
    <https://en.wikipedia.org/wiki/Structure_factor>`__ :math:`S(k)` at given
    :math:`k` values by averaging over all :math:`\vec{k}` vectors of the same
    magnitude. Note that freud employs the physics convention in which
    :math:`k` is used, as opposed to the crystallographic one where :math:`q`
    is used. The relation is :math:`k=2 \pi q`. The static structure factor
    calculation is implemented using the Debye scattering equation:

    .. math::

        S(k) = \frac{1}{N} \sum_{i=0}^{N} \sum_{j=0}^{N} \text{sinc}(k r_{ij})

    where :math:`N` is the number of particles, :math:`\text{sinc}` function is
    defined as :math:`\sin x / x` (no factor of :math:`\pi` as in some
    conventions). For more information see `this Wikipedia article
    <https://en.wikipedia.org/wiki/Structure_factor>`__. For a full derivation
    see :cite:`Farrow2009`. Note that the definition requires :math:`S(0) = N`.

    .. note::

        For 2D systems freud uses the Bessel function :math:`J_0` instead of the
        :math:`\text{sinc}` function in the equation above. See
        :cite:`Wieder2012` for more information. For users wishing to calculate
        the structure factor of quasi 2D systems (i.e. a 2D simulation is used
        to model a real system such as particles on a 2D interface or similar)
        the 3D formula should be used. In these cases users should use a 3D box
        with its longest dimension being in the z-direction and particle
        positions of the form :math:`(x, y, 0)`.

    This implementation uses an evenly spaced number of :math:`k` points between
    `k_min`` and ``k_max``. If ``k_min`` is set to 0 (the default behavior), the
    computed structure factor will show :math:`S(0) = N`.

    The Debye implementation provides a much faster algorithm, but gives worse
    results than :py:attr:`freud.diffraction.StaticStructureFactorDirect`
    at low :math:`k` values.

    .. note::
        This code assumes all particles have a form factor :math:`f` of 1.

    Partial structure factors can be computed by providing a set of
    ``query_points`` and the total number of points in the system ``N_total`` to
    the :py:meth:`compute` method. The normalization criterion is based on the
    Faber-Ziman formalism. For particle types :math:`\alpha` and :math:`\beta`,
    we compute the total scattering function as a sum of the partial scattering
    functions as:

    .. math::

        S(k) - 1 = \sum_{\alpha}\sum_{\beta} \frac{N_{\alpha}
        N_{\beta}}{N_{total}^2} \left(S_{\alpha \beta}(k) - 1\right)

    Args:
        num_k_values (unsigned int):
            Number of values to use in :math:`k` space.
        k_max (float):
            Maximum :math:`k` value to include in the calculation.
        k_min (float, optional):
            Minimum :math:`k` value included in the calculation. Note that there
            are practical restrictions on the validity of the calculation in the
            long wavelength regime, see :py:attr:`min_valid_k` (Default value =
            0).
    """
    cdef freud._diffraction.StaticStructureFactorDebye * thisptr

    def __cinit__(self, unsigned int num_k_values, float k_max, float k_min=0):
        if type(self) is StaticStructureFactorDebye:
            self.thisptr = self.ssfptr = new \
                freud._diffraction.StaticStructureFactorDebye(num_k_values,
                                                              k_max,
                                                              k_min)

    def __dealloc__(self):
        if type(self) is StaticStructureFactorDebye:
            del self.thisptr

    @property
    def num_k_values(self):
        """int: The number of k values used."""
        return len(self.k_values)

    @property
    def k_values(self):
        """:class:`numpy.ndarray`: The :math:`k` values for the calculation."""
        return np.array(self.ssfptr.getBinCenters(), copy=True)

    @property
    def bounds(self):
        """tuple: A tuple indicating the smallest and largest :math:`k` values
        used."""
        k_values = self.k_values
        return (k_values[0], k_values[len(k_values)-1])

    def compute(self, system, query_points=None, N_total=None, reset=True):
        r"""Computes static structure factor.

        Example for a single component system::

            >>> box, points = freud.data.make_random_system(10, 100, seed=0)
            >>> sf = freud.diffraction.StaticStructureFactorDebye(
            ...     num_k_values=100, k_max=10, k_min=0
            ... )
            >>> sf.compute((box, points))
            freud.diffraction.StaticStructureFactorDebye(...)

        Example for partial mixed structure factor for a multiple component
        system with types A and B::

            >>> N_particles = 100
            >>> box, points = freud.data.make_random_system(
            ...     10, N_particles, seed=0
            ... )
            >>> A_points = points[:N_particles//2]
            >>> B_points = points[N_particles//2:]
            >>> sf = freud.diffraction.StaticStructureFactorDebye(
            ...     num_k_values=100, k_max=10, k_min=0
            ... )
            >>> sf.compute(
            ...     system=(box, A_points),
            ...     query_points=B_points,
            ...     N_total=N_particles
            ... )
            freud.diffraction.StaticStructureFactorDebye(...)

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
                Note that box is allowed to change when accumulating
                average static structure factor.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the partial structure factor.
                Uses the system's points if :code:`None`. See class
                documentation for information about the normalization of partial
                structure factors. If :code:`None`, the full scattering is
                computed. (Default value = :code:`None`).
            N_total (int, optional):
                Total number of points in the system. This is required if
                ``query_points`` are provided. See class documentation for
                information about the normalization of partial structure
                factors.
            reset (bool, optional):
                Whether to erase the previously computed values before adding
                the new computation; if False, will accumulate data (Default
                value: True).
        """  # noqa E501
        if (query_points is None) != (N_total is None):
            raise ValueError(
                "If query_points are provided, N_total must also be provided "
                "in order to correctly compute the normalization of the "
                "partial structure factor."
            )

        if reset:
            self._reset()

        cdef:
            freud.locality.NeighborQuery nq
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq = freud.locality.NeighborQuery.from_system(system)

        if query_points is None:
            query_points = nq.points
        else:
            query_points = freud.util._convert_array(
                query_points, shape=(None, 3))
        l_query_points = query_points
        num_query_points = l_query_points.shape[0]

        if N_total is None:
            N_total = num_query_points

        self.thisptr.accumulate(
            nq.get_ptr(),
            <vec3[float]*> &l_query_points[0, 0],
            num_query_points, N_total)
        return self

    def _reset(self):
        self.thisptr.reset()

    @_Compute._computed_property
    def min_valid_k(self):
        """float: Minimum valid value of k for the computed system box, equal
        to :math:`2\\pi/(L/2)=4\\pi/L` where :math:`L` is the minimum side length.
        For more information see :cite:`Liu2016`."""
        return self.thisptr.getMinValidK()

    def __repr__(self):
        return ("freud.diffraction.{cls}(num_k_values={num_k_values}, "
                "k_max={k_max}, k_min={k_min})").format(
                    cls=type(self).__name__,
                    num_k_values=self.num_k_values,
                    k_max=self.k_max,
                    k_min=self.k_min)

    def plot(self, ax=None, **kwargs):
        r"""Plot static structure factor.

        .. note::
            This function plots :math:`S(k)` for values above
            :py:attr:`min_valid_k`.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        return freud.plot.line_plot(self.k_values[self.k_values>self.min_valid_k],
                                    self.S_k[self.k_values>self.min_valid_k],
                                    title="Static Structure Factor",
                                    xlabel=r"$k$",
                                    ylabel=r"$S(k)$",
                                    ax=ax)


cdef class StaticStructureFactorDirect(_StaticStructureFactor):
    r"""Computes a 1D static structure factor by operating on a
    :math:`k` space grid.

    This computes the static `structure factor
    <https://en.wikipedia.org/wiki/Structure_factor>`__ :math:`S(k)` at given
    :math:`k` values by averaging over all :math:`\vec{k}` vectors directions of
    the same magnitude. Note that freud employs the physics convention in which
    :math:`k` is used, as opposed to the crystallographic one where :math:`q` is
    used. The relation is :math:`k=2 \pi q`. This is implemented using the
    following formula:

    .. math::

        S(\vec{k}) = \frac{1}{N}  \sum_{i=0}^{N} \sum_{j=0}^N e^{i\vec{k} \cdot
        \vec{r}_{ij}}

    where :math:`N` is the number of particles. Note that the definition
    requires :math:`S(0) = N`.

    This implementation provides a much slower algorithm, but gives better
    results than the :py:attr:`freud.diffraction.StaticStructureFactorDebye`
    method at low k values.

    The :math:`\vec{k}` vectors are sampled isotropically from a grid defined by
    the box's reciprocal lattice vectors. This sampling of reciprocal space is
    based on the MIT licensed `Dynasor library
    <https://gitlab.com/materials-modeling/dynasor/>`__, modified to use
    parallelized C++ and to support larger ranges of :math:`k` values.
    For more information see :cite:`Fransson2021`.

    .. note::
        Currently 2D boxes are not supported for this method. Use Debye instead.

    .. note::
        This code assumes all particles have a form factor :math:`f` of 1.

    Partial structure factors can be computed by providing ``query_points`` and
    total number of points in the system ``N_total`` to the :py:meth:`compute`
    method. The normalization criterion is based on the Faber-Ziman formalism.
    For particle types :math:`\alpha` and :math:`\beta`, we compute the total
    scattering function as a sum of the partial scattering functions as:

    .. math::

        S(k) - 1 = \sum_{\alpha}\sum_{\beta} \frac{N_{\alpha}
        N_{\beta}}{N_{total}^2} \left(S_{\alpha \beta}(k) - 1\right)

    Args:
        bins (unsigned int):
            Number of bins in :math:`k` space.
        k_max (float):
            Maximum :math:`k` value to include in the calculation.
        k_min (float, optional):
            Minimum :math:`k` value included in the calculation. Note that
            there are practical restrictions on the validity of the
            calculation in the long wavelength regime, see :py:attr:`min_valid_k`
            (Default value = 0).
        num_sampled_k_points (unsigned int, optional):
            The desired number of :math:`\vec{k}` vectors to sample from the
            reciprocal lattice grid. If set to 0, all :math:`\vec{k}` vectors
            are used. If greater than 0, the :math:`\vec{k}` vectors are sampled
            from the full grid with uniform radial density, resulting in a
            sample of ``num_sampled_k_points`` vectors on average (Default
            value = 0).
    """

    cdef freud._diffraction.StaticStructureFactorDirect * thisptr

    def __cinit__(self, unsigned int bins, float k_max, float k_min=0,
                  unsigned int num_sampled_k_points=0):
        if type(self) is StaticStructureFactorDirect:
            self.thisptr = self.ssfptr = \
                new freud._diffraction.StaticStructureFactorDirect(
                    bins, k_max, k_min, num_sampled_k_points)

    def __dealloc__(self):
        if type(self) is StaticStructureFactorDirect:
            del self.thisptr

    @property
    def nbins(self):
        """float: Number of bins in the histogram."""
        return len(self.bin_centers)

    @property
    def bin_edges(self):
        """:class:`numpy.ndarray`: The edges of each bin of :math:`k`."""
        return np.array(self.ssfptr.getBinEdges(), copy=True)

    @property
    def bin_centers(self):
        """:class:`numpy.ndarray`: The centers of each bin of :math:`k`."""
        return np.array(self.ssfptr.getBinCenters(), copy=True)

    @property
    def bounds(self):
        """tuple: A tuple indicating upper and lower bounds of the
        histogram."""
        bin_edges = self.bin_edges
        return (bin_edges[0], bin_edges[len(bin_edges)-1])

    def compute(self, system, query_points=None, N_total=None, reset=True):
        r"""Computes static structure factor.

        Example for a single component system::

            >>> box, points = freud.data.make_random_system(10, 100, seed=0)
            >>> sf = freud.diffraction.StaticStructureFactorDirect(
            ...     bins=100, k_max=10, k_min=0
            ... )
            >>> sf.compute((box, points))
            freud.diffraction.StaticStructureFactorDirect(...)

        Example for partial mixed structure factor for multiple component
        system with types A and B::

            >>> N_particles = 100
            >>> box, points = freud.data.make_random_system(
            ...     10, N_particles, seed=0
            ... )
            >>> A_points = points[:N_particles//2]
            >>> B_points = points[N_particles//2:]
            >>> sf = freud.diffraction.StaticStructureFactorDirect(
            ...     bins=100, k_max=10, k_min=0
            ... )
            >>> sf.compute(
            ...     (box, A_points),
            ...     query_points=B_points,
            ...     N_total=N_particles
            ... )
            freud.diffraction.StaticStructureFactorDirect(...)

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`. Note that box is
                allowed to change when accumulating average static structure factor.
                For non-orthorhombic boxes the points are wrapped into a orthorhombic
                box.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the partial structure factor.
                Uses the system's points if :code:`None`. See class
                documentation for information about the normalization of partial
                structure factors. If :code:`None`, the full scattering is
                computed. (Default value = :code:`None`).
            N_total (int, optional):
                Total number of points in the system. This is required if
                ``query_points`` are provided. See class documentation for
                information about the normalization of partial structure
                factors.
            reset (bool, optional):
                Whether to erase the previously computed values before adding
                the new computation; if False, will accumulate data (Default
                value = True).
        """  # noqa E501
        if (query_points is None) != (N_total is None):
            raise ValueError(
                "If query_points are provided, N_total must also be provided "
                "in order to correctly compute the normalization of the "
                "partial structure factor."
            )
        # Convert points to float32 to avoid errors when float64 is passed
        temp_nq = freud.locality.NeighborQuery.from_system(system)
        cdef freud.locality.NeighborQuery nq = \
            freud.locality.NeighborQuery.from_system(
                (temp_nq.box, freud.util._convert_array(temp_nq.points)))

        if reset:
            self._reset()

        cdef:
            const float[:, ::1] l_points = nq.points
            unsigned int num_points = l_points.shape[0]
            const vec3[float]* l_query_points_ptr = NULL
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        if query_points is not None:
            l_query_points = freud.util._convert_array(query_points)
            num_query_points = l_query_points.shape[0]
            l_query_points_ptr = <vec3[float]*> &l_query_points[0, 0]

        if N_total is None:
            N_total = num_points

        self.thisptr.accumulate(
            nq.get_ptr(),
            l_query_points_ptr, num_query_points, N_total
        )
        return self

    def _reset(self):
        self.thisptr.reset()

    @_Compute._computed_property
    def min_valid_k(self):
        """float: Minimum valid value of k for the computed system box, equal
        to :math:`2\\pi/L` where :math:`L` is the minimum side length.
        For more information see :cite:`Liu2016`."""
        return self.thisptr.getMinValidK()

    @property
    def num_sampled_k_points(self):
        r"""int: The target number of :math:`\vec{k}` points to use when
        constructing :math:`k` space grid."""
        return self.thisptr.getNumSampledKPoints()

    @_Compute._computed_property
    def k_points(self):
        r""":class:`numpy.ndarray`: The :math:`\vec{k}` points used in the
        calculation."""
        cdef vector[vec3[float]] k_points = self.thisptr.getKPoints()
        return np.asarray([[k.x, k.y, k.z] for k in k_points])

    def __repr__(self):
        return ("freud.diffraction.{cls}(bins={bins}, "
                "k_max={k_max}, k_min={k_min}, "
                "num_sampled_k_points={num_sampled_k_points})").format(
                    cls=type(self).__name__,
                    bins=self.nbins,
                    k_max=self.k_max,
                    k_min=self.k_min,
                    num_sampled_k_points=self.num_sampled_k_points)

    def plot(self, ax=None, **kwargs):
        r"""Plot static structure factor.

        .. note::
            This function plots :math:`S(k)` for values above
            :py:attr:`min_valid_k`.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        return freud.plot.line_plot(self.bin_centers[self.bin_centers>self.min_valid_k],
                                    self.S_k[self.bin_centers>self.min_valid_k],
                                    title="Static Structure Factor",
                                    xlabel=r"$k$",
                                    ylabel=r"$S(k)$",
                                    ax=ax)


cdef class DiffractionPattern(_Compute):
    r"""Computes a 2D diffraction pattern.

    The diffraction image represents the scattering of incident radiation,
    and is useful for identifying translational and/or rotational symmetry
    present in the system. This class computes the static `structure factor
    <https://en.wikipedia.org/wiki/Structure_factor>`__ :math:`S(\vec{k})` for
    a plane of wavevectors :math:`\vec{k}` orthogonal to a view axis. The
    view orientation :math:`(1, 0, 0, 0)` defaults to looking down the
    :math:`z` axis (at the :math:`xy` plane). The points in the system are
    converted to fractional coordinates, then binned into a grid whose
    resolution is given by ``grid_size``. A higher ``grid_size`` will lead to
    a higher resolution. The points are convolved with a Gaussian of width
    :math:`\sigma`, given by ``peak_width``. This convolution is performed
    as a multiplication in Fourier space. The computed diffraction pattern
    can be accessed as a square array of shape ``(output_size, output_size)``.

    The :math:`\vec{k}=0` peak is always located at index
    ``(output_size // 2, output_size // 2)`` and is normalized to have a value
    of :math:`S(\vec{k}=0) = N`, per common convention. The
    remaining :math:`\vec{k}` vectors are computed such that each peak in the
    diffraction pattern satisfies the relationship :math:`\vec{k} \cdot
    \vec{R} = 2 \pi N` for some integer :math:`N` and lattice vector of
    the system :math:`\vec{R}`. See the `reciprocal lattice Wikipedia page
    <https://en.wikipedia.org/wiki/Reciprocal_lattice>`__ for more information.

    This method is based on the implementations in the open-source
    `GIXStapose application <https://github.com/cmelab/GIXStapose>`_ and its
    predecessor, diffractometer :cite:`Jankowski2017`.

    Note:
        freud only supports diffraction patterns for cubic boxes.

    Args:
        grid_size (unsigned int):
            Resolution of the diffraction grid (Default value = 512).
        output_size (unsigned int):
            Resolution of the output diffraction image, uses ``grid_size`` if
            not provided or ``None`` (Default value = :code:`None`).
    """
    cdef int _grid_size
    cdef int _output_size
    cdef int _N_points
    cdef double[:] _k_values_orig
    cdef double[:, :, :] _k_vectors_orig
    cdef double[:] _k_values
    cdef double[:, :, :] _k_vectors
    cdef double[:, :] _diffraction
    cdef unsigned int _frame_counter
    cdef double _box_matrix_scale_factor
    cdef double[:] _view_orientation
    cdef double _k_scale_factor
    cdef cbool _k_values_cached
    cdef cbool _k_vectors_cached

    def __init__(self, grid_size=512, output_size=None):
        self._grid_size = int(grid_size)
        self._output_size = int(grid_size) if output_size is None \
            else int(output_size)

        # Cache these because they are system-independent.
        self._k_values_orig = np.empty(self.output_size)
        self._k_vectors_orig = np.empty((
            self.output_size, self.output_size, 3))

        # Store these computed arrays which are exposed as properties.
        self._k_values = np.empty_like(self._k_values_orig)
        self._k_vectors = np.empty_like(self._k_vectors_orig)
        self._diffraction = np.zeros((self.output_size, self.output_size))
        self._frame_counter = 0

    def _calc_proj(self, view_orientation, box):
        """Calculate the inverse shear matrix from finding the projected box
        vectors whose area of parallogram is the largest.

        Args:
            view_orientation ((:math:`4`) :class:`numpy.ndarray`):
                View orientation as a quaternion.
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

    def _transform(self, img, box, inv_shear, zoom):
        """Zoom, shear, and scale diffraction intensities.

        Args:
            img ((``grid_size, grid_size``) :class:`numpy.ndarray`):
                Array of diffraction intensities.
            box (:class:`~.box.Box`):
                Simulation box.
            inv_shear ((2, 2) :class:`numpy.ndarray`):
                Inverse shear matrix.
            zoom (float):
                Scaling factor for incident wavevectors.

        Returns:
            (``output_size, output_size``) :class:`numpy.ndarray`:
                Transformed array of diffraction intensities.
        """  # noqa: E501

        # The adjustments to roll and roll_shift ensure that the peak
        # corresponding to k=0 is located at exactly
        # (output_size//2, output_size//2), regardless of whether the grid_size
        # and output_size are odd or even. This keeps the peak aligned at the
        # center of a single pixel, which should always have the maximum value.

        roll = img.shape[0] / 2
        if img.shape[0] % 2 == 1:
            roll -= 0.5

        roll_shift = self.output_size / zoom / 2
        if self.output_size % 2 == 1:
            roll_shift -= 0.5 / zoom

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

        zoom_matrix = np.diag((zoom, zoom, 1))

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

    def compute(self, system, view_orientation=None, zoom=4, peak_width=1, reset=True):
        r"""Computes diffraction pattern.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            view_orientation ((:math:`4`) :class:`numpy.ndarray`, optional):
                View orientation. Uses :math:`(1, 0, 0, 0)` if not provided
                or :code:`None` (Default value = :code:`None`).
            zoom (float, optional):
                Scaling factor for incident wavevectors (Default value = 4).
            peak_width (float, optional):
                Width of Gaussian convolved with points, in system length units
                (Default value = 1).
            reset (bool, optional):
                Whether to erase the previously computed values before adding
                the new computations; if False, will accumulate data (Default
                value = True).
        """
        if reset:
            self._diffraction = np.zeros((self.output_size, self.output_size))
            self._frame_counter = 0

        system = freud.locality.NeighborQuery.from_system(system)

        if not system.box.cubic:
            raise ValueError("freud.diffraction.DiffractionPattern only "
                             "supports cubic boxes")

        if view_orientation is None:
            view_orientation = np.array([1., 0., 0., 0.])
        view_orientation = freud.util._convert_array(
            view_orientation, (4,), np.double)

        # Compute the box projection matrix
        inv_shear = self._calc_proj(view_orientation, system.box)

        # Rotate points by the view quaternion and shear by the box projection
        xy = rowan.rotate(view_orientation, system.points)[:, 0:2]
        xy = xy @ inv_shear.T

        # Map positions to [0, 1] and compute a histogram "image"
        # Use grid_size+1 bin edges so that there are grid_size bins
        xy += 0.5
        xy %= 1
        im, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=np.linspace(0, 1, self.grid_size+1))

        # Compute FFT and convolve with Gaussian
        cdef double complex[:, :] diffraction_fft
        diffraction_fft = np.fft.fft2(im)
        diffraction_fft = scipy.ndimage.fourier_gaussian(
            diffraction_fft, peak_width / zoom)
        diffraction_fft = np.fft.fftshift(diffraction_fft)

        # Compute the squared modulus of the FFT, which is S(k)
        cdef double[:, :] diffraction_frame
        diffraction_frame = np.real(
            diffraction_fft * np.conjugate(diffraction_fft))

        # Transform the image (scale, shear, zoom) and normalize S(k) by the
        # number of points
        self._N_points = len(system.points)
        diffraction_frame = self._transform(
            diffraction_frame, system.box, inv_shear, zoom) / self._N_points

        # Add to the diffraction pattern and increment the frame counter
        self._diffraction += np.asarray(diffraction_frame)
        self._frame_counter += 1

        # Compute a cached array of k-vectors that can be rotated and scaled
        if not self._called_compute:
            # Create a 1D axis of k-vector magnitudes
            self._k_values_orig = np.fft.fftshift(np.fft.fftfreq(
                n=self.output_size))

            # Create a 3D meshgrid of k-vectors with shape
            # (output_size, output_size, 3)
            self._k_vectors_orig = np.asarray(np.meshgrid(
                self._k_values_orig, self._k_values_orig, [0])).T[0]

        # Cache the view orientation and box matrix scale factor for
        # lazy evaluation of k-values and k-vectors
        self._box_matrix_scale_factor = np.max(system.box.to_matrix())
        self._view_orientation = view_orientation
        self._k_scale_factor = 2 * np.pi * self.output_size / \
            (self._box_matrix_scale_factor * zoom)
        self._k_values_cached = False
        self._k_vectors_cached = False

        return self

    @property
    def grid_size(self):
        """int: Resolution of the diffraction grid."""
        return self._grid_size

    @property
    def output_size(self):
        """int: Resolution of the output diffraction image."""
        return self._output_size

    @_Compute._computed_property
    def diffraction(self):
        """
        (``output_size``, ``output_size``) :class:`numpy.ndarray`:
            Diffraction pattern.
        """
        return np.asarray(self._diffraction) / self._frame_counter

    @_Compute._computed_property
    def N_points(self):
        """int: Number of points used in the last computation."""
        return self._N_points

    @_Compute._computed_property
    def k_values(self):
        """(``output_size``,) :class:`numpy.ndarray`: k-values."""
        if not self._k_values_cached:
            self._k_values = np.asarray(self._k_values_orig) * self._k_scale_factor
            self._k_values_cached = True
        return np.asarray(self._k_values)

    @_Compute._computed_property
    def k_vectors(self):
        """(``output_size``, ``output_size``, 3) :class:`numpy.ndarray`: \
        k-vectors."""
        if not self._k_vectors_cached:
            self._k_vectors = rowan.rotate(
                self._view_orientation,
                self._k_vectors_orig) * self._k_scale_factor
            self._k_vectors_cached = True
        return np.asarray(self._k_vectors)

    def __repr__(self):
        return ("freud.diffraction.{cls}(grid_size={grid_size}, "
                "output_size={output_size})").format(
                    cls=type(self).__name__,
                    grid_size=self.grid_size,
                    output_size=self.output_size)

    def to_image(self, cmap='afmhot', vmin=None, vmax=None):
        """Generates image of diffraction pattern.

        Args:
            cmap (str, optional):
                Colormap name to use (Default value = :code:`'afmhot'`).
            vmin (float):
                Minimum of the color scale. Uses :code:`4e-6 * N_points` if
                not provided or :code:`None` (Default value = :code:`None`).
            vmax (float):
                Maximum of the color scale. Uses :code:`0.7 * N_points` if
                not provided or :code:`None` (Default value = :code:`None`).

        Returns:
            ((output_size, output_size, 4) :class:`numpy.ndarray`):
                RGBA array of pixels.
        """
        import matplotlib.cm
        import matplotlib.colors

        if vmin is None:
            vmin = 4e-6 * self.N_points

        if vmax is None:
            vmax = 0.7 * self.N_points

        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.get_cmap(cmap)
        image = cmap(norm(np.clip(self.diffraction, vmin, vmax)))
        return (image * 255).astype(np.uint8)

    def plot(self, ax=None, cmap='afmhot', vmin=None, vmax=None):
        """Plot Diffraction Pattern.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)
            cmap (str, optional):
                Colormap name to use (Default value = :code:`'afmhot'`).
            vmin (float):
                Minimum of the color scale. Uses :code:`4e-6 * N_points` if
                not provided or :code:`None` (Default value = :code:`None`).
            vmax (float):
                Maximum of the color scale. Uses :code:`0.7 * N_points` if
                not provided or :code:`None` (Default value = :code:`None`).

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        if vmin is None:
            vmin = 4e-6 * self.N_points

        if vmax is None:
            vmax = 0.7 * self.N_points

        import freud.plot
        return freud.plot.diffraction_plot(
            self.diffraction, self.k_values, self.N_points,
            ax, cmap, vmin, vmax)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None
