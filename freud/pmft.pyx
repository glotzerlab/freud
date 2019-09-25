# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.pmft` module allows for the calculation of the Potential of
Mean Force and Torque (PMFT) [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in a
number of different coordinate systems. The shape of the arrays computed by
this module depend on the coordinate system used, with space discretized into a
set of bins created by the PMFT object's constructor. Each reference point's
neighboring points are assigned to bins, determined by the relative positions
and/or orientations of the particles. Next, the positional correlation function
(PCF) is computed by normalizing the binned histogram, by dividing out the
number of accumulated frames, bin sizes (the Jacobian), and reference point
number density. The PMFT is then defined as the negative logarithm of the PCF.
For further descriptions of the numerical methods used to compute the PMFT,
refer to the supplementary information of [vanAndersKlotsa2014]_.

.. note::
    The coordinate system in which the calculation is performed is not the same
    as the coordinate system in which particle positions and orientations
    should be supplied. Only certain coordinate systems are available for
    certain particle positions and orientations:

    * 2D particle coordinates (position: [:math:`x`, :math:`y`, :math:`0`],
      orientation: :math:`\theta`):

        * :math:`r`, :math:`\theta_1`, :math:`\theta_2`.
        * :math:`x`, :math:`y`.
        * :math:`x`, :math:`y`, :math:`\theta`.

    * 3D particle coordinates:

        * :math:`x`, :math:`y`, :math:`z`.

.. note::
    For any bins where the histogram is zero (i.e. no observations were made
    with that relative position/orientation of particles), the PCF will be zero
    and the PMFT will return :code:`nan`.
"""

import numpy as np
import freud.common
import freud.locality
import warnings

from freud.common cimport Compute
from freud.locality cimport SpatialHistogram
from freud.util cimport vec3, quat
from cython.operator cimport dereference

cimport freud._pmft
cimport freud.locality
cimport freud.box

cimport numpy as np


# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class _PMFT(SpatialHistogram):
    R"""Compute the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ for a
    given set of points.

    This class provides an abstract interface for computing the PMFT.
    It must be specialized for a specific coordinate system; although in
    principle the PMFT is coordinate independent, the binning process must be
    performed in a particular coordinate system.
    """
    cdef freud._pmft.PMFT * pmftptr

    def __cinit__(self):
        pass

    def __dealloc__(self):
        if type(self) is _PMFT:
            del self.pmftptr

    @Compute._computed_property()
    def PMFT(self):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            result = -np.log(np.copy(self.PCF))
        return result

    @Compute._computed_property()
    def PCF(self):
        return freud.util.make_managed_numpy_array(
            &self.pmftptr.getPCF(),
            freud.util.arr_type_t.FLOAT)


cdef class PMFTR12(_PMFT):
    R"""Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in a 2D
    system described by :math:`r`, :math:`\theta_1`, :math:`\theta_2`.

    .. note::
        **2D:** :class:`freud.pmft.PMFTR12` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Args:
        r_max (float):
            Maximum distance at which to compute the PMFT.
        bins (unsigned int or sequence of length 3):
            If an unsigned int, the number of bins in:math:`r`,
            :math:`\theta_1`, and :math:`\theta_2`. If a sequence of three
            integers, interpreted as :code:`(num_bins_r, num_bins_t1,
            num_bins_t2)`.

    Attributes:
        PCF (:math:`\left(N_{r}, N_{\theta1}, N_{\theta2}\right)`):
            The positional correlation function.
        PMFT (:math:`\left(N_{r}, N_{\theta1}, N_{\theta2}\right)`):
            The potential of mean force and torque.
        r_max (float):
            The cutoff used in the cell list.
    """  # noqa: E501
    cdef freud._pmft.PMFTR12 * pmftr12ptr

    def __cinit__(self, r_max, bins):
        if type(self) is PMFTR12:
            try:
                n_r, n_t1, n_t2 = bins
            except TypeError:
                n_r = n_t1 = n_t2 = bins
            self.pmftr12ptr = self.pmftptr = self.histptr = \
                new freud._pmft.PMFTR12(r_max, n_r, n_t1, n_t2)
            self.r_max = r_max

    def __dealloc__(self):
        if type(self) is PMFTR12:
            del self.pmftr12ptr

    @Compute._compute()
    def accumulate(self, box, points, orientations, query_points=None,
                   query_orientations=None, nlist=None, query_args=None):
        R"""Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`points` if not
                provided or :code:`None`. (Default value = :code:`None`).
            query_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`orientations` if not provided or :code:`None`.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef:
            freud.box.Box b
            freud.locality.NeighborQuery nq
            freud.locality.NlistptrWrapper nlistptr
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        b, nq, nlistptr, qargs, l_query_points, num_query_points = \
            self.preprocess_arguments(box, points, query_points, nlist,
                                      query_args, dimensions=2)

        orientations = freud.common.convert_array(
            np.atleast_1d(orientations.squeeze()),
            shape=(nq.points.shape[0], ))
        if query_orientations is None:
            query_orientations = orientations
        else:
            query_orientations = freud.common.convert_array(
                np.atleast_1d(query_orientations.squeeze()),
                shape=(l_query_points.shape[0], ))
        cdef const float[::1] l_orientations = orientations
        cdef const float[::1] l_query_orientations = query_orientations

        self.pmftr12ptr.accumulate(nq.get_ptr(),
                                   <float*> &l_orientations[0],
                                   <vec3[float]*> &l_query_points[0, 0],
                                   <float*> &l_query_orientations[0],
                                   num_query_points, nlistptr.get_ptr(),
                                   dereference(qargs.thisptr))
        return self

    @Compute._compute()
    def compute(self, box, points, orientations, query_points=None,
                query_orientations=None, nlist=None, query_args=None):
        R"""Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`points` if not
                provided or :code:`None`. (Default value = :code:`None`).
            query_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`orientations` if not provided or :code:`None`.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(box, points, orientations,
                        query_points, query_orientations, nlist, query_args)
        return self

    def __repr__(self):
        bounds = self.bounds
        return ("freud.pmft.{cls}(r_max={r_max}, bins=({bins}))").format(
            cls=type(self).__name__,
            r_max=self.r_max,
            bins=', '.join([str(b) for b in self.nbins]))


cdef class PMFTXYT(_PMFT):
    R"""Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ for
    systems described by coordinates :math:`x`, :math:`y`, :math:`\theta`
    listed in the ``X``, ``Y``, and ``T`` arrays.

    The values of :math:`x, y, \theta` at which to compute the PCF are
    controlled by ``x_max``, ``y_max``, and ``bins`` parameters to the
    constructor. The ``x_max`` and ``y_max`` parameters determine the
    minimum/maximum :math:`x, y` values (:math:`\min \left(\theta \right) = 0`,
    (:math:`\max \left( \theta \right) = 2\pi`) at which to compute the PCF.
    The ``bins`` may be either an integer, in which case it is interpreted as
    the number of bins in each dimension, or a sequence of length 3, in which
    case it is interpreted as the number of bins in :math:`x`, :math:`y`, and
    :math:`\theta`.

    .. note::
        **2D:** :class:`freud.pmft.PMFTXYT` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        bins (unsigned int or sequence of length 3):
            If an unsigned int, the number of bins in:math:`x`, :math:`y`, and
            :math:`t`. If a sequence of three integers, interpreted as
            :code:`(num_bins_x, num_bins_y, num_bins_t)`.

    Attributes:
        PCF (:math:`\left(N_{x}, N_{y}, N_{\theta}\right)` :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\left(N_{x}, N_{y}, N_{\theta}\right)` :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_max (float):
            The cutoff used in the cell list.
    """  # noqa: E501
    cdef freud._pmft.PMFTXYT * pmftxytptr

    def __cinit__(self, x_max, y_max, bins):
        if type(self) is PMFTXYT:
            try:
                n_x, n_y, n_t = bins
            except TypeError:
                n_x = n_y = n_t = bins

            self.pmftxytptr = self.pmftptr = self.histptr = \
                new freud._pmft.PMFTXYT(x_max, y_max, n_x, n_y, n_t)
            self.r_max = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXYT:
            del self.pmftxytptr

    @Compute._compute()
    def accumulate(self, box, points, orientations, query_points=None,
                   query_orientations=None, nlist=None, query_args=None):
        R"""Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`points` if not
                provided or :code:`None`. (Default value = :code:`None`).
            query_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`orientations` if not provided or :code:`None`.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef:
            freud.box.Box b
            freud.locality.NeighborQuery nq
            freud.locality.NlistptrWrapper nlistptr
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        b, nq, nlistptr, qargs, l_query_points, num_query_points = \
            self.preprocess_arguments(box, points, query_points, nlist,
                                      query_args, dimensions=2)

        orientations = freud.common.convert_array(
            np.atleast_1d(orientations.squeeze()),
            shape=(nq.points.shape[0], ))
        if query_orientations is None:
            query_orientations = orientations
        else:
            query_orientations = freud.common.convert_array(
                np.atleast_1d(query_orientations.squeeze()),
                shape=(query_points.shape[0], ))
        cdef const float[::1] l_orientations = orientations
        cdef const float[::1] l_query_orientations = query_orientations

        self.pmftxytptr.accumulate(nq.get_ptr(),
                                   <float*> &l_orientations[0],
                                   <vec3[float]*> &l_query_points[0, 0],
                                   <float*> &l_query_orientations[0],
                                   num_query_points, nlistptr.get_ptr(),
                                   dereference(qargs.thisptr))
        return self

    @Compute._compute()
    def compute(self, box, points, orientations, query_points=None,
                query_orientations=None, nlist=None, query_args=None):
        R"""Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`points` if not
                provided or :code:`None`. (Default value = :code:`None`).
            query_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`orientations` if not provided or :code:`None`.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(box, points, orientations,
                        query_points, query_orientations, nlist, query_args)
        return self

    def __repr__(self):
        bounds = self.bounds
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, "
                "bins=({bins}))").format(cls=type(self).__name__,
                                         x_max=bounds[0][1],
                                         y_max=bounds[1][1],
                                         bins=', '.join(
                                             [str(b) for b in self.nbins]))


cdef class PMFTXY2D(_PMFT):
    R"""Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in
    coordinates :math:`x`, :math:`y` listed in the ``X`` and ``Y`` arrays.

    The values of :math:`x` and :math:`y` at which to compute the PCF are
    controlled by ``x_max``, ``y_max``, and ``bins`` parameters to the
    constructor. The ``x_max`` and ``y_max`` parameters determine the
    minimum/maximum distance at which to compute the PCF.  The ``bins`` may be
    either an integer, in which case it is interpreted as the number of bins in
    each dimension, or a sequence of length 2, in which case it is interpreted
    as the number of bins in :math:`x` and :math:`y` respectively.

    .. note::
        **2D:** :class:`freud.pmft.PMFTXY2D` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        bins (unsigned int or sequence of length 2):
            If an unsigned int, the number of bins in:math:`x`, :math:`y`, and
            :math:`z`. If a sequence of two integers, interpreted as
            :code:`(num_bins_x, num_bins_y)`.

    Attributes:
        PCF (:math:`\left(N_{x}, N_{y}\right)` :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\left(N_{x}, N_{y}\right)` :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_max (float):
            The cutoff used in the cell list.
    """  # noqa: E501
    cdef freud._pmft.PMFTXY2D * pmftxy2dptr

    def __cinit__(self, x_max, y_max, bins):
        if type(self) is PMFTXY2D:
            try:
                n_x, n_y = bins
            except TypeError:
                n_x = n_y = bins

            self.pmftxy2dptr = self.pmftptr = self.histptr = \
                new freud._pmft.PMFTXY2D(x_max, y_max, n_x, n_y)
            self.r_max = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXY2D:
            del self.pmftxy2dptr

    @Compute._compute()
    def accumulate(self, box, points, orientations, query_points=None,
                   nlist=None, query_args=None):
        R"""Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`points` if not
                provided or :code:`None`. (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef:
            freud.box.Box b
            freud.locality.NeighborQuery nq
            freud.locality.NlistptrWrapper nlistptr
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        b, nq, nlistptr, qargs, l_query_points, num_query_points = \
            self.preprocess_arguments(box, points, query_points, nlist,
                                      query_args, dimensions=2)

        orientations = freud.common.convert_array(
            np.atleast_1d(orientations.squeeze()),
            shape=(nq.points.shape[0], ))
        cdef const float[::1] l_orientations = orientations

        self.pmftxy2dptr.accumulate(nq.get_ptr(),
                                    <float*> &l_orientations[0],
                                    <vec3[float]*> &l_query_points[0, 0],
                                    num_query_points, nlistptr.get_ptr(),
                                    dereference(qargs.thisptr))
        return self

    @Compute._compute()
    def compute(self, box, points, orientations, query_points=None,
                nlist=None, query_args=None):
        R"""Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`points` if not
                provided or :code:`None`. (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(box, points, orientations,
                        query_points, nlist, query_args)
        return self

    @Compute._computed_property()
    def bin_counts(self):
        # Currently this returns a 3D array that must be squeezed due to the
        # internal choices in the histogramming; this will be fixed in future
        # changes.
        return np.squeeze(super(PMFTXY2D, self).bin_counts)

    @Compute._computed_property()
    def PCF(self):
        # Currently this returns a 3D array that must be squeezed due to the
        # internal choices in the histogramming; this will be fixed in future
        # changes.
        return np.squeeze(super(PMFTXY2D, self).PCF)

    def __repr__(self):
        bounds = self.bounds
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, "
                "bins=({bins}))").format(cls=type(self).__name__,
                                         x_max=bounds[0][1],
                                         y_max=bounds[1][1],
                                         bins=', '.join(
                                             [str(b) for b in self.nbins]))

    def _repr_png_(self):
        import freud.plot
        try:
            return freud.plot.ax_to_bytes(self.plot())
        except AttributeError:
            return None

    @Compute._computed_method()
    def plot(self, ax=None):
        """Plot PMFTXY2D.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        return freud.plot.pmft_plot(self, ax)


cdef class PMFTXYZ(_PMFT):
    R"""Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in
    coordinates :math:`x`, :math:`y`, :math:`z`, listed in the ``X``, ``Y``,
    and ``Z`` arrays.

    The values of :math:`x, y, z` at which to compute the PCF are controlled by
    ``x_max``, ``y_max``, ``z_max``, and ``bins`` parameters to the constructor.
    The ``x_max``, ``y_max``, and ``z_max`` parameters] determine the
    minimum/maximum distance at which to compute the pair correlation function.
    The ``bins`` may be either an integer, in which case it is interpreted as
    the number of bins in each dimension, or a sequence of length 3, in which
    case it is interpreted as the number of bins in :math:`x`, :math:`y`, and
    :math:`z` respectively.

    .. note::
        3D: :class:`freud.pmft.PMFTXYZ` is only defined for 3D systems.
        The points must be passed in as :code:`[x, y, z]`.

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        z_max (float):
            Maximum :math:`z` distance at which to compute the PMFT.
        bins (unsigned int or sequence of length 3):
            If an unsigned int, the number of bins in :math:`x`, :math:`y`, and
            :math:`z`. If a sequence of three integers, interpreted as
            :code:`(num_bins_x, num_bins_y, num_bins_z)`.
        shiftvec (list):
            Vector pointing from ``[0, 0, 0]`` to the center of the PMFT.

    Attributes:
        PCF (:math:`\left(N_{x}, N_{y}, N_{z}\right)` :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\left(N_{x}, N_{y}, N_{z}\right)` :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_max (float):
            The cutoff used in the cell list.
    """  # noqa: E501
    cdef freud._pmft.PMFTXYZ * pmftxyzptr
    cdef shiftvec

    def __cinit__(self, x_max, y_max, z_max, bins,
                  shiftvec=[0, 0, 0]):
        cdef vec3[float] c_shiftvec

        try:
            n_x, n_y, n_z = bins
        except TypeError:
            n_x = n_y = n_z = bins

        if type(self) is PMFTXYZ:
            c_shiftvec = vec3[float](
                shiftvec[0], shiftvec[1], shiftvec[2])
            self.pmftxyzptr = self.pmftptr = self.histptr = \
                new freud._pmft.PMFTXYZ(
                    x_max, y_max, z_max, n_x, n_y, n_z, c_shiftvec)
            self.shiftvec = np.array(shiftvec, dtype=np.float32)
            self.r_max = np.sqrt(x_max**2 + y_max**2 + z_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXYZ:
            del self.pmftxyzptr

    @Compute._compute()
    def accumulate(self, box, points, orientations, query_points=None,
                   face_orientations=None, nlist=None, query_args=None):
        R"""Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Reference orientations as quaternions used in computation.
            query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`points` if not
                provided or :code:`None`. (Default value = :code:`None`).
            face_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`, optional):
                Orientations of particle faces to account for particle
                symmetry. If not supplied by user, unit quaternions will be
                supplied. If a 2D array of shape (:math:`N_f`, 4) or a
                3D array of shape (1, :math:`N_f`, 4) is supplied, the
                supplied quaternions will be broadcast for all particles.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef:
            freud.box.Box b
            freud.locality.NeighborQuery nq
            freud.locality.NlistptrWrapper nlistptr
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        b, nq, nlistptr, qargs, l_query_points, num_query_points = \
            self.preprocess_arguments(box, points, query_points, nlist,
                                      query_args, dimensions=3)
        l_query_points = l_query_points - self.shiftvec.reshape(1, 3)

        orientations = freud.common.convert_array(
            np.atleast_1d(orientations),
            shape=(nq.points.shape[0], 4))

        cdef const float[:, ::1] l_orientations = orientations

        # handle multiple ways to input
        if face_orientations is None:
            # set to unit quaternion, q = [1, 0, 0, 0]
            face_orientations = np.zeros(
                shape=(nq.points.shape[0], 1, 4), dtype=np.float32)
            face_orientations[:, :, 0] = 1.0
        else:
            if face_orientations.ndim < 2 or face_orientations.ndim > 3:
                raise ValueError("points must be a 2 or 3 dimensional array")
            face_orientations = freud.common.convert_array(face_orientations)
            if face_orientations.ndim == 2:
                if face_orientations.shape[1] != 4:
                    raise ValueError(
                        "2nd dimension for orientations must have 4 values:"
                        "w, x, y, z")
                # need to broadcast into new array
                tmp_face_orientations = np.zeros(
                    shape=(nq.points.shape[0],
                           face_orientations.shape[0],
                           face_orientations.shape[1]),
                    dtype=np.float32)
                tmp_face_orientations[:] = face_orientations
                face_orientations = tmp_face_orientations
            else:
                # Make sure that the first dimension is actually the number
                # of particles
                if face_orientations.shape[2] != 4:
                    raise ValueError(
                        "2nd dimension for orientations must have 4 values:"
                        "w, x, y, z")
                elif face_orientations.shape[0] not in (
                        1, nq.points.shape[0]):
                    raise ValueError(
                        "If provided as a 3D array, the first dimension of "
                        "the face_orientations array must be either of "
                        "size 1 or N_particles")
                elif face_orientations.shape[0] == 1:
                    face_orientations = np.repeat(
                        face_orientations, nq.points.shape[0], axis=0)

        cdef const float[:, :, ::1] l_face_orientations = face_orientations
        cdef unsigned int num_faces = l_face_orientations.shape[1]
        self.pmftxyzptr.accumulate(
            nq.get_ptr(),
            <quat[float]*> &l_orientations[0, 0],
            <vec3[float]*> &l_query_points[0, 0],
            num_query_points,
            <quat[float]*> &l_face_orientations[0, 0, 0],
            num_faces, nlistptr.get_ptr(), dereference(qargs.thisptr))
        return self

    @Compute._compute()
    def compute(self, box, points, orientations, query_points=None,
                face_orientations=None, nlist=None,
                query_args=None):
        R"""Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Reference orientations as quaternions used in computation.
            query_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`points` if not
                provided or :code:`None`. (Default value = :code:`None`).
            face_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`, optional):
                Orientations of particle faces to account for particle
                symmetry. If not supplied by user, unit quaternions will be
                supplied. If a 2D array of shape (:math:`N_f`, 4) or a
                3D array of shape (1, :math:`N_f`, 4) is supplied, the
                supplied quaternions will be broadcast for all particles.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(box, points, orientations,
                        query_points, face_orientations,
                        nlist, query_args)
        return self

    def __repr__(self):
        bounds = self.bounds
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, "
                "z_max={z_max}, bins=({bins}), "
                "shiftvec={shiftvec})").format(
                    cls=type(self).__name__,
                    x_max=bounds[0][1],
                    y_max=bounds[1][1],
                    z_max=bounds[2][1],
                    bins=', '.join([str(b) for b in self.nbins]),
                    shiftvec=self.shiftvec.tolist())
