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
from freud.util._VectorMath cimport vec3, quat
from cython.operator cimport dereference

cimport freud._pmft
cimport freud.locality
cimport freud.box

cimport numpy as np


# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class _PMFT(Compute):
    R"""Compute the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ for a
    given set of points.

    This class provides an abstract interface for computing the PMFT.
    It must be specialized for a specific coordinate system; although in
    principle the PMFT is coordinate independent, the binning process must be
    performed in a particular coordinate system.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>
    """
    cdef freud._pmft.PMFT * pmftptr
    cdef float rmax

    def __cinit__(self):
        pass

    def __dealloc__(self):
        if type(self) is _PMFT:
            del self.pmftptr

    @Compute._computed_property()
    def box(self):
        return freud.box.BoxFromCPP(self.pmftptr.getBox())

    @Compute._reset
    def reset(self):
        R"""Resets the values of the PCF histograms in memory."""
        self.pmftptr.reset()

    @Compute._computed_property()
    def PMFT(self):
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            result = -np.log(np.copy(self.PCF))
        return result

    @property
    def r_cut(self):
        return self.pmftptr.getRCut()


cdef class PMFTR12(_PMFT):
    R"""Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in a 2D
    system described by :math:`r`, :math:`\theta_1`, :math:`\theta_2`.

    .. note::
        **2D:** :class:`freud.pmft.PMFTR12` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        r_max (float):
            Maximum distance at which to compute the PMFT.
        n_r (unsigned int):
            Number of bins in :math:`r`.
        n_t1 (unsigned int):
            Number of bins in :math:`\theta_1`.
        n_t2 (unsigned int):
            Number of bins in :math:`\theta_2`.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        bin_counts (:math:`\left(N_{r}, N_{\theta2}, N_{\theta1}\right)`):
            Bin counts.
        PCF (:math:`\left(N_{r}, N_{\theta2}, N_{\theta1}\right)`):
            The positional correlation function.
        PMFT (:math:`\left(N_{r}, N_{\theta2}, N_{\theta1}\right)`):
            The potential of mean force and torque.
        r_cut (float):
            The cutoff used in the cell list.
        R (:math:`\left(N_{r}\right)` :class:`numpy.ndarray`):
            The array of :math:`r`-values for the PCF histogram.
        T1 (:math:`\left(N_{\theta1}\right)` :class:`numpy.ndarray`):
            The array of :math:`\theta_1`-values for the PCF histogram.
        T2 (:math:`\left(N_{\theta2}\right)` :class:`numpy.ndarray`):
            The array of :math:`\theta_2`-values for the PCF histogram.
        inverse_jacobian (:math:`\left(N_{r}, N_{\theta2}, N_{\theta1}\right)`):
            The inverse Jacobian used in the PMFT.
        n_bins_R (unsigned int):
            The number of bins in the :math:`r`-dimension of the histogram.
        n_bins_T1 (unsigned int):
            The number of bins in the :math:`\theta_1`-dimension of the
            histogram.
        n_bins_T2 (unsigned int):
            The number of bins in the :math:`\theta_2`-dimension of the
            histogram.
    """  # noqa: E501
    cdef freud._pmft.PMFTR12 * pmftr12ptr

    def __cinit__(self, r_max, n_r, n_t1, n_t2):
        if type(self) is PMFTR12:
            self.pmftr12ptr = self.pmftptr = new freud._pmft.PMFTR12(
                r_max, n_r, n_t1, n_t2)
            self.rmax = r_max

    def __dealloc__(self):
        if type(self) is PMFTR12:
            del self.pmftr12ptr

    @Compute._compute()
    def accumulate(self, box, ref_points, ref_orientations, points=None,
                   orientations=None, nlist=None):
        R"""Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef freud.box.Box b = freud.common.convert_box(box)

        if not b.dimensions == 2:
            raise ValueError("Your box must be 2-dimensional!")

        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))

        ref_orientations = freud.common.convert_array(
            ref_orientations.squeeze(), shape=(ref_points.shape[0], ))

        points = freud.common.convert_array(points, shape=(None, 3))

        orientations = freud.common.convert_array(
            orientations.squeeze(), shape=(points.shape[0], ))

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points = points
        cdef const float[::1] l_ref_orientations = ref_orientations
        cdef const float[::1] l_orientations = orientations
        cdef unsigned int nRef = l_ref_points.shape[0]
        cdef unsigned int nP = l_points.shape[0]
        with nogil:
            self.pmftr12ptr.accumulate(dereference(b.thisptr),
                                       nlist_.get_ptr(),
                                       <vec3[float]*> &l_ref_points[0, 0],
                                       <float*> &l_ref_orientations[0],
                                       nRef,
                                       <vec3[float]*> &l_points[0, 0],
                                       <float*> &l_orientations[0],
                                       nP)
        return self

    @Compute._compute()
    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        R"""Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    @Compute._computed_property()
    def bin_counts(self):
        cdef unsigned int n_bins_R = self.pmftr12ptr.getNBinsR()
        cdef unsigned int n_bins_T2 = self.pmftr12ptr.getNBinsT2()
        cdef unsigned int n_bins_T1 = self.pmftr12ptr.getNBinsT1()
        cdef const unsigned int[:, :, ::1] bin_counts = \
            <unsigned int[:n_bins_R, :n_bins_T2, :n_bins_T1]> \
            self.pmftr12ptr.getBinCounts().get()
        return np.asarray(bin_counts, dtype=np.uint32)

    @Compute._computed_property()
    def PCF(self):
        cdef unsigned int n_bins_R = self.pmftr12ptr.getNBinsR()
        cdef unsigned int n_bins_T2 = self.pmftr12ptr.getNBinsT2()
        cdef unsigned int n_bins_T1 = self.pmftr12ptr.getNBinsT1()
        cdef const float[:, :, ::1] PCF = \
            <float[:n_bins_R, :n_bins_T2, :n_bins_T1]> \
            self.pmftr12ptr.getPCF().get()
        return np.asarray(PCF)

    @property
    def R(self):
        cdef unsigned int n_bins_R = self.pmftr12ptr.getNBinsR()
        cdef const float[::1] R = \
            <float[:n_bins_R]> self.pmftr12ptr.getR().get()
        return np.asarray(R)

    @property
    def T1(self):
        cdef unsigned int n_bins_T1 = self.pmftr12ptr.getNBinsT1()
        cdef const float[::1] T1 = \
            <float[:n_bins_T1]> self.pmftr12ptr.getT1().get()
        return np.asarray(T1)

    @property
    def T2(self):
        cdef unsigned int n_bins_T2 = self.pmftr12ptr.getNBinsT2()
        cdef const float[::1] T2 = \
            <float[:n_bins_T2]> self.pmftr12ptr.getT2().get()
        return np.asarray(T2)

    @property
    def inverse_jacobian(self):
        cdef unsigned int n_bins_R = self.pmftr12ptr.getNBinsR()
        cdef unsigned int n_bins_T2 = self.pmftr12ptr.getNBinsT2()
        cdef unsigned int n_bins_T1 = self.pmftr12ptr.getNBinsT1()
        cdef const float[:, :, ::1] inverse_jacobian = \
            <float[:n_bins_R, :n_bins_T2, :n_bins_T1]> \
            self.pmftr12ptr.getInverseJacobian().get()
        return np.asarray(inverse_jacobian)

    @property
    def n_bins_R(self):
        return self.pmftr12ptr.getNBinsR()

    @property
    def n_bins_T1(self):
        return self.pmftr12ptr.getNBinsT1()

    @property
    def n_bins_T2(self):
        return self.pmftr12ptr.getNBinsT2()

    def __repr__(self):
        return ("freud.pmft.{cls}(r_max={r_max}, n_r={n_r}, n_t1={n_t1}, "
                "n_t2={n_t2})").format(cls=type(self).__name__,
                                       r_max=self.rmax,
                                       n_r=self.n_bins_R,
                                       n_t1=self.n_bins_T1,
                                       n_t2=self.n_bins_T2)

    def __str__(self):
        return repr(self)


cdef class PMFTXYT(_PMFT):
    R"""Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ for
    systems described by coordinates :math:`x`, :math:`y`, :math:`\theta`
    listed in the ``X``, ``Y``, and ``T`` arrays.

    The values of :math:`x, y, \theta` at which to compute the PCF are
    controlled by ``x_max``, ``y_max``, and ``n_x``, ``n_y``, ``n_t``
    parameters to the constructor. The ``x_max`` and ``y_max`` parameters
    determine the minimum/maximum :math:`x, y` values
    (:math:`\min \left(\theta \right) = 0`,
    (:math:`\max \left( \theta \right) = 2\pi`) at which to compute the
    PCF and ``n_x``, ``n_y``, ``n_t`` are the number of bins in
    :math:`x, y, \theta`.

    .. note::
        **2D:** :class:`freud.pmft.PMFTXYT` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        n_x (unsigned int):
            Number of bins in :math:`x`.
        n_y (unsigned int):
            Number of bins in :math:`y`.
        n_t (unsigned int):
            Number of bins in :math:`\theta`.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        bin_counts (:math:`\left(N_{\theta}, N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            Bin counts.
        PCF (:math:`\left(N_{\theta}, N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\left(N_{\theta}, N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_cut (float):
            The cutoff used in the cell list.
        X (:math:`\left(N_{x}\right)` :class:`numpy.ndarray`):
            The array of :math:`x`-values for the PCF histogram.
        Y (:math:`\left(N_{y}\right)` :class:`numpy.ndarray`):
            The array of :math:`y`-values for the PCF histogram.
        T (:math:`\left(N_{\theta}\right)` :class:`numpy.ndarray`):
            The array of :math:`\theta`-values for the PCF histogram.
        jacobian (float):
            The Jacobian used in the PMFT.
        n_bins_X (unsigned int):
            The number of bins in the :math:`x`-dimension of the histogram.
        n_bins_Y (unsigned int):
            The number of bins in the :math:`y`-dimension of the histogram.
        n_bins_T (unsigned int):
            The number of bins in the :math:`\theta`-dimension of the
            histogram.
    """  # noqa: E501
    cdef freud._pmft.PMFTXYT * pmftxytptr
    cdef xmax
    cdef ymax

    def __cinit__(self, x_max, y_max, n_x, n_y, n_t):
        if type(self) is PMFTXYT:
            self.pmftxytptr = self.pmftptr = new freud._pmft.PMFTXYT(
                x_max, y_max, n_x, n_y, n_t)
            self.rmax = np.sqrt(x_max**2 + y_max**2)
            self.xmax = x_max
            self.ymax = y_max

    def __dealloc__(self):
        if type(self) is PMFTXYT:
            del self.pmftxytptr

    @Compute._compute()
    def accumulate(self, box, ref_points, ref_orientations, points=None,
                   orientations=None, nlist=None):
        R"""Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef freud.box.Box b = freud.common.convert_box(box)

        if not b.dimensions == 2:
            raise ValueError("Your box must be 2-dimensional!")

        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))

        ref_orientations = freud.common.convert_array(
            ref_orientations.squeeze(), shape=(ref_points.shape[0], ))

        points = freud.common.convert_array(points, shape=(None, 3))

        orientations = freud.common.convert_array(
            orientations.squeeze(), shape=(points.shape[0], ))

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points = points
        cdef const float[::1] l_ref_orientations = ref_orientations
        cdef const float[::1] l_orientations = orientations
        cdef unsigned int nRef = l_ref_points.shape[0]
        cdef unsigned int nP = l_points.shape[0]
        with nogil:
            self.pmftxytptr.accumulate(dereference(b.thisptr),
                                       nlist_.get_ptr(),
                                       <vec3[float]*> &l_ref_points[0, 0],
                                       <float*> &l_ref_orientations[0],
                                       nRef,
                                       <vec3[float]*> &l_points[0, 0],
                                       <float*> &l_orientations[0],
                                       nP)
        return self

    @Compute._compute()
    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        R"""Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    @Compute._computed_property()
    def bin_counts(self):
        cdef unsigned int n_bins_T = self.pmftxytptr.getNBinsT()
        cdef unsigned int n_bins_Y = self.pmftxytptr.getNBinsY()
        cdef unsigned int n_bins_X = self.pmftxytptr.getNBinsX()
        cdef const unsigned int[:, :, ::1] bin_counts = \
            <unsigned int[:n_bins_T, :n_bins_Y, :n_bins_X]> \
            self.pmftxytptr.getBinCounts().get()
        return np.asarray(bin_counts, dtype=np.uint32)

    @Compute._computed_property()
    def PCF(self):
        cdef unsigned int n_bins_T = self.pmftxytptr.getNBinsT()
        cdef unsigned int n_bins_Y = self.pmftxytptr.getNBinsY()
        cdef unsigned int n_bins_X = self.pmftxytptr.getNBinsX()
        cdef const float[:, :, ::1] PCF = \
            <float[:n_bins_T, :n_bins_Y, :n_bins_X]> \
            self.pmftxytptr.getPCF().get()
        return np.asarray(PCF)

    @property
    def X(self):
        cdef unsigned int n_bins_X = self.pmftxytptr.getNBinsX()
        cdef const float[::1] X = \
            <float[:n_bins_X]> self.pmftxytptr.getX().get()
        return np.asarray(X)

    @property
    def Y(self):
        cdef unsigned int n_bins_Y = self.pmftxytptr.getNBinsY()
        cdef const float[::1] Y = \
            <float[:n_bins_Y]> self.pmftxytptr.getY().get()
        return np.asarray(Y)

    @property
    def T(self):
        cdef unsigned int n_bins_T = self.pmftxytptr.getNBinsT()
        cdef const float[::1] T = \
            <float[:n_bins_T]> self.pmftxytptr.getT().get()
        return np.asarray(T)

    @property
    def jacobian(self):
        return self.pmftxytptr.getJacobian()

    @property
    def n_bins_X(self):
        return self.pmftxytptr.getNBinsX()

    @property
    def n_bins_Y(self):
        return self.pmftxytptr.getNBinsY()

    @property
    def n_bins_T(self):
        return self.pmftxytptr.getNBinsT()

    def __repr__(self):
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, n_x={n_x}, "
                "n_y={n_y}, n_t={n_t})").format(cls=type(self).__name__,
                                                x_max=self.xmax,
                                                y_max=self.ymax,
                                                n_x=self.n_bins_X,
                                                n_y=self.n_bins_Y,
                                                n_t=self.n_bins_T)

    def __str__(self):
        return repr(self)


cdef class PMFTXY2D(_PMFT):
    R"""Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in
    coordinates :math:`x`, :math:`y` listed in the ``X`` and ``Y`` arrays.

    The values of :math:`x` and :math:`y` at which to compute the PCF are
    controlled by ``x_max``, ``y_max``, ``n_x``, and ``n_y`` parameters to the
    constructor. The ``x_max`` and ``y_max`` parameters determine the
    minimum/maximum distance at which to compute the PCF and ``n_x`` and
    ``n_y`` are the number of bins in :math:`x` and :math:`y`.

    .. note::
        **2D:** :class:`freud.pmft.PMFTXY2D` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        n_x (unsigned int):
            Number of bins in :math:`x`.
        n_y (unsigned int):
            Number of bins in :math:`y`.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        bin_counts (:math:`\left(N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            Bin counts.
        PCF (:math:`\left(N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\left(N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_cut (float):
            The cutoff used in the cell list.
        X (:math:`\left(N_{x}\right)` :class:`numpy.ndarray`):
            The array of :math:`x`-values for the PCF histogram.
        Y (:math:`\left(N_{y}\right)` :class:`numpy.ndarray`):
            The array of :math:`y`-values for the PCF histogram.
        jacobian (float):
            The Jacobian used in the PMFT.
        n_bins_X (unsigned int):
            The number of bins in the :math:`x`-dimension of the histogram.
        n_bins_Y (unsigned int):
            The number of bins in the :math:`y`-dimension of the histogram.
    """  # noqa: E501
    cdef freud._pmft.PMFTXY2D * pmftxy2dptr
    cdef xmax
    cdef ymax

    def __cinit__(self, x_max, y_max, n_x, n_y):
        if type(self) is PMFTXY2D:
            self.pmftxy2dptr = self.pmftptr = new freud._pmft.PMFTXY2D(
                x_max, y_max, n_x, n_y)
            self.rmax = np.sqrt(x_max**2 + y_max**2)
            self.xmax = x_max
            self.ymax = y_max

    def __dealloc__(self):
        if type(self) is PMFTXY2D:
            del self.pmftxy2dptr

    @Compute._compute()
    def accumulate(self, box, ref_points, ref_orientations, points=None,
                   orientations=None, nlist=None):
        R"""Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef freud.box.Box b = freud.common.convert_box(box)

        if not b.dimensions == 2:
            raise ValueError("Your box must be 2-dimensional!")

        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))

        ref_orientations = freud.common.convert_array(
            ref_orientations.squeeze(), shape=(ref_points.shape[0], ))

        points = freud.common.convert_array(points, shape=(None, 3))

        orientations = freud.common.convert_array(
            orientations.squeeze(), shape=(points.shape[0], ))

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points = points
        cdef const float[::1] l_ref_orientations = ref_orientations
        cdef const float[::1] l_orientations = orientations
        cdef unsigned int nRef = l_ref_points.shape[0]
        cdef unsigned int nP = l_points.shape[0]
        with nogil:
            self.pmftxy2dptr.accumulate(dereference(b.thisptr),
                                        nlist_.get_ptr(),
                                        <vec3[float]*> &l_ref_points[0, 0],
                                        <float*> &l_ref_orientations[0],
                                        nRef,
                                        <vec3[float]*> &l_points[0, 0],
                                        <float*> &l_orientations[0],
                                        nP)
        return self

    @Compute._compute()
    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        R"""Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 1) or (:math:`N_{particles}`,) :class:`numpy.ndarray`, optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    @Compute._computed_property()
    def bin_counts(self):
        cdef unsigned int n_bins_Y = self.pmftxy2dptr.getNBinsY()
        cdef unsigned int n_bins_X = self.pmftxy2dptr.getNBinsX()
        cdef const unsigned int[:, ::1] bin_counts = \
            <unsigned int[:n_bins_Y, :n_bins_X]> \
            self.pmftxy2dptr.getBinCounts().get()
        return np.asarray(bin_counts, dtype=np.uint32)

    @Compute._computed_property()
    def PCF(self):
        cdef unsigned int n_bins_Y = self.pmftxy2dptr.getNBinsY()
        cdef unsigned int n_bins_X = self.pmftxy2dptr.getNBinsX()
        cdef const float[:, ::1] PCF = \
            <float[:n_bins_Y, :n_bins_X]> \
            self.pmftxy2dptr.getPCF().get()
        return np.asarray(PCF)

    @property
    def X(self):
        cdef unsigned int n_bins_X = self.pmftxy2dptr.getNBinsX()
        cdef const float[::1] X = \
            <float[:n_bins_X]> self.pmftxy2dptr.getX().get()
        return np.asarray(X)

    @property
    def Y(self):
        cdef unsigned int n_bins_Y = self.pmftxy2dptr.getNBinsY()
        cdef const float[::1] Y = \
            <float[:n_bins_Y]> self.pmftxy2dptr.getY().get()
        return np.asarray(Y)

    @property
    def n_bins_X(self):
        return self.pmftxy2dptr.getNBinsX()

    @property
    def n_bins_Y(self):
        return self.pmftxy2dptr.getNBinsY()

    @property
    def jacobian(self):
        return self.pmftxy2dptr.getJacobian()

    def __repr__(self):
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, n_x={n_x}, "
                "n_y={n_y})").format(cls=type(self).__name__,
                                     x_max=self.xmax,
                                     y_max=self.ymax,
                                     n_x=self.n_bins_X,
                                     n_y=self.n_bins_Y)

    def __str__(self):
        return repr(self)

    def _repr_png_(self):
        import plot
        return plot.ax_to_bytes(self.plot())

    def plot(self, ax=None):
        """Plot PMFTXY2D.

        Args:
            ax (:class:`matplotlib.axes`): Axis to plot on. If :code:`None`,
                make a new figure and axis. (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes`): Axis with the plot.
        """
        import plot
        return plot.pmft_plot(self, ax)


cdef class PMFTXYZ(_PMFT):
    R"""Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in
    coordinates :math:`x`, :math:`y`, :math:`z`, listed in the ``X``, ``Y``,
    and ``Z`` arrays.

    The values of :math:`x, y, z` at which to compute the PCF are controlled by
    ``x_max``, ``y_max``, ``z_max``, ``n_x``, ``n_y``, and ``n_z`` parameters
    to the constructor. The ``x_max``, ``y_max``, and ``z_max`` parameters]
    determine the minimum/maximum distance at which to compute the PCF and
    ``n_x``, ``n_y``, and ``n_z`` are the number of bins in :math:`x, y, z`.

    .. note::
        3D: :class:`freud.pmft.PMFTXYZ` is only defined for 3D systems.
        The points must be passed in as :code:`[x, y, z]`.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float):
            Maximum :math:`x` distance at which to compute the PMFT.
        y_max (float):
            Maximum :math:`y` distance at which to compute the PMFT.
        z_max (float):
            Maximum :math:`z` distance at which to compute the PMFT.
        n_x (unsigned int):
            Number of bins in :math:`x`.
        n_y (unsigned int):
            Number of bins in :math:`y`.
        n_z (unsigned int):
            Number of bins in :math:`z`.
        shiftvec (list):
            Vector pointing from ``[0, 0, 0]`` to the center of the PMFT.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        bin_counts (:math:`\left(N_{z}, N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            Bin counts.
        PCF (:math:`\left(N_{z}, N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\left(N_{z}, N_{y}, N_{x}\right)` :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_cut (float):
            The cutoff used in the cell list.
        X (:math:`\left(N_{x}\right)` :class:`numpy.ndarray`):
            The array of :math:`x`-values for the PCF histogram.
        Y (:math:`\left(N_{y}\right)` :class:`numpy.ndarray`):
            The array of :math:`y`-values for the PCF histogram.
        Z (:math:`\left(N_{z}\right)` :class:`numpy.ndarray`):
            The array of :math:`z`-values for the PCF histogram.
        jacobian (float):
            The Jacobian used in the PMFT.
        n_bins_X (unsigned int):
            The number of bins in the :math:`x`-dimension of the histogram.
        n_bins_Y (unsigned int):
            The number of bins in the :math:`y`-dimension of the histogram.
        n_bins_Z (unsigned int):
            The number of bins in the :math:`z`-dimension of the histogram.
    """  # noqa: E501
    cdef freud._pmft.PMFTXYZ * pmftxyzptr
    cdef shiftvec
    cdef xmax
    cdef ymax
    cdef zmax

    def __cinit__(self, x_max, y_max, z_max, n_x, n_y, n_z,
                  shiftvec=[0, 0, 0]):
        cdef vec3[float] c_shiftvec
        if type(self) is PMFTXYZ:
            c_shiftvec = vec3[float](
                shiftvec[0], shiftvec[1], shiftvec[2])
            self.pmftxyzptr = self.pmftptr = new freud._pmft.PMFTXYZ(
                x_max, y_max, z_max, n_x, n_y, n_z, c_shiftvec)
            self.shiftvec = np.array(shiftvec, dtype=np.float32)
            self.rmax = np.sqrt(x_max**2 + y_max**2 + z_max**2)
            self.xmax = x_max
            self.ymax = y_max
            self.zmax = z_max

    def __dealloc__(self):
        if type(self) is PMFTXYZ:
            del self.pmftxyzptr

    @Compute._compute()
    def accumulate(self, box, ref_points, ref_orientations, points=None,
                   orientations=None, face_orientations=None, nlist=None):
        R"""Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Reference orientations as quaternions used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`, optional):
                Orientations as quaternions used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
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
        cdef freud.box.Box b = freud.common.convert_box(box)

        if not b.dimensions == 3:
            raise ValueError("Your box must be 3-dimensional!")

        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))
        ref_orientations = freud.common.convert_array(
            ref_orientations, shape=(ref_points.shape[0], 4))

        points = freud.common.convert_array(points, shape=(None, 3))
        points = points - self.shiftvec.reshape(1, 3)

        orientations = freud.common.convert_array(
            orientations, shape=(points.shape[0], 4))

        # handle multiple ways to input
        if face_orientations is None:
            # set to unit quaternion, q = [1, 0, 0, 0]
            face_orientations = np.zeros(
                shape=(ref_points.shape[0], 1, 4), dtype=np.float32)
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
                    shape=(ref_points.shape[0],
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
                        1, ref_points.shape[0]):
                    raise ValueError(
                        "If provided as a 3D array, the first dimension of "
                        "the face_orientations array must be either of "
                        "size 1 or N_particles")
                elif face_orientations.shape[0] == 1:
                    face_orientations = np.repeat(
                        face_orientations, ref_points.shape[0], axis=0)

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points = points
        cdef const float[:, ::1] l_ref_orientations = ref_orientations
        cdef const float[:, ::1] l_orientations = orientations
        cdef const float[:, :, ::1] l_face_orientations = face_orientations
        cdef unsigned int nRef = l_ref_points.shape[0]
        cdef unsigned int nP = l_points.shape[0]
        cdef unsigned int nFaces = l_face_orientations.shape[1]
        with nogil:
            self.pmftxyzptr.accumulate(
                dereference(b.thisptr),
                nlist_.get_ptr(),
                <vec3[float]*> &l_ref_points[0, 0],
                <quat[float]*> &l_ref_orientations[0, 0],
                nRef,
                <vec3[float]*> &l_points[0, 0],
                <quat[float]*> &l_orientations[0, 0],
                nP,
                <quat[float]*> &l_face_orientations[0, 0, 0],
                nFaces)
        return self

    @Compute._compute()
    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, face_orientations=None, nlist=None):
        R"""Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Reference orientations as quaternions used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`, optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`, optional):
                Orientations as quaternions used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
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
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, face_orientations, nlist)
        return self

    @Compute._computed_property()
    def bin_counts(self):
        cdef unsigned int n_bins_Z = self.pmftxyzptr.getNBinsZ()
        cdef unsigned int n_bins_Y = self.pmftxyzptr.getNBinsY()
        cdef unsigned int n_bins_X = self.pmftxyzptr.getNBinsX()
        cdef const unsigned int[:, :, ::1] bin_counts = \
            <unsigned int[:n_bins_Z, :n_bins_Y, :n_bins_X]> \
            self.pmftxyzptr.getBinCounts().get()
        return np.asarray(bin_counts, dtype=np.uint32)

    @Compute._computed_property()
    def PCF(self):
        cdef unsigned int n_bins_Z = self.pmftxyzptr.getNBinsZ()
        cdef unsigned int n_bins_Y = self.pmftxyzptr.getNBinsY()
        cdef unsigned int n_bins_X = self.pmftxyzptr.getNBinsX()
        cdef const float[:, :, ::1] PCF = \
            <float[:n_bins_Z, :n_bins_Y, :n_bins_X]> \
            self.pmftxyzptr.getPCF().get()
        return np.asarray(PCF)

    @property
    def X(self):
        cdef unsigned int n_bins_X = self.pmftxyzptr.getNBinsX()
        cdef const float[::1] X = \
            <float[:n_bins_X]> self.pmftxyzptr.getX().get()
        return np.asarray(X) + self.shiftvec[0]

    @property
    def Y(self):
        cdef unsigned int n_bins_Y = self.pmftxyzptr.getNBinsY()
        cdef const float[::1] Y = \
            <float[:n_bins_Y]> self.pmftxyzptr.getY().get()
        return np.asarray(Y) + self.shiftvec[1]

    @property
    def Z(self):
        cdef unsigned int n_bins_Z = self.pmftxyzptr.getNBinsZ()
        cdef const float[::1] Z = \
            <float[:n_bins_Z]> self.pmftxyzptr.getZ().get()
        return np.asarray(Z) + self.shiftvec[2]

    @property
    def n_bins_X(self):
        return self.pmftxyzptr.getNBinsX()

    @property
    def n_bins_Y(self):
        return self.pmftxyzptr.getNBinsY()

    @property
    def n_bins_Z(self):
        return self.pmftxyzptr.getNBinsZ()

    @property
    def jacobian(self):
        return self.pmftxyzptr.getJacobian()

    def __repr__(self):
        return ("freud.pmft.{cls}(x_max={x_max}, y_max={y_max}, "
                "z_max={z_max}, n_x={n_x}, n_y={n_y}, n_z={n_z}, "
                "shiftvec={shiftvec})").format(
                    cls=type(self).__name__,
                    x_max=self.xmax,
                    y_max=self.ymax,
                    z_max=self.zmax,
                    n_x=self.n_bins_X,
                    n_y=self.n_bins_Y,
                    n_z=self.n_bins_Z,
                    shiftvec=self.shiftvec.tolist())

    def __str__(self):
        return repr(self)
