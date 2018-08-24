# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The PMFT Module allows for the calculation of the Potential of Mean Force and
Torque (PMFT) [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in a number of
different coordinate systems. The PMFT is defined as the negative algorithm of
positional correlation function (PCF). A given set of reference points is given
around which the PCF is computed and averaged in a sea of data points. The
resulting values are accumulated in a PCF array listing the value of the PCF at
a discrete set of points. The specific points are determined by the particular
coordinate system used to represent the system.

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
"""

import numpy as np
import freud.common
import freud.locality
import warnings
from freud.errors import FreudDeprecationWarning

from freud.util._VectorMath cimport vec3, quat
from libc.string cimport memcpy
from cython.operator cimport dereference

cimport freud._pmft
cimport freud.locality
cimport freud.box

cimport numpy as np


# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class _PMFT:
    """Compute the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ for a
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

    @property
    def box(self):
        return freud.box.BoxFromCPP(self.pmftptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    def reset(self):
        """Resets the values of the PCF histograms in memory."""
        self.pmftptr.reset()

    def resetPCF(self):
        warnings.warn("Use .reset() instead of this method. "
                      "This method will be removed in the future.",
                      FreudDeprecationWarning)
        self.reset()

    def reducePCF(self):
        warnings.warn("This method is automatically called internally. It "
                      "will be removed in the future.",
                      FreudDeprecationWarning)
        self.pmftptr.reducePCF()

    @property
    def PMFT(self):
        return -np.log(np.copy(self.PCF))

    def getPMFT(self):
        warnings.warn("The getPMFT function is deprecated in favor "
                      "of the PMFT class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.PMFT

    @property
    def r_cut(self):
        cdef float r_cut = self.pmftptr.getRCut()
        return r_cut

    def getRCut(self):
        warnings.warn("The getRCut function is deprecated in favor "
                      "of the r_cut class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.r_cut


cdef class PMFTR12(_PMFT):
    """Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in a 2D
    system described by :math:`r`, :math:`\\theta_1`, :math:`\\theta_2`.

    .. note::
        **2D:** :py:class:`freud.pmft.PMFTR12` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        r_max (float):
            Maximum distance at which to compute the PMFT.
        n_r (unsigned int):
            Number of bins in r.
        n_t1 (unsigned int):
            Number of bins in t1.
        n_t2 (unsigned int):
            Number of bins in t2.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        bin_counts (:math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)`):
            Bin counts.
        PCF (:math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)`):
            The positional correlation function.
        PMFT (:math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)`):
            The potential of mean force and torque.
        r_cut (float):
            The cutoff used in the cell list.
        R (:math:`\\left(N_{r}\\right)` :class:`numpy.ndarray`):
            The array of r-values for the PCF histogram.
        T1 (:math:`\\left(N_{\\theta1}\\right)` :class:`numpy.ndarray`):
            The array of T1-values for the PCF histogram.
        T2 (:math:`\\left(N_{\\theta2}\\right)` :class:`numpy.ndarray`):
            The array of T2-values for the PCF histogram.
        inverse_jacobian (:math:`\\left(N_{r}, N_{\\theta2}, \
        N_{\\theta1}\\right)`):
            The inverse Jacobian used in the PMFT.
        n_bins_r (unsigned int):
            The number of bins in the r-dimension of histogram.
        n_bins_T1 (unsigned int):
            The number of bins in the T1-dimension of histogram.
        n_bins_T2 (unsigned int):
            The number of bins in the T2-dimension of histogram.
    """
    cdef freud._pmft.PMFTR12 * pmftr12ptr

    def __cinit__(self, r_max, n_r, n_t1, n_t2):
        if type(self) is PMFTR12:
            self.pmftr12ptr = self.pmftptr = new freud._pmft.PMFTR12(
                r_max, n_r, n_t1, n_t2)
            self.rmax = r_max

    def __dealloc__(self):
        if type(self) is PMFTR12:
            del self.pmftr12ptr

    def accumulate(self, box, ref_points, ref_orientations, points=None,
                   orientations=None, nlist=None):
        """Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`,
            optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`,
            optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        if points is None:
            points = ref_points
        if orientations is None:
            orientations = ref_orientations

        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
            ref_orientations, 1, dtype=np.float32, contiguous=True,
            array_name="ref_orientations")

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
            orientations, 1, dtype=np.float32, contiguous=True,
            array_name="orientations")

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        with nogil:
            self.pmftr12ptr.accumulate(dereference(b.thisptr),
                                       nlist_.get_ptr(),
                                       <vec3[float]*> l_ref_points.data,
                                       <float*> l_ref_orientations.data,
                                       nRef,
                                       <vec3[float]*> l_points.data,
                                       <float*> l_orientations.data,
                                       nP)
        return self

    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        """Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`,
            optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`,
            optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """
        self.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    @property
    def bin_counts(self):
        cdef unsigned int * bin_counts = self.pmftr12ptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.pmftr12ptr.getNBinsR()
        nbins[1] = <np.npy_intp> self.pmftr12ptr.getNBinsT2()
        nbins[2] = <np.npy_intp> self.pmftr12ptr.getNBinsT1()
        cdef np.ndarray[np.uint32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32,
                                         <void*> bin_counts)
        return result

    def getBinCounts(self):
        warnings.warn("The getBinCounts function is deprecated in favor "
                      "of the bin_counts class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bin_counts

    @property
    def PCF(self):
        cdef float * pcf = self.pmftr12ptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.pmftr12ptr.getNBinsR()
        nbins[1] = <np.npy_intp> self.pmftr12ptr.getNBinsT2()
        nbins[2] = <np.npy_intp> self.pmftr12ptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*> pcf)
        return result

    def getPCF(self):
        warnings.warn("The getPCF function is deprecated in favor "
                      "of the PCF class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.PCF

    @property
    def R(self):
        cdef float * r = self.pmftr12ptr.getR().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftr12ptr.getNBinsR()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> r)
        return result

    def getR(self):
        warnings.warn("The getR function is deprecated in favor "
                      "of the R class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.R

    @property
    def T1(self):
        cdef float * T1 = self.pmftr12ptr.getT1().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftr12ptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> T1)
        return result

    def getT1(self):
        warnings.warn("The getT1 function is deprecated in favor "
                      "of the T1 class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.T1

    @property
    def T2(self):
        cdef float * T2 = self.pmftr12ptr.getT2().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftr12ptr.getNBinsT2()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> T2)
        return result

    def getT2(self):
        warnings.warn("The getT2 function is deprecated in favor "
                      "of the T2 class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.T2

    @property
    def inverse_jacobian(self):
        cdef float * inv_jac = self.pmftr12ptr.getInverseJacobian().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.pmftr12ptr.getNBinsR()
        nbins[1] = <np.npy_intp> self.pmftr12ptr.getNBinsT2()
        nbins[2] = <np.npy_intp> self.pmftr12ptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32,
                                         <void*> inv_jac)
        return result

    def getInverseJacobian(self):
        warnings.warn("The getInverseJacobian function is deprecated in favor "
                      "of the inverse_jacobian class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.inverse_jacobian

    @property
    def n_bins_r(self):
        cdef unsigned int r = self.pmftr12ptr.getNBinsR()
        return r

    def getNBinsR(self):
        warnings.warn("The getNBinsR function is deprecated in favor "
                      "of the n_bins_r class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_r

    @property
    def n_bins_T1(self):
        cdef unsigned int T1 = self.pmftr12ptr.getNBinsT1()
        return T1

    def getNBinsT1(self):
        warnings.warn("The getNBinsT1 function is deprecated in favor "
                      "of the n_bins_T1 class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_T1

    @property
    def n_bins_T2(self):
        cdef unsigned int T2 = self.pmftr12ptr.getNBinsT2()
        return T2

    def getNBinsT2(self):
        warnings.warn("The getNBinsT2 function is deprecated in favor "
                      "of the n_bins_T2 class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_T2

cdef class PMFTXYT(_PMFT):
    """Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ for
    systems described by coordinates :math:`x`, :math:`y`, :math:`\\theta`
    listed in the x, y, and t arrays.

    The values of x, y, t to compute the PCF at are controlled by x_max, y_max
    and n_bins_x, n_bins_y, n_bins_t parameters to the constructor.
    The x_max and y_max parameters determine the minimum/maximum x, y values
    (:math:`\\min \\left(\\theta \\right) = 0`,
    (:math:`\\max \\left( \\theta \\right) = 2\\pi`) at which to compute the
    PCF and n_bins_x, n_bins_y, n_bins_t is the number of bins in x, y, t.

    .. note::
        **2D:** :py:class:`freud.pmft.PMFTXYT` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float):
            Maximum x distance at which to compute the PMFT.
        y_max (float):
            Maximum y distance at which to compute the PMFT.
        n_x (unsigned int):
            Number of bins in x.
        n_y (unsigned int):
            Number of bins in y.
        n_t (unsigned int):
            Number of bins in t.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        bin_counts (:math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)` \
        :class:`numpy.ndarray`):
            Bin counts.
        PCF (:math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)` \
        :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)` \
        :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_cut (float):
            The cutoff used in the cell list.
        X (:math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`):
            The array of x-values for the PCF histogram.
        Y (:math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`):
            The array of y-values for the PCF histogram.
        T (:math:`\\left(N_{\\theta}\\right)` :class:`numpy.ndarray`):
            The array of T-values for the PCF histogram.
        jacobian (float):
            The Jacobian used in the PMFT.
        n_bins_x (unsigned int):
            The number of bins in the x-dimension of histogram.
        n_bins_y (unsigned int):
            The number of bins in the y-dimension of histogram.
        n_bins_T (unsigned int):
            The number of bins in the T-dimension of histogram.
    """
    cdef freud._pmft.PMFTXYT * pmftxytptr

    def __cinit__(self, x_max, y_max, n_x, n_y, n_t):
        if type(self) is PMFTXYT:
            self.pmftxytptr = self.pmftptr = new freud._pmft.PMFTXYT(
                x_max, y_max, n_x, n_y, n_t)
            self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXYT:
            del self.pmftxytptr

    def accumulate(self, box, ref_points, ref_orientations, points=None,
                   orientations=None, nlist=None):
        """Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`,
            optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`,
            optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
            ref_orientations, 1, dtype=np.float32, contiguous=True,
            array_name="ref_orientations")

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
            orientations, 1, dtype=np.float32, contiguous=True,
            array_name="orientations")

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        with nogil:
            self.pmftxytptr.accumulate(dereference(b.thisptr),
                                       nlist_.get_ptr(),
                                       <vec3[float]*> l_ref_points.data,
                                       <float*> l_ref_orientations.data,
                                       nRef,
                                       <vec3[float]*> l_points.data,
                                       <float*> l_orientations.data,
                                       nP)
        return self

    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        """Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`,
            optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`,
            optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """
        self.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    @property
    def bin_counts(self):
        cdef unsigned int * bin_counts = self.pmftxytptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.pmftxytptr.getNBinsT()
        nbins[1] = <np.npy_intp> self.pmftxytptr.getNBinsY()
        nbins[2] = <np.npy_intp> self.pmftxytptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32,
                                         <void*> bin_counts)
        return result

    def getBinCounts(self):
        warnings.warn("The getBinCounts function is deprecated in favor "
                      "of the bin_counts class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bin_counts

    @property
    def PCF(self):
        cdef float * pcf = self.pmftxytptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.pmftxytptr.getNBinsT()
        nbins[1] = <np.npy_intp> self.pmftxytptr.getNBinsY()
        nbins[2] = <np.npy_intp> self.pmftxytptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*> pcf)
        return result

    def getPCF(self):
        warnings.warn("The getPCF function is deprecated in favor "
                      "of the PCF class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.PCF

    @property
    def X(self):
        cdef float * x = self.pmftxytptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftxytptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> x)
        return result

    def getX(self):
        warnings.warn("The getX function is deprecated in favor "
                      "of the X class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.X

    @property
    def Y(self):
        cdef float * y = self.pmftxytptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftxytptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> y)
        return result

    def getY(self):
        warnings.warn("The getY function is deprecated in favor "
                      "of the Y class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Y

    @property
    def T(self):
        cdef float * t = self.pmftxytptr.getT().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftxytptr.getNBinsT()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> t)
        return result

    def getT(self):
        warnings.warn("The getT function is deprecated in favor "
                      "of the T class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.T

    @property
    def jacobian(self):
        cdef float j = self.pmftxytptr.getJacobian()
        return j

    def getJacobian(self):
        warnings.warn("The getJacobian function is deprecated in favor "
                      "of the jacobian class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.jacobian

    @property
    def n_bins_X(self):
        cdef unsigned int x = self.pmftxytptr.getNBinsX()
        return x

    def getNBinsX(self):
        warnings.warn("The getNBinsX function is deprecated in favor "
                      "of the n_bins_X class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_X

    @property
    def n_bins_Y(self):
        cdef unsigned int y = self.pmftxytptr.getNBinsY()
        return y

    def getNBinsY(self):
        warnings.warn("The getNBinsY function is deprecated in favor "
                      "of the n_bins_Y class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_Y

    @property
    def n_bins_T(self):
        cdef unsigned int t = self.pmftxytptr.getNBinsT()
        return t

    def getNBinsT(self):
        warnings.warn("The getNBinsT function is deprecated in favor "
                      "of the n_bins_T class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_T


cdef class PMFTXY2D(_PMFT):
    """Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in
    coordinates :math:`x`, :math:`y` listed in the x and y arrays.

    The values of x and y to compute the PCF at are controlled by x_max, y_max,
    n_x, and n_y parameters to the constructor.
    The x_max and y_max parameters determine the minimum/maximum distance at
    which to compute the PCF and n_x and n_y are the number of bins in x and y.

    .. note::
        **2D:** :py:class:`freud.pmft.PMFTXY2D` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float):
            Maximum x distance at which to compute the PMFT.
        y_max (float):
            Maximum y distance at which to compute the PMFT.
        n_x (unsigned int):
            Number of bins in x.
        n_y (unsigned int):
            Number of bins in y.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        bin_counts (:math:`\\left(N_{y}, N_{x}\\right)` \
        :class:`numpy.ndarray`):
            Bin counts.
        PCF (:math:`\\left(N_{y}, N_{x}\\right)` :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\\left(N_{y}, N_{x}\\right)` :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_cut (float):
            The cutoff used in the cell list.
        X (:math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`):
            The array of x-values for the PCF histogram.
        Y (:math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`):
            The array of y-values for the PCF histogram.
        jacobian (float):
            The Jacobian used in the PMFT.
        n_bins_x (unsigned int):
            The number of bins in the x-dimension of histogram.
        n_bins_y (unsigned int):
            The number of bins in the y-dimension of histogram.
    """
    cdef freud._pmft.PMFTXY2D * pmftxy2dptr

    def __cinit__(self, x_max, y_max, n_x, n_y):
        if type(self) is PMFTXY2D:
            self.pmftxy2dptr = self.pmftptr = new freud._pmft.PMFTXY2D(
                x_max, y_max, n_x, n_y)
            self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXY2D:
            del self.pmftxy2dptr

    def accumulate(self, box, ref_points, ref_orientations, points=None,
                   orientations=None, nlist=None):
        """Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`,
            optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`,
            optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
            ref_orientations, 1, dtype=np.float32, contiguous=True,
            array_name="ref_orientations")

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
            orientations, 1, dtype=np.float32, contiguous=True,
            array_name="orientations")

        defaulted_nlist = freud.locality.make_default_nlist(
            b, ref_points, points, self.rmax, nlist, None)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        with nogil:
            self.pmftxy2dptr.accumulate(dereference(b.thisptr),
                                        nlist_.get_ptr(),
                                        <vec3[float]*> l_ref_points.data,
                                        <float*> l_ref_orientations.data,
                                        n_ref,
                                        <vec3[float]*> l_points.data,
                                        <float*> l_orientations.data,
                                        n_p)
        return self

    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, nlist=None):
        """Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`,
            optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`,
            optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """
        self.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    @property
    def PCF(self):
        cdef float * pcf = self.pmftxy2dptr.getPCF().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> self.pmftxy2dptr.getNBinsY()
        nbins[1] = <np.npy_intp> self.pmftxy2dptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32, <void*> pcf)
        return result

    def getPCF(self):
        warnings.warn("The getPCF function is deprecated in favor "
                      "of the PCF class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.PCF

    @property
    def bin_counts(self):
        cdef unsigned int * bin_counts = self.pmftxy2dptr.getBinCounts().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> self.pmftxy2dptr.getNBinsY()
        nbins[1] = <np.npy_intp> self.pmftxy2dptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32,
                                         <void*> bin_counts)
        return result

    def getBinCounts(self):
        warnings.warn("The getBinCounts function is deprecated in favor "
                      "of the bin_counts class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bin_counts

    @property
    def X(self):
        cdef float * x = self.pmftxy2dptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftxy2dptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> x)
        return result

    def getX(self):
        warnings.warn("The getX function is deprecated in favor "
                      "of the X class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.X

    @property
    def Y(self):
        cdef float * y = self.pmftxy2dptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftxy2dptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> y)
        return result

    def getY(self):
        warnings.warn("The getY function is deprecated in favor "
                      "of the Y class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Y

    @property
    def n_bins_X(self):
        cdef unsigned int x = self.pmftxy2dptr.getNBinsX()
        return x

    def getNBinsX(self):
        warnings.warn("The getNBinsX function is deprecated in favor "
                      "of the n_bins_X class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_X

    @property
    def n_bins_Y(self):
        cdef unsigned int y = self.pmftxy2dptr.getNBinsY()
        return y

    def getNBinsY(self):
        warnings.warn("The getNBinsY function is deprecated in favor "
                      "of the n_bins_Y class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_Y

    @property
    def jacobian(self):
        cdef float j = self.pmftxy2dptr.getJacobian()
        return j

    def getJacobian(self):
        warnings.warn("The getJacobian function is deprecated in favor "
                      "of the jacobian class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.jacobian


cdef class PMFTXYZ(_PMFT):
    """Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in
    coordinates :math:`x`, :math:`y`, :math:`z`, listed in the x, y, and z
    arrays.

    The values of x, y, z to compute the PCF at are controlled by x_max, y_max,
    z_max, n_x, n_y, and n_z parameters to the constructor. The x_max, y_max,
    and z_max parameters determine the minimum/maximum distance at which to
    compute the PCF and n_x, n_y, and n_z are the number of bins in x, y, z.

    .. note::
        3D: :py:class:`freud.pmft.PMFTXYZ` is only defined for 3D systems.
        The points must be passed in as :code:`[x, y, z]`.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float):
            Maximum x distance at which to compute the PMFT.
        y_max (float):
            Maximum y distance at which to compute the PMFT.
        z_max (float):
            Maximum z distance at which to compute the PMFT.
        n_x (unsigned int):
            Number of bins in x.
        n_y (unsigned int):
            Number of bins in y.
        n_z (unsigned int):
            Number of bins in z.
        shiftvec (list):
            Vector pointing from [0,0,0] to the center of the PMFT.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        bin_counts (:math:`\\left(N_{z}, N_{y}, N_{x}\\right)` \
        :class:`numpy.ndarray`):
            Bin counts.
        PCF (:math:`\\left(N_{z}, N_{y}, N_{x}\\right)` \
        :class:`numpy.ndarray`):
            The positional correlation function.
        PMFT (:math:`\\left(N_{z}, N_{y}, N_{x}\\right)` \
        :class:`numpy.ndarray`):
            The potential of mean force and torque.
        r_cut (float):
            The cutoff used in the cell list.
        X (:math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`):
            The array of x-values for the PCF histogram.
        Y (:math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`):
            The array of y-values for the PCF histogram.
        Z (:math:`\\left(N_{z}\\right)` :class:`numpy.ndarray`):
            The array of z-values for the PCF histogram.
        jacobian (float):
            The Jacobian used in the PMFT.
        n_bins_x (unsigned int):
            The number of bins in the x-dimension of histogram.
        n_bins_y (unsigned int):
            The number of bins in the y-dimension of histogram.
        n_bins_z (unsigned int):
            The number of bins in the z-dimension of histogram.
    """
    cdef freud._pmft.PMFTXYZ * pmftxyzptr
    cdef shiftvec

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

    def __dealloc__(self):
        if type(self) is PMFTXYZ:
            del self.pmftxyzptr

    def reset(self):
        """Resets the values of the PCF histograms in memory."""
        self.pmftxyzptr.reset()

    def resetPCF(self):
        warnings.warn("Use .reset() instead of this method. "
                      "This method will be removed in the future.",
                      FreudDeprecationWarning)
        self.reset()

    def accumulate(self, box, ref_points, ref_orientations, points=None,
                   orientations=None, face_orientations=None, nlist=None):
        """Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`,
            optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`,
            optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            face_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`, optional):
                Orientations of particle faces to account for particle
                symmetry. If not supplied by user, unit quaternions will be
                supplied. If a 2D array of shape (:math:`N_f`, 4) or a
                3D array of shape (1, :math:`N_f`, 4) is supplied, the
                supplied quaternions will be broadcast for all particles.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
            ref_points, 2, dtype=np.float32, contiguous=True,
            array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
            ref_orientations, 2, dtype=np.float32, contiguous=True,
            array_name="ref_orientations")
        if ref_orientations.shape[1] != 4:
            raise ValueError(
                "The 2nd dimension must have 4 values: q0, q1, q2, q3")

        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')
        points = points - self.shiftvec.reshape(1, 3)

        orientations = freud.common.convert_array(
            orientations, 2, dtype=np.float32, contiguous=True,
            array_name="orientations")
        if orientations.shape[1] != 4:
            raise ValueError(
                "The 2nd dimension must have 4 values: q0, q1, q2, q3")

        # handle multiple ways to input
        if face_orientations is None:
            # set to unit quaternion q = [1,0,0,0]
            face_orientations = np.zeros(
                shape=(ref_points.shape[0], 1, 4), dtype=np.float32)
            face_orientations[:, :, 0] = 1.0
        else:
            if face_orientations.ndim < 2 or face_orientations.ndim > 3:
                raise ValueError("points must be a 2 or 3 dimensional array")
            face_orientations = freud.common.convert_array(
                face_orientations, face_orientations.ndim,
                dtype=np.float32, contiguous=True,
                array_name=("face_orientations must be a {} "
                            "dimensional array").format(
                                face_orientations.ndim))
            if face_orientations.ndim == 2:
                if face_orientations.shape[1] != 4:
                    raise ValueError(
                        ("2nd dimension for orientations must have 4 values:"
                            "s, x, y, z"))
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
                        "s, x, y, z")
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

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef np.ndarray[float, ndim=3] l_face_orientations = face_orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef unsigned int nFaces = <unsigned int> face_orientations.shape[1]
        with nogil:
            self.pmftxyzptr.accumulate(
                dereference(b.thisptr),
                nlist_.get_ptr(),
                <vec3[float]*> l_ref_points.data,
                <quat[float]*> l_ref_orientations.data,
                nRef,
                <vec3[float]*> l_points.data,
                <quat[float]*> l_orientations.data,
                nP,
                <quat[float]*> l_face_orientations.data,
                nFaces)
        return self

    def compute(self, box, ref_points, ref_orientations, points=None,
                orientations=None, face_orientations=None, nlist=None):
        """Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in computation.
            ref_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`):
                Reference orientations as angles used in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`,
            optional):
                Points used in computation. Uses :code:`ref_points` if not
                provided or :code:`None`.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`,
            optional):
                Orientations as angles used in computation. Uses
                :code:`ref_orientations` if not provided or :code:`None`.
            face_orientations ((:math:`N_{particles}`, 4) \
            :class:`numpy.ndarray`, optional):
                Orientations of particle faces to account for particle
                symmetry. If not supplied by user, unit quaternions will be
                supplied. If a 2D array of shape (:math:`N_f`, 4) or a
                3D array of shape (1, :math:`N_f`, 4) is supplied, the
                supplied quaternions will be broadcast for all particles.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList used to find bonds (Default value =
                :code:`None`).
        """
        self.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, face_orientations, nlist)
        return self

    def reducePCF(self):
        warnings.warn("This method is automatically called internally. It "
                      "will be removed in the future.",
                      FreudDeprecationWarning)
        self.pmftxyzptr.reducePCF()

    @property
    def bin_counts(self):
        cdef unsigned int * bin_counts = self.pmftxyzptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.pmftxyzptr.getNBinsZ()
        nbins[1] = <np.npy_intp> self.pmftxyzptr.getNBinsY()
        nbins[2] = <np.npy_intp> self.pmftxyzptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32,
                                         <void*> bin_counts)
        return result

    def getBinCounts(self):
        warnings.warn("The getBinCounts function is deprecated in favor "
                      "of the bin_counts class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.bin_counts

    @property
    def PCF(self):
        cdef float * pcf = self.pmftxyzptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.pmftxyzptr.getNBinsZ()
        nbins[1] = <np.npy_intp> self.pmftxyzptr.getNBinsY()
        nbins[2] = <np.npy_intp> self.pmftxyzptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*> pcf)
        return result

    def getPCF(self):
        warnings.warn("The getPCF function is deprecated in favor "
                      "of the PCF class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.PCF

    def getPMFT(self):
        warnings.warn("The getPMFT function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return -np.log(np.copy(self.PCF))

    @property
    def X(self):
        cdef float * x = self.pmftxyzptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftxyzptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> x)
        return result + self.shiftvec[0]

    def getX(self):
        warnings.warn("The getx function is deprecated in favor "
                      "of the X class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.X

    @property
    def Y(self):
        cdef float * y = self.pmftxyzptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftxyzptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> y)
        return result + self.shiftvec[1]

    def getY(self):
        warnings.warn("The getY function is deprecated in favor "
                      "of the Y class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Y

    @property
    def Z(self):
        cdef float * z = self.pmftxyzptr.getZ().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.pmftxyzptr.getNBinsZ()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> z)
        return result + self.shiftvec[2]

    def getZ(self):
        warnings.warn("The getZ function is deprecated in favor "
                      "of the Z class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Z

    @property
    def n_bins_X(self):
        cdef unsigned int x = self.pmftxyzptr.getNBinsX()
        return x

    def getNBinsX(self):
        warnings.warn("The getNBinsX function is deprecated in favor "
                      "of the n_bins_X class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_X

    @property
    def n_bins_Y(self):
        cdef unsigned int y = self.pmftxyzptr.getNBinsY()
        return y

    def getNBinsY(self):
        warnings.warn("The getNBinsY function is deprecated in favor "
                      "of the n_bins_Y class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_Y

    @property
    def n_bins_Z(self):
        cdef unsigned int z = self.pmftxyzptr.getNBinsZ()
        return z

    def getNBinsZ(self):
        warnings.warn("The getNBinsZ function is deprecated in favor "
                      "of the Z class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.n_bins_Z

    @property
    def jacobian(self):
        cdef float j = self.pmftxyzptr.getJacobian()
        return j

    def getJacobian(self):
        warnings.warn("The getJacobian function is deprecated in favor "
                      "of the jacobian class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.jacobian
