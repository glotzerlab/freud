# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
cimport freud._box as _box
cimport freud._pmft as pmft
from libc.string cimport memcpy
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
# DTYPE = np.float32
# ctypedef np.float32_t DTYPE_t

cdef class PMFTR12:
    """Computes the PMFT [Cit2]_ for a given set of points.

    A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given :math:`r`, :math:`\\theta_1`,
    :math:`\\theta_2` listed in the r, t1, and t2 arrays.

    The values of r, t1, t2 to compute the pcf at are controlled by r_max and nbins_r, nbins_t1, nbins_t2 parameters
    to the constructor. rmax determines the minimum/maximum r (:math:`\\min \\left( \\theta_1 \\right) =
    \\min \\left( \\theta_2 \\right) = 0`, (:math:`\\max \\left( \\theta_1 \\right) = \\max \\left( \\theta_2 \\right) = 2\\pi`)
    at which to compute the pcf and nbins_r, nbins_t1, nbins_t2 is the number of bins in r, t1, t2.

    .. note:: 2D: This calculation is defined for 2D systems only. However particle positions are still required to be \
    (x, y, 0)

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param r_max: maximum distance at which to compute the pmft
    :param n_r: number of bins in r
    :param n_t1: number of bins in t1
    :param n_t2: number of bins in t2
    :type r_max: float
    :type n_r: unsigned int
    :type n_t1: unsigned int
    :type n_t2: unsigned int

    """
    cdef pmft.PMFTR12 *thisptr
    cdef rmax

    def __cinit__(self, r_max, n_r, n_t1, n_t2):
        self.thisptr = new pmft.PMFTR12(r_max, n_r, n_t1, n_t2)
        self.rmax = r_max

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box()`
        """
        return self.getBox()

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box()`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations, nlist=None):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(ref_points, 2, dtype=np.float32, contiguous=True,
            dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(ref_orientations, 1, dtype=np.float32, contiguous=True,
            dim_message="ref_orientations must be a 1 dimensional array")

        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
            dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(orientations, 1, dtype=np.float32, contiguous=True,
            dim_message="orientations must be a 1 dimensional array")

        defaulted_nlist = make_default_nlist(box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList *nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    nlist_ptr,
                                    <vec3[float]*>l_ref_points.data,
                                    <float*>l_ref_orientations.data,
                                    nRef,
                                    <vec3[float]*>l_points.data,
                                    <float*>l_orientations.data,
                                    nP)
        return self

    def compute(self, box, ref_points, ref_orientations, points, orientations, nlist=None):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations, nlist)
        return self

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTR12.getPCF()`.
        """
        self.thisptr.reducePCF()

    @property
    def bin_counts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.uint32`
        """
        return self.getBinCounts()

    def getBinCounts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.uint32`
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsT2()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[np.uint32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result

    @property
    def PCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getBinCounts()

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsT2()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    @property
    def PMFT(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getPMFT()

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return -np.log(np.copy(self.getPCF()))

    @property
    def R(self):
        """
        Get the array of r-values for the PCF histogram

        :return: bin centers of r-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getR()

    def getR(self):
        """
        Get the array of r-values for the PCF histogram

        :return: bin centers of r-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* r = self.thisptr.getR().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>r)
        return result

    @property
    def T1(self):
        """
        Get the array of T1-values for the PCF histogram

        :return: bin centers of T1-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta1}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getT1()

    def getT1(self):
        """
        Get the array of T1-values for the PCF histogram

        :return: bin centers of T1-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta1}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* T1 = self.thisptr.getT1().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>T1)
        return result

    @property
    def T2(self):
        """
        Get the array of T2-values for the PCF histogram

        :return: bin centers of T2-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta1}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getT2()

    def getT2(self):
        """
        Get the array of T2-values for the PCF histogram

        :return: bin centers of T2-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* T2 = self.thisptr.getT2().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT2()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>T2)
        return result

    @property
    def inverse_jacobian(self):
        """
        Get the array of T2-values for the PCF histogram

        :return: bin centers of T2-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta1}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getInverseJacobian()

    def getInverseJacobian(self):
        """
        Get the inverse jacobian used in the pmft

        :return: Inverse Jacobian
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* inv_jac = self.thisptr.getInverseJacobian().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsT2()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>inv_jac)
        return result

    @property
    def n_bins_r(self):
        """
        Get the number of bins in the r-dimension of histogram

        :return: :math:`N_r`
        :rtype: unsigned int
        """
        return self.getNBinsR()

    def getNBinsR(self):
        """
        Get the number of bins in the r-dimension of histogram

        :return: :math:`N_r`
        :rtype: unsigned int
        """
        cdef unsigned int r = self.thisptr.getNBinsR()
        return r

    @property
    def n_bins_T1(self):
        """
        Get the number of bins in the T1-dimension of histogram

        :return: :math:`N_{\\theta_1}`
        :rtype: unsigned int
        """
        return self.getNBinsT1()

    def getNBinsT1(self):
        """
        Get the number of bins in the T1-dimension of histogram

        :return: :math:`N_{\\theta_1}`
        :rtype: unsigned int
        """
        cdef unsigned int T1 = self.thisptr.getNBinsT1()
        return T1

    @property
    def n_bins_T2(self):
        """
        Get the number of bins in the T2-dimension of histogram

        :return: :math:`N_{\\theta_2}`
        :rtype: unsigned int
        """
        return self.getNBinsT2()

    def getNBinsT2(self):
        """
        Get the number of bins in the T2-dimension of histogram

        :return: :math:`N_{\\theta_2}`
        :rtype: unsigned int
        """
        cdef unsigned int T2 = self.thisptr.getNBinsT2()
        return T2

    @property
    def r_cut(self):
        """
        Get the r_cut value used in the cell list

        :return: r_cut
        :rtype: float
        """
        return self.getRCut()

    def getRCut(self):
        """
        Get the r_cut value used in the cell list

        :return: r_cut
        :rtype: float
        """
        cdef float r_cut = self.thisptr.getRCut()
        return r_cut

cdef class PMFTXYT:
    """Computes the PMFT [Cit2]_ for a given set of points.

    A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given :math:`x`, :math:`y`,
    :math:`\\theta` listed in the x, y, and t arrays.

    The values of x, y, t to compute the pcf at are controlled by x_max, y_max and n_bins_x, n_bins_y, n_bins_t parameters
    to the constructor. x_max, y_max determine the minimum/maximum x, y values (:math:`\\min \\left( \\theta \\right) = 0`,
    (:math:`\\max \\left( \\theta \\right) = 2\\pi`) at which to compute the pcf and n_bins_x, n_bins_y, n_bins_t is the
    number of bins in x, y, t.

    .. note:: 2D: This calculation is defined for 2D systems only. However particle positions are still required to be \
    (x, y, 0)

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to compute the pmft
    :param y_max: maximum y distance at which to compute the pmft
    :param n_x: number of bins in x
    :param n_y: number of bins in y
    :param n_t: number of bins in t
    :type x_max: float
    :type y_max: float
    :type n_x: unsigned int
    :type n_y: unsigned int
    :type n_t: unsigned int

    """
    cdef pmft.PMFTXYT *thisptr
    cdef rmax

    def __cinit__(self, x_max, y_max, n_x, n_y, n_t):
        self.thisptr = new pmft.PMFTXYT(x_max, y_max, n_x, n_y, n_t)
        self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box()`
        """
        return self.getBox()

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations, nlist=None):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(ref_points, 2, dtype=np.float32, contiguous=True,
            dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(ref_orientations, 1, dtype=np.float32, contiguous=True,
            dim_message="ref_orientations must be a 1 dimensional array")

        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
            dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(orientations, 1, dtype=np.float32, contiguous=True,
            dim_message="orientations must be a 1 dimensional array")

        defaulted_nlist = make_default_nlist(box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList *nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    nlist_ptr,
                                    <vec3[float]*>l_ref_points.data,
                                    <float*>l_ref_orientations.data,
                                    nRef,
                                    <vec3[float]*>l_points.data,
                                    <float*>l_orientations.data,
                                    nP)
        return self

    def compute(self, box, ref_points, ref_orientations, points, orientations, nlist=None):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations, nlist)
        return self

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTXYT.getPCF()`.
        """
        self.thisptr.reducePCF()

    @property
    def bin_counts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.uint32`
        """
        return self.getBinCounts()

    def getBinCounts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)`, dtype= :class:`numpy.uint32`
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result

    @property
    def PCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getBinCounts()

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    @property
    def PMFT(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getPMFT()

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        return -np.log(np.copy(self.getPCF()))

    @property
    def X(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getX()

    def getX(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* x = self.thisptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>x)
        return result

    @property
    def Y(self):
        """
        Get the array of y-values for the PCF histogram

        :return: bin centers of y-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getX()

    def getY(self):
        """
        Get the array of y-values for the PCF histogram

        :return: bin centers of y-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* y = self.thisptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>y)
        return result

    @property
    def T(self):
        """
        Get the array of t-values for the PCF histogram

        :return: bin centers of t-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getT()

    def getT(self):
        """
        Get the array of t-values for the PCF histogram

        :return: bin centers of t-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{\\theta}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* t = self.thisptr.getT().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>t)
        return result

    @property
    def jacobian(self):
        """
        Get the jacobian used in the pmft

        :return: Inverse Jacobian
        :rtype: float
        """
        return self.getJacobian()

    def getJacobian(self):
        """
        Get the jacobian used in the pmft

        :return: Inverse Jacobian
        :rtype: float
        """
        cdef float j = self.thisptr.getJacobian()
        return j

    @property
    def n_bins_X(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: :math:`N_x`
        :rtype: unsigned int
        """
        return self.getNBinsX()

    def getNBinsX(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: :math:`N_x`
        :rtype: unsigned int
        """
        cdef unsigned int x = self.thisptr.getNBinsX()
        return x

    @property
    def n_bins_Y(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: :math:`N_y`
        :rtype: unsigned int
        """
        return self.getNBinsY()

    def getNBinsY(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: :math:`N_y`
        :rtype: unsigned int
        """
        cdef unsigned int y = self.thisptr.getNBinsY()
        return y

    @property
    def n_bins_T(self):
        """
        Get the number of bins in the T-dimension of histogram

        :return: :math:`N_{\\theta}`
        :rtype: unsigned int
        """
        return self.getNBinsT()

    def getNBinsT(self):
        """
        Get the number of bins in the t-dimension of histogram

        :return: :math:`N_{\\theta}`
        :rtype: unsigned int
        """
        cdef unsigned int t = self.thisptr.getNBinsT()
        return t

    @property
    def r_cut(self):
        """
        Get the r_cut value used in the cell list

        :return: r_cut
        :rtype: float
        """
        return self.getRCut()

    def getRCut(self):
        """
        Get the r_cut value used in the cell list

        :return: r_cut
        :rtype: float
        """
        cdef float r_cut = self.thisptr.getRCut()
        return r_cut

cdef class PMFTXY2D:
    """Computes the PMFT [Cit2]_ for a given set of points.

    A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given :math:`x`, :math:`y`
    listed in the x and y arrays.

    The values of x and y to compute the pcf at are controlled by x_max, y_max, n_x, and n_y parameters
    to the constructor. x_max and y_max determine the minimum/maximum distance at which to compute the pcf and
    n_x and n_y are the number of bins in x and y.

    .. note:: 2D: This calculation is defined for 2D systems only.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to compute the pmft
    :param y_max: maximum y distance at which to compute the pmft
    :param n_x: number of bins in x
    :param n_y: number of bins in y
    :type x_max: float
    :type y_max: float
    :type n_x: unsigned int
    :type n_y: unsigned int
    """
    cdef pmft.PMFTXY2D *thisptr
    cdef rmax

    def __cinit__(self, x_max, y_max, n_x, n_y):
        self.thisptr = new pmft.PMFTXY2D(x_max, y_max, n_x, n_y)
        self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box()`
        """
        return self.getBox()

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations, nlist=None):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(ref_points, 2, dtype=np.float32, contiguous=True,
            dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(ref_orientations, 1, dtype=np.float32, contiguous=True,
            dim_message="ref_orientations must be a 1 dimensional array")

        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
            dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(orientations, 1, dtype=np.float32, contiguous=True,
            dim_message="orientations must be a 1 dimensional array")

        defaulted_nlist = make_default_nlist(box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList *nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    nlist_ptr,
                                    <vec3[float]*>l_ref_points.data,
                                    <float*>l_ref_orientations.data,
                                    n_ref,
                                    <vec3[float]*>l_points.data,
                                    <float*>l_orientations.data,
                                    n_p)
        return self

    def compute(self, box, ref_points, ref_orientations, points, orientations, nlist=None):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}\\right)`, dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations, nlist)
        return self

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTXY2D.getPCF()`.
        """
        self.thisptr.reducePCF()

    @property
    def PCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getBinCounts()

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}, N_{y}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    @property
    def PMFT(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getPMFT()

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}, N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        return -np.log(np.copy(self.getPCF()))

    @property
    def bin_counts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.uint32`
        """
        return self.getBinCounts()

    def getBinCounts(self):
        """
        Get the raw bin counts (non-normalized).

        :return: Bin Counts
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}, N_{x}\\right)`, dtype= :class:`numpy.uint32`
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result

    @property
    def X(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getX()

    def getX(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* x = self.thisptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>x)
        return result

    @property
    def Y(self):
        """
        Get the array of y-values for the PCF histogram

        :return: bin centers of y-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getX()

    def getY(self):
        """
        Get the array of y-values for the PCF histogram


        :return: bin centers of y-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* y = self.thisptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>y)
        return result

    @property
    def n_bins_X(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: :math:`N_x`
        :rtype: unsigned int
        """
        return self.getNBinsX()

    def getNBinsX(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: :math:`N_x`
        :rtype: unsigned int
        """
        cdef unsigned int x = self.thisptr.getNBinsX()
        return x

    @property
    def n_bins_Y(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: :math:`N_y`
        :rtype: unsigned int
        """
        return self.getNBinsY()

    def getNBinsY(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: :math:`N_y`
        :rtype: unsigned int
        """
        cdef unsigned int y = self.thisptr.getNBinsY()
        return y

    @property
    def jacobian(self):
        """
        Get the jacobian used in the pmft

        :return: Inverse Jacobian
        :rtype: float
        """
        return self.getJacobian()

    def getJacobian(self):
        """
        Get the jacobian

        :return: jacobian
        :rtype: float
        """
        cdef float j = self.thisptr.getJacobian()
        return j

    @property
    def r_cut(self):
        """
        Get the r_cut value used in the cell list

        :return: r_cut
        :rtype: float
        """
        return self.getRCut()

    def getRCut(self):
        """
        Get the r_cut value used in the cell list

        :return: r_cut
        :rtype: float
        """
        cdef float r_cut = self.thisptr.getRCut()
        return r_cut

cdef class PMFTXYZ:
    """Computes the PMFT [Cit2]_ for a given set of points.

    A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given :math:`x`, :math:`y`, :math:`z`,
    listed in the x, y, and z arrays.

    The values of x, y, z to compute the pcf at are controlled by x_max, y_max, z_max, n_x, n_y, and n_z parameters
    to the constructor. x_max, y_max, and z_max determine the minimum/maximum distance at which to compute the pcf and
    n_x, n_y, n_z is the number of bins in x, y, z.

    .. note:: 3D: This calculation is defined for 3D systems only.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param x_max: maximum x distance at which to compute the pmft
    :param y_max: maximum y distance at which to compute the pmft
    :param z_max: maximum z distance at which to compute the pmft
    :param n_x: number of bins in x
    :param n_y: number of bins in y
    :param n_z: number of bins in z
    :param shiftvec: vector pointing from [0,0,0] to the center of the pmft
    :type x_max: float
    :type y_max: float
    :type z_max: float
    :type n_x: unsigned int
    :type n_y: unsigned int
    :type n_z: unsigned int
    :type shiftvec: list
    """
    cdef pmft.PMFTXYZ *thisptr
    cdef shiftvec
    cdef rmax

    def __cinit__(self, x_max, y_max, z_max, n_x, n_y, n_z, shiftvec=[0,0,0]):
        cdef vec3[float] c_shiftvec = vec3[float](shiftvec[0],shiftvec[1],shiftvec[2])
        self.thisptr = new pmft.PMFTXYZ(x_max, y_max, z_max, n_x, n_y, n_z, c_shiftvec)
        self.shiftvec = np.array(shiftvec, dtype=np.float32)
        self.rmax = np.sqrt(x_max**2 + y_max**2 + z_max**2)

    def __dealloc__(self):
        del self.thisptr

    @property
    def box(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box()`
        """
        return self.getBox()

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations, face_orientations=None, nlist=None):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param face_orientations: Optional - orientations of particle faces to account for particle symmetry.
            * If not supplied by user, unit quaternions will be supplied.
            * If a 2D array of shape (:math:`N_f`, :math:`4`) or a 3D array of shape (1, :math:`N_f`, :math:`4`) \
                is supplied, the supplied quaternions will be broadcast for all particles
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 4\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 4\\right)`, dtype= :class:`numpy.float32`
        :type face_orientations: :class:`numpy.ndarray`, shape= :math:`\\left( \\left(N_{particles}, \\right), N_{faces}, 4\\right)`, \
            dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(ref_points, 2, dtype=np.float32, contiguous=True,
            dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(ref_orientations, 2, dtype=np.float32, contiguous=True,
            dim_message="ref_orientations must be a 2 dimensional array")
        if ref_orientations.shape[1] != 4:
            raise ValueError("the 2nd dimension must have 4 values: q0, q1, q2, q3")

        points = freud.common.convert_array(points, 2, dtype=np.float32, contiguous=True,
            dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')
        points = points - self.shiftvec.reshape(1,3)

        orientations = freud.common.convert_array(orientations, 2, dtype=np.float32, contiguous=True,
            dim_message="orientations must be a 2 dimensional array")
        if orientations.shape[1] != 4:
            raise ValueError("the 2nd dimension must have 4 values: q0, q1, q2, q3")

        # handle multiple ways to input
        if face_orientations is None:
            # set to unit quaternion q = [1,0,0,0]
            face_orientations = np.zeros(shape=(ref_points.shape[0], 1, 4), dtype=np.float32)
            face_orientations[:,:,0] = 1.0
        else:
            if (len(face_orientations.shape) < 2) or (len(face_orientations.shape) > 3):
                raise ValueError("points must be a 2 or 3 dimensional array")
            face_orientations = freud.common.convert_array(face_orientations, face_orientations.ndim, dtype=np.float32, contiguous=True,
                dim_message="face_orientations must be a {} dimensional array".format(face_orientations.ndim))
            if face_orientations.ndim == 2:
                if face_orientations.shape[1] != 4:
                    raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
                # need to broadcast into new array
                tmp_face_orientations = np.zeros(shape=(ref_points.shape[0], face_orientations.shape[0], face_orientations.shape[1]), dtype=np.float32)
                tmp_face_orientations[:] = face_orientations
                face_orientations = tmp_face_orientations
            else:
				# Make sure that the first dimensions is actually the number of particles
                if face_orientations.shape[2] != 4:
                    raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
                elif face_orientations.shape[0] not in (1, ref_points.shape[0]):
                    raise ValueError("If provided as a 3D array, the first dimension of the face_orientations array must be either of size 1 or N_particles")
                elif face_orientations.shape[0] == 1:
                    face_orientations = np.repeat(face_orientations, ref_points.shape[0], axis = 0)

        defaulted_nlist = make_default_nlist(box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList *nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef np.ndarray[float, ndim=3] l_face_orientations = face_orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef unsigned int nFaces = <unsigned int> face_orientations.shape[1]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    nlist_ptr,
                                    <vec3[float]*>l_ref_points.data,
                                    <quat[float]*>l_ref_orientations.data,
                                    nRef,
                                    <vec3[float]*>l_points.data,
                                    <quat[float]*>l_orientations.data,
                                    nP,
                                    <quat[float]*>l_face_orientations.data,
                                    nFaces)
        return self

    def compute(self, box, ref_points, ref_orientations, points, orientations, face_orientations, nlist=None):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param face_orientations: orientations of particle faces to account for particle symmetry
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 4\\right)`, dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 3\\right)`, dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`, shape= :math:`\\left(N_{particles}, 4\\right)`, dtype= :class:`numpy.float32`
        :type face_orientations: :class:`numpy.ndarray`, shape= :math:`\\left( \\left(N_{particles}, \\right), N_{faces}, 4\\right)`, \
            dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations, face_orientations, nlist)
        return self

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by \
        :py:meth:`freud.pmft.PMFTXYZ.getPCF()`.
        """
        self.thisptr.reducePCF()

    @property
    def bin_counts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.uint32`
        """
        return self.getBinCounts()

    def getBinCounts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{z}, N_{y}, N_{x}\\right)`, dtype= :class:`numpy.uint32`
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsZ()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result

    @property
    def PCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getBinCounts()

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{z}, N_{y}, N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsZ()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    @property
    def PMFT(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{r}, N_{\\theta1}, N_{\\theta2}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getPMFT()

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{z}, N_{y}, N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        return -np.log(np.copy(self.getPCF()))

    @property
    def X(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getX()

    def getX(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{x}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* x = self.thisptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>x)
        return result + self.shiftvec[0]

    @property
    def Y(self):
        """
        Get the array of y-values for the PCF histogram

        :return: bin centers of y-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getX()

    def getY(self):
        """
        Get the array of y-values for the PCF histogram

        :return: bin centers of y-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* y = self.thisptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>y)
        return result + self.shiftvec[1]

    @property
    def Z(self):
        """
        Get the array of z-values for the PCF histogram

        :return: bin centers of z-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{y}\\right)`, dtype= :class:`numpy.float32`
        """
        return self.getZ()

    def getZ(self):
        """
        Get the array of z-values for the PCF histogram

        :return: bin centers of z-dimension of histogram
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(N_{z}\\right)`, dtype= :class:`numpy.float32`
        """
        cdef float* z = self.thisptr.getZ().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsZ()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>z)
        return result + self.shiftvec[2]

    @property
    def n_bins_X(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: :math:`N_x`
        :rtype: unsigned int
        """
        return self.getNBinsX()

    def getNBinsX(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: :math:`N_x`
        :rtype: unsigned int
        """
        cdef unsigned int x = self.thisptr.getNBinsX()
        return x

    @property
    def n_bins_Y(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: :math:`N_y`
        :rtype: unsigned int
        """
        return self.getNBinsY()

    def getNBinsY(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: :math:`N_y`
        :rtype: unsigned int
        """
        cdef unsigned int y = self.thisptr.getNBinsY()
        return y

    @property
    def n_bins_Z(self):
        """
        Get the number of bins in the z-dimension of histogram

        :return: :math:`N_z`
        :rtype: unsigned int
        """
        return self.getNBinsZ()

    def getNBinsZ(self):
        """
        Get the number of bins in the z-dimension of histogram

        :return: :math:`N_z`
        :rtype: unsigned int
        """
        cdef unsigned int z = self.thisptr.getNBinsZ()
        return z

    @property
    def jacobian(self):
        """
        Get the jacobian used in the pmft

        :return: Inverse Jacobian
        :rtype: float
        """
        return self.getJacobian()

    def getJacobian(self):
        """
        Get the jacobian

        :return: jacobian
        :rtype: float
        """
        cdef float j = self.thisptr.getJacobian()
        return j
