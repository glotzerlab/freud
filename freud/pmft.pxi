# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import numpy as np
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libc.string cimport memcpy
cimport freud._box as _box
cimport freud._pmft as pmft
cimport numpy as np

cdef class _PMFT:
    """Compute the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ for a given set of points.

    This class provides an abstract interface for computing the PMFT.
    It must be specialized for a specific coordinate system; although in principle the PMFT is coordinate independent, the binning process must be performed in a particular coordinate system.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>
    """
    cdef pmft.PMFT * pmftptr
    cdef float rmax

    def __cinit__(self):
        pass

    def __dealloc__(self):
        if type(self) is _PMFT:
            del self.pmftptr

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
            :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(self.pmftptr.getBox())

    def resetPCF(self):
        """Resets the values of the PCF histograms in memory."""
        self.pmftptr.reset()

    def reducePCF(self):
        """Reduces the histogram in the values over N processors to a single
        histogram. This is called automatically by :py:meth:`freud.pmft.PMFT.PCF`.
        """
        self.pmftptr.reducePCF()

    @property
    def bin_counts(self):
        return self.getBinCounts()

    @property
    def PCF(self):
        return self.getPCF()

    @property
    def PMFT(self):
        return self.getPMFT()

    def getPMFT(self):
        """Get the potential of mean force and torque.

        Returns:
            (matches PCF) :class:`numpy.ndarray`: PMFT.
        """
        return -np.log(np.copy(self.getPCF()))

    @property
    def r_cut(self):
        return self.getRCut()

    def getRCut(self):
        """Get the r_cut value used in the cell list.

        Returns:
          float: r_cut.
        """
        cdef float r_cut = self.pmftptr.getRCut()
        return r_cut


cdef class PMFTR12(_PMFT):
    """Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in a 2D
    system described by :math:`r`, :math:`\\theta_1`, :math:`\\theta_2`.

    .. note::
        2D: :py:class:`freud.pmft.PMFTR12` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        r_max (float): Maximum distance at which to compute the PMFT.
        n_r (unsigned int): Number of bins in r.
        n_t1 (unsigned int): Number of bins in t1.
        n_t2 (unsigned int): Number of bins in t2.

    Attributes:
        box (:py:class:`freud.box.Box`): Box used in the calculation.
        bin_counts (:math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)`): Bin counts.
        PCF (:math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)`): The positional correlation function.
        PMFT (:math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)`): The potential of mean force and torque.
        r_cut (float): The cutoff used in the cell list.
        R (:math:`\\left(N_{r}\\right)` :class:`numpy.ndarray`): The array of r-values for the PCF histogram.
        T1 (:math:`\\left(N_{\\theta1}\\right)` :class:`numpy.ndarray`): The array of T1-values for the PCF histogram.
        T2 (:math:`\\left(N_{\\theta2}\\right)` :class:`numpy.ndarray`): The array of T2-values for the PCF histogram.
        inverse_jacobian (:math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)`): The inverse Jacobian used in the PMFT.
        n_bins_r (unsigned int): The number of bins in the r-dimension of histogram.
        n_bins_T1 (unsigned int): The number of bins in the T1-dimension of histogram.
        n_bins_T2 (unsigned int): The number of bins in the T2-dimension of histogram.
    """
    cdef pmft.PMFTR12 * pmftr12ptr

    def __cinit__(self, r_max, n_r, n_t1, n_t2):
        if type(self) is PMFTR12:
            self.pmftr12ptr = self.pmftptr = new pmft.PMFTR12(r_max, n_r, n_t1, n_t2)
            self.rmax = r_max

    def __dealloc__(self):
        if type(self) is PMFTR12:
            del self.pmftr12ptr

    def accumulate(self, box, ref_points, ref_orientations, points,
                   orientations, nlist=None):
        """Calculates the positional correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Reference points to
                                                        calculate the local
                                                        density.
            ref_orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Angles of reference
                                                             points to use in the
                                                             calculation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Points to calculate the local
                                                    density.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Angles of particles to
                                                         use in the calculation.
            nlist(:class:`freud.locality.NeighborList`, optional): NeighborList to use to find bonds (Default value = None).
        """
        box = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 1, dtype=np.float32, contiguous=True,
                array_name="ref_orientations")

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 1, dtype=np.float32, contiguous=True,
                array_name="orientations")

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim= 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim= 2] l_points = points
        cdef np.ndarray[float, ndim= 1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim= 1] l_orientations = orientations
        cdef unsigned int nRef = <unsigned int > ref_points.shape[0]
        cdef unsigned int nP = <unsigned int > points.shape[0]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.pmftr12ptr.accumulate(l_box,
                                    nlist_ptr,
                                    < vec3[float]*>l_ref_points.data,
                                    < float*>l_ref_orientations.data,
                                    nRef,
                                    < vec3[float]*>l_points.data,
                                    < float*>l_orientations.data,
                                    nP)
        return self

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                nlist=None):
        """Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Reference points to
                                                        calculate the local density.
            ref_orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Reference orientations as angles to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Points to calculate the local density.
            orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Orientations as angles to use in computation.
            nlist (:class:`freud.locality.NeighborList`, optional): NeighborList to use to find bonds (Default value = None).
        """
        self.pmftr12ptr.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    def getBinCounts(self):
        """Get the raw bin counts.

        Returns:
            :math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)` :class:`numpy.ndarray`: Bin Counts.
        """
        cdef unsigned int * bin_counts = self.pmftr12ptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp > self.pmftr12ptr.getNBinsR()
        nbins[1] = <np.npy_intp > self.pmftr12ptr.getNBinsT2()
        nbins[2] = <np.npy_intp > self.pmftr12ptr.getNBinsT1()
        cdef np.ndarray[np.uint32_t, ndim = 3
                        ] result = np.PyArray_SimpleNewFromData(
                                3, nbins, np.NPY_UINT32, < void*>bin_counts)
        return result

    def getPCF(self):
        """Get the positional correlation function.

        Returns:
            :math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)` :class:`numpy.ndarray`: PCF.
        """
        cdef float * pcf = self.pmftr12ptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp > self.pmftr12ptr.getNBinsR()
        nbins[1] = <np.npy_intp > self.pmftr12ptr.getNBinsT2()
        nbins[2] = <np.npy_intp > self.pmftr12ptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim = 3
                        ] result = np.PyArray_SimpleNewFromData(
                                3, nbins, np.NPY_FLOAT32, < void*>pcf)
        return result

    @property
    def R(self):
        return self.getR()

    def getR(self):
        """Get the array of r-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{r}\\right)` :class:`numpy.ndarray`: Bin centers of r-dimension of histogram.
        """
        cdef float * r = self.pmftr12ptr.getR().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftr12ptr.getNBinsR()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>r)
        return result

    @property
    def T1(self):
        return self.getT1()

    def getT1(self):
        """Get the array of T1-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{\\theta_1}\\right)` :class:`numpy.ndarray`: Bin centers of T1-dimension of histogram.
        """
        cdef float * T1 = self.pmftr12ptr.getT1().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftr12ptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>T1)
        return result

    @property
    def T2(self):
        return self.getT2()

    def getT2(self):
        """Get the array of T2-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{\\theta_2}\\right)` :class:`numpy.ndarray`: Bin centers of T2-dimension of histogram.
        """
        cdef float * T2 = self.pmftr12ptr.getT2().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftr12ptr.getNBinsT2()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>T2)
        return result

    @property
    def inverse_jacobian(self):
        return self.getInverseJacobian()

    def getInverseJacobian(self):
        """Get the inverse Jacobian used in the PMFT.

        Returns:
            :math:`\\left(N_{r}, N_{\\theta2}, N_{\\theta1}\\right)` :class:`numpy.ndarray`: Inverse Jacobian.
        """
        cdef float * inv_jac = self.pmftr12ptr.getInverseJacobian().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp > self.pmftr12ptr.getNBinsR()
        nbins[1] = <np.npy_intp > self.pmftr12ptr.getNBinsT2()
        nbins[2] = <np.npy_intp > self.pmftr12ptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim = 3
                        ] result = np.PyArray_SimpleNewFromData(
                                3, nbins, np.NPY_FLOAT32, < void*>inv_jac)
        return result

    @property
    def n_bins_r(self):
        return self.getNBinsR()

    def getNBinsR(self):
        """Get the number of bins in the r-dimension of histogram.

        Returns:
            unsigned int: :math:`N_r`.
        """
        cdef unsigned int r = self.pmftr12ptr.getNBinsR()
        return r

    @property
    def n_bins_T1(self):
        return self.getNBinsT1()

    def getNBinsT1(self):
        """Get the number of bins in the T1-dimension of histogram.

        Returns:
            unsigned int: :math:`N_{\\theta_1}`.
        """
        cdef unsigned int T1 = self.pmftr12ptr.getNBinsT1()
        return T1

    @property
    def n_bins_T2(self):
        return self.getNBinsT2()

    def getNBinsT2(self):
        """Get the number of bins in the T2-dimension of histogram.

        Returns:
            unsigned int: :math:`N_{\\theta_2}`.
        """
        cdef unsigned int T2 = self.pmftr12ptr.getNBinsT2()
        return T2

cdef class PMFTXYT(_PMFT):
    """Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ for
    systems described by coordinates :math:`x`, :math:`y`, :math:`\\theta`
    listed in the x, y, and t arrays.

    The values of x, y, t to compute the PCF at are controlled by x_max, y_max and n_bins_x, n_bins_y, n_bins_t parameters to the constructor.
    The x_max and y_max parameters determine the minimum/maximum x, y values (:math:`\\min \\left(\\theta \\right) = 0`, (:math:`\\max \\left( \\theta \\right) = 2\\pi`) at which to compute the PCF and n_bins_x, n_bins_y, n_bins_t is the number of bins in x, y, t.

    .. note::
        2D: :py:class:`freud.pmft.PMFTXYT` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float): Maximum x distance at which to compute the PMFT.
        y_max (float): Maximum y distance at which to compute the PMFT.
        n_x (unsigned int): Number of bins in x.
        n_y (unsigned int): Number of bins in y.
        n_t (unsigned int): Number of bins in t.

    Attributes:
        box (:py:class:`freud.box.Box`): Box used in the calculation.
        bin_counts (:math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): Bin counts.
        PCF (:math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): The positional correlation function.
        PMFT (:math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): The potential of mean force and torque.
        r_cut (float): The cutoff used in the cell list.
        X (:math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`): The array of x-values for the PCF histogram.
        Y (:math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`): The array of y-values for the PCF histogram.
        T (:math:`\\left(N_{\\theta}\\right)` :class:`numpy.ndarray`): The array of T-values for the PCF histogram.
        jacobian (float): The Jacobian used in the PMFT.
        n_bins_x (unsigned int): The number of bins in the x-dimension of histogram.
        n_bins_y (unsigned int): The number of bins in the y-dimension of histogram.
        n_bins_T (unsigned int): The number of bins in the T-dimension of histogram.
    """
    cdef pmft.PMFTXYT * pmftxytptr

    def __cinit__(self, x_max, y_max, n_x, n_y, n_t):
        if type(self) is PMFTXYT:
            self.pmftxytptr = self.pmftptr = new pmft.PMFTXYT(x_max, y_max, n_x, n_y, n_t)
            self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXYT:
            del self.pmftxytptr

    def accumulate(self, box, ref_points, ref_orientations, points,
                   orientations, nlist=None):
        """Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Reference points to
                                                        calculate the local density.
            ref_orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Reference orientations as angles to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Points to calculate the local density.
            orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): orientations as angles to use in computation.
            nlist (:class:`freud.locality.NeighborList`, optional): NeighborList to use to find bonds (Default value = None).
        """
        box = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 1, dtype=np.float32, contiguous=True,
                array_name="ref_orientations")

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 1, dtype=np.float32, contiguous=True,
                array_name="orientations")

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim= 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim= 2] l_points = points
        cdef np.ndarray[float, ndim= 1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim= 1] l_orientations = orientations
        cdef unsigned int nRef = <unsigned int > ref_points.shape[0]
        cdef unsigned int nP = <unsigned int > points.shape[0]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.pmftxytptr.accumulate(l_box,
                                    nlist_ptr,
                                    < vec3[float]*>l_ref_points.data,
                                    < float*>l_ref_orientations.data,
                                    nRef,
                                    < vec3[float]*>l_points.data,
                                    < float*>l_orientations.data,
                                    nP)
        return self

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                nlist=None):
        """Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Reference points to
                                                        calculate the local density.
            ref_orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Reference orientations as angles to use in computation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Points to calculate the local density.
            orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): orientations as angles to use in computation.
            nlist (:class:`freud.locality.NeighborList`, optional): NeighborList to use to find bonds (Default value = None).
        """
        self.pmftxytptr.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    def getBinCounts(self):
        """Get the raw bin counts.

        Returns:
            :math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`: Bin Counts.
        """
        cdef unsigned int * bin_counts = self.pmftxytptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp > self.pmftxytptr.getNBinsT()
        nbins[1] = <np.npy_intp > self.pmftxytptr.getNBinsY()
        nbins[2] = <np.npy_intp > self.pmftxytptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim = 3
                        ] result = np.PyArray_SimpleNewFromData(
                                3, nbins, np.NPY_UINT32, < void*>bin_counts)
        return result

    def getPCF(self):
        """Get the positional correlation function.

        Returns:
            :math:`\\left(N_{\\theta}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`: PCF.
        """
        cdef float * pcf = self.pmftxytptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp > self.pmftxytptr.getNBinsT()
        nbins[1] = <np.npy_intp > self.pmftxytptr.getNBinsY()
        nbins[2] = <np.npy_intp > self.pmftxytptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim = 3
                        ] result = np.PyArray_SimpleNewFromData(
                                3, nbins, np.NPY_FLOAT32, < void*>pcf)
        return result

    @property
    def X(self):
        return self.getX()

    def getX(self):
        """Get the array of x-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`: Bin centers of x-dimension of histogram.
        """
        cdef float * x = self.pmftxytptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftxytptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>x)
        return result

    @property
    def Y(self):
        return self.getY()

    def getY(self):
        """Get the array of y-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`: Bin centers of y-dimension of histogram.
        """
        cdef float * y = self.pmftxytptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftxytptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>y)
        return result

    @property
    def T(self):
        return self.getT()

    def getT(self):
        """Get the array of t-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{\\theta}\\right)` :class:`numpy.ndarray`: Bin centers of t-dimension of histogram.
        """
        cdef float * t = self.pmftxytptr.getT().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftxytptr.getNBinsT()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>t)
        return result

    @property
    def jacobian(self):
        return self.getJacobian()

    def getJacobian(self):
        """Get the Jacobian used in the PMFT.

        Returns:
          float: Jacobian.
        """
        cdef float j = self.pmftxytptr.getJacobian()
        return j

    @property
    def n_bins_X(self):
        return self.getNBinsX()

    def getNBinsX(self):
        """Get the number of bins in the x-dimension of histogram.

        Returns:
          unsigned int: :math:`N_x`.
        """
        cdef unsigned int x = self.pmftxytptr.getNBinsX()
        return x

    @property
    def n_bins_Y(self):
        return self.getNBinsY()

    def getNBinsY(self):
        """Get the number of bins in the y-dimension of histogram.

        Returns:
          unsigned int: :math:`N_y`.
        """
        cdef unsigned int y = self.pmftxytptr.getNBinsY()
        return y

    @property
    def n_bins_T(self):
        return self.getNBinsT()

    def getNBinsT(self):
        """Get the number of bins in the t-dimension of histogram.

        Returns:
          unsigned int: :math:`N_{\\theta}`.
        """
        cdef unsigned int t = self.pmftxytptr.getNBinsT()
        return t


cdef class PMFTXY2D(_PMFT):
    """Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in
    coordinates :math:`x`, :math:`y` listed in the x and y arrays.

    The values of x and y to compute the PCF at are controlled by x_max, y_max, n_x, and n_y parameters to the constructor.
    The x_max and y_max parameters determine the minimum/maximum distance at which to compute the PCF and n_x and n_y are the number of bins in x and y.

    .. note::
        2D: :py:class:`freud.pmft.PMFTXY2D` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float): Maximum x distance at which to compute the PMFT.
        y_max (float): Maximum y distance at which to compute the PMFT.
        n_x (unsigned int): Number of bins in x.
        n_y (unsigned int): Number of bins in y.

    Attributes:
        box (:py:class:`freud.box.Box`): Box used in the calculation.
        bin_counts (:math:`\\left(N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): Bin counts.
        PCF (:math:`\\left(N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): The positional correlation function.
        PMFT (:math:`\\left(N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): The potential of mean force and torque.
        r_cut (float): The cutoff used in the cell list.
        X (:math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`): The array of x-values for the PCF histogram.
        Y (:math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`): The array of y-values for the PCF histogram.
        jacobian (float): The Jacobian used in the PMFT.
        n_bins_x (unsigned int): The number of bins in the x-dimension of histogram.
        n_bins_y (unsigned int): The number of bins in the y-dimension of histogram.
    """
    cdef pmft.PMFTXY2D * pmftxy2dptr

    def __cinit__(self, x_max, y_max, n_x, n_y):
        if type(self) is PMFTXY2D:
            self.pmftxy2dptr = self.pmftptr = new pmft.PMFTXY2D(x_max, y_max, n_x, n_y)
            self.rmax = np.sqrt(x_max**2 + y_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXY2D:
            del self.pmftxy2dptr

    def accumulate(self, box, ref_points, ref_orientations, points,
                   orientations, nlist=None):
        """Calculates the positional correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Reference points to
                                                        calculate the local density.
            ref_orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Angles of reference
                                                             points to use in the calculation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Points to calculate the local
                                                    density.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Angles of particles to
                                                         use in the calculation.
            nlist (:class:`freud.locality.NeighborList`, optional): NeighborList to use to find bonds (Default value = None).
        """
        box = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 1, dtype=np.float32, contiguous=True,
                array_name="ref_orientations")

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 1, dtype=np.float32, contiguous=True,
                array_name="orientations")

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim= 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim= 2] l_points = points
        cdef np.ndarray[float, ndim= 1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim= 1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int > ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int > points.shape[0]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.pmftxy2dptr.accumulate(l_box,
                                    nlist_ptr,
                                    < vec3[float]*>l_ref_points.data,
                                    < float*>l_ref_orientations.data,
                                    n_ref,
                                    < vec3[float]*>l_points.data,
                                    < float*>l_orientations.data,
                                    n_p)
        return self

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                nlist=None):
        """Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Reference points to
                                                        calculate the local density.
            ref_orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Angles of reference
                                                             points to use in the calculation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Points to calculate the local
                                                    density.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Angles of particles to
                                                         use in the calculation.
            nlist (:class:`freud.locality.NeighborList`, optional): NeighborList to use to find bonds (Default value = None).
        """
        self.pmftxy2dptr.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, nlist)
        return self

    def getPCF(self):
        """Get the positional correlation function.

        Returns:
            :math:`\\left(N_{y}, N_{x}\\right)` :class:`numpy.ndarray`: PCF.
        """
        cdef float * pcf = self.pmftxy2dptr.getPCF().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.pmftxy2dptr.getNBinsY()
        nbins[1] = <np.npy_intp > self.pmftxy2dptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim = 2
                        ] result = np.PyArray_SimpleNewFromData(
                                2, nbins, np.NPY_FLOAT32, < void*>pcf)
        return result

    def getBinCounts(self):
        """Get the raw bin counts (non-normalized).

        Returns:
            :math:`\\left(N_{y}, N_{x}\\right)` :class:`numpy.ndarray`: Bin Counts.
        """
        cdef unsigned int * bin_counts = self.pmftxy2dptr.getBinCounts().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.pmftxy2dptr.getNBinsY()
        nbins[1] = <np.npy_intp > self.pmftxy2dptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim = 2
                        ] result = np.PyArray_SimpleNewFromData(
                                2, nbins, np.NPY_UINT32, < void*>bin_counts)
        return result

    @property
    def X(self):
        return self.getX()

    def getX(self):
        """Get the array of x-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`: Bin centers of x-dimension of histogram.
        """
        cdef float * x = self.pmftxy2dptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftxy2dptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>x)
        return result

    @property
    def Y(self):
        return self.getY()

    def getY(self):
        """Get the array of y-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`: Bin centers of y-dimension of histogram.
        """
        cdef float * y = self.pmftxy2dptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftxy2dptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>y)
        return result

    @property
    def n_bins_X(self):
        return self.getNBinsX()

    def getNBinsX(self):
        """Get the number of bins in the x-dimension of histogram.

        Returns:
            unsigned int: :math:`N_x`.
        """
        cdef unsigned int x = self.pmftxy2dptr.getNBinsX()
        return x

    @property
    def n_bins_Y(self):
        return self.getNBinsY()

    def getNBinsY(self):
        """Get the number of bins in the y-dimension of histogram.

        Returns:
            unsigned int: :math:`N_y`.
        """
        cdef unsigned int y = self.pmftxy2dptr.getNBinsY()
        return y

    @property
    def jacobian(self):
        return self.getJacobian()

    def getJacobian(self):
        """Get the Jacobian.

        Returns:
            float: Jacobian.
        """
        cdef float j = self.pmftxy2dptr.getJacobian()
        return j


cdef class PMFTXYZ(_PMFT):
    """Computes the PMFT [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in
    coordinates :math:`x`, :math:`y`, :math:`z`, listed in the x, y, and z
    arrays.

    The values of x, y, z to compute the PCF at are controlled by x_max, y_max, z_max, n_x, n_y, and n_z parameters to the constructor.
    The x_max, y_max, and z_max parameters determine the minimum/maximum distance at which to compute the PCF and n_x, n_y, and n_z are the number of bins in x, y, z.

    .. note::
        3D: :py:class:`freud.pmft.PMFTXYZ` is only defined for 3D systems.
        The points must be passed in as :code:`[x, y, z]`.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        x_max (float): Maximum x distance at which to compute the PMFT.
        y_max (float): Maximum y distance at which to compute the PMFT.
        z_max (float): Maximum z distance at which to compute the PMFT.
        n_x (unsigned int): Number of bins in x.
        n_y (unsigned int): Number of bins in y.
        n_z (unsigned int): Number of bins in z.
        shiftvec (list): Vector pointing from [0,0,0] to the center of the PMFT.

    Attributes:
        box (:py:class:`freud.box.Box`): Box used in the calculation.
        bin_counts (:math:`\\left(N_{z}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): Bin counts.
        PCF (:math:`\\left(N_{z}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): The positional correlation function.
        PMFT (:math:`\\left(N_{z}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`): The potential of mean force and torque.
        r_cut (float): The cutoff used in the cell list.
        X (:math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`): The array of x-values for the PCF histogram.
        Y (:math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`): The array of y-values for the PCF histogram.
        Z (:math:`\\left(N_{z}\\right)` :class:`numpy.ndarray`): The array of z-values for the PCF histogram.
        jacobian (float): The Jacobian used in the PMFT.
        n_bins_x (unsigned int): The number of bins in the x-dimension of histogram.
        n_bins_y (unsigned int): The number of bins in the y-dimension of histogram.
        n_bins_z (unsigned int): The number of bins in the z-dimension of histogram.
    """
    cdef pmft.PMFTXYZ * pmftxyzptr
    cdef shiftvec

    def __cinit__(self, x_max, y_max, z_max, n_x, n_y, n_z,
                  shiftvec=[0, 0, 0]):
        cdef vec3[float] c_shiftvec
        if type(self) is PMFTXYZ:
            c_shiftvec = vec3[float](
                    shiftvec[0], shiftvec[1], shiftvec[2])
            self.pmftxyzptr = self.pmftptr = new pmft.PMFTXYZ(
                    x_max, y_max, z_max, n_x, n_y, n_z, c_shiftvec)
            self.shiftvec = np.array(shiftvec, dtype=np.float32)
            self.rmax = np.sqrt(x_max**2 + y_max**2 + z_max**2)

    def __dealloc__(self):
        if type(self) is PMFTXYZ:
            del self.pmftxyzptr

    def resetPCF(self):
        """Resets the values of the PCF histograms in memory."""
        self.pmftxyzptr.reset()

    def accumulate(self, box, ref_points, ref_orientations, points,
                   orientations, face_orientations=None, nlist=None):
        """Calculates the positional correlation function and adds to the
        current histogram.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Reference points to
                                                        calculate the local density.
            ref_orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Angles of reference
                                                             points to use in the calculation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): Points to calculate the local
                                                    density.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): Angles of particles to
                                                         use in the calculation.
            face_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`, optional):
                               Orientations of particle faces to account for
                               particle symmetry. If not supplied by user, unit
                               quaternions will be supplied. If a 2D array of
                               shape (:math:`N_f`, :math:`4`) or a 3D array of
                               shape (1, :math:`N_f`, :math:`4`) is supplied,
                               the supplied quaternions will be broadcast for
                               all particles.  (Default value = None).
            nlist (:class:`freud.locality.NeighborList`, optional): NeighborList to use to find bonds (Default value = None).
        """
        box = freud.common.convert_box(box)
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
                "the 2nd dimension must have 4 values: q0, q1, q2, q3")

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')
        points = points - self.shiftvec.reshape(1, 3)

        orientations = freud.common.convert_array(
                orientations, 2, dtype=np.float32, contiguous=True,
                array_name="orientations")
        if orientations.shape[1] != 4:
            raise ValueError(
                "the 2nd dimension must have 4 values: q0, q1, q2, q3")

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
                    array_name=("face_orientations must be a {}"
                                 "dimensional array").format(
                                        face_orientations.ndim))
            if face_orientations.ndim == 2:
                if face_orientations.shape[1] != 4:
                    raise ValueError(
                        ("2nd dimension for orientations must have 4 values:"
                            "s, x, y, z"))
                # need to broadcast into new array
                tmp_face_orientations = np.zeros(
                        shape=(
                            ref_points.shape[0],
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
                        ("2nd dimension for orientations must have 4 values:"
                            "s, x, y, z"))
                elif face_orientations.shape[0] not in (
                        1, ref_points.shape[0]):
                    raise ValueError(
                        ("If provided as a 3D array, the first dimension of"
                            " the face_orientations array must be either of"
                            " size 1 or N_particles"))
                elif face_orientations.shape[0] == 1:
                    face_orientations = np.repeat(
                        face_orientations, ref_points.shape[0], axis=0)

        defaulted_nlist = make_default_nlist(
            box, ref_points, points, self.rmax, nlist, None)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim= 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim= 2] l_points = points
        cdef np.ndarray[float, ndim= 2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim= 2] l_orientations = orientations
        cdef np.ndarray[float, ndim= 3] l_face_orientations = face_orientations
        cdef unsigned int nRef = <unsigned int > ref_points.shape[0]
        cdef unsigned int nP = <unsigned int > points.shape[0]
        cdef unsigned int nFaces = <unsigned int > face_orientations.shape[1]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.pmftxyzptr.accumulate(l_box,
                                    nlist_ptr,
                                    < vec3[float]*>l_ref_points.data,
                                    < quat[float]*>l_ref_orientations.data,
                                    nRef,
                                    < vec3[float]*>l_points.data,
                                    < quat[float]*>l_orientations.data,
                                    nP,
                                    < quat[float]*>l_face_orientations.data,
                                    nFaces)
        return self

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                face_orientations=None, nlist=None):
        """Calculates the positional correlation function for the given points.
        Will overwrite the current histogram.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): reference points to
                                                        calculate the local density.
            ref_orientations((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): angles of reference
                                                             points to use in the calculation.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`): points to calculate the local
                                                    density.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`): angles of particles to
                                                         use in the calculation.
            face_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`, optional):
                               Orientations of particle faces to account for
                               particle symmetry. If not supplied by user, unit
                               quaternions will be supplied. If a 2D array of
                               shape (:math:`N_f`, :math:`4`) or a 3D array of
                               shape (1, :math:`N_f`, :math:`4`) is supplied,
                               the supplied quaternions will be broadcast for
                               all particles.  (Default value = None).
            nlist(:class:`freud.locality.NeighborList`, optional): NeighborList to use to find bonds (Default value = None).
        """
        self.pmftxyzptr.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, face_orientations, nlist)
        return self

    def reducePCF(self):
        """Reduces the histogram in the values over N processors to a single
        histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTXYZ.PCF`.
        """
        self.pmftxyzptr.reducePCF()

    def getBinCounts(self):
        """Get the raw bin counts.

        Returns:
            :math:`\\left(N_{z}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`: Bin Counts.
        """
        cdef unsigned int * bin_counts = self.pmftxyzptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp > self.pmftxyzptr.getNBinsZ()
        nbins[1] = <np.npy_intp > self.pmftxyzptr.getNBinsY()
        nbins[2] = <np.npy_intp > self.pmftxyzptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim = 3
                        ] result = np.PyArray_SimpleNewFromData(
                                3, nbins, np.NPY_UINT32, < void*>bin_counts)
        return result

    def getPCF(self):
        """Get the positional correlation function.

        Returns:
            :math:`\\left(N_{z}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`: PCF.
        """
        cdef float * pcf = self.pmftxyzptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp > self.pmftxyzptr.getNBinsZ()
        nbins[1] = <np.npy_intp > self.pmftxyzptr.getNBinsY()
        nbins[2] = <np.npy_intp > self.pmftxyzptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim = 3
                        ] result = np.PyArray_SimpleNewFromData(
                                3, nbins, np.NPY_FLOAT32, < void*>pcf)
        return result

    def getPMFT(self):
        """Get the potential of mean force and torque.

        Returns:
            :math:`\\left(N_{z}, N_{y}, N_{x}\\right)` :class:`numpy.ndarray`: PMFT.
        """
        return -np.log(np.copy(self.getPCF()))

    @property
    def X(self):
        return self.getX()

    def getX(self):
        """Get the array of x-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{x}\\right)` :class:`numpy.ndarray`: Bin centers of x-dimension of histogram.
        """
        cdef float * x = self.pmftxyzptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftxyzptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>x)
        return result + self.shiftvec[0]

    @property
    def Y(self):
        return self.getY()

    def getY(self):
        """Get the array of y-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{y}\\right)` :class:`numpy.ndarray`: Bin centers of y-dimension of histogram.
        """
        cdef float * y = self.pmftxyzptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftxyzptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>y)
        return result + self.shiftvec[1]

    @property
    def Z(self):
        return self.getZ()

    def getZ(self):
        """Get the array of z-values for the PCF histogram.

        Returns:
            :math:`\\left(N_{z}\\right)` :class:`numpy.ndarray`: Bin centers of z-dimension of histogram.
        """
        cdef float * z = self.pmftxyzptr.getZ().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.pmftxyzptr.getNBinsZ()
        cdef np.ndarray[np.float32_t, ndim = 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>z)
        return result + self.shiftvec[2]

    @property
    def n_bins_X(self):
        return self.getNBinsX()

    def getNBinsX(self):
        """Get the number of bins in the x-dimension of histogram.

        Returns:
           unsigned int: :math:`N_x`.
        """
        cdef unsigned int x = self.pmftxyzptr.getNBinsX()
        return x

    @property
    def n_bins_Y(self):
        return self.getNBinsY()

    def getNBinsY(self):
        """Get the number of bins in the y-dimension of histogram.

        Returns:
            unsigned int: :math:`N_y`.
        """
        cdef unsigned int y = self.pmftxyzptr.getNBinsY()
        return y

    @property
    def n_bins_Z(self):
        return self.getNBinsZ()

    def getNBinsZ(self):
        """Get the number of bins in the z-dimension of histogram.

        Returns:
            unsigned int: :math:`N_z`.
        """
        cdef unsigned int z = self.pmftxyzptr.getNBinsZ()
        return z

    @property
    def jacobian(self):
        return self.getJacobian()

    def getJacobian(self):
        """Get the Jacobian.

        Returns:
            float: Jacobian.
        """
        cdef float j = self.pmftxyzptr.getJacobian()
        return j
