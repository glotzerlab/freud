
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

    def __cinit__(self, r_max, n_r, n_t1, n_t2):
        self.thisptr = new pmft.PMFTR12(r_max, n_r, n_t1, n_t2)

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if (ref_points.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (ref_orientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if len(ref_points.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(ref_orientations.shape) != 1 or len(orientations.shape) != 1:
            raise ValueError("orientations must be a 2 dimensional array")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    <vec3[float]*>l_ref_points.data,
                                    <float*>l_ref_orientations.data,
                                    nRef,
                                    <vec3[float]*>l_points.data,
                                    <float*>l_orientations.data,
                                    nP)

    def compute(self, box, ref_points, ref_orientations, points, orientations):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTR12.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getBinCounts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: np.ndarray(shape=(R, T2, T1), dtype=np.float32)
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsT2()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[np.uint32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: np.ndarray(shape=(R, T2, T1), dtype=np.float32)
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsT2()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: np.ndarray(shape=(R, T2, T1), dtype=np.float32)
        """
        return -np.log(np.copy(self.getPCF()))

    def getR(self):
        """
        Get the array of r-values for the PCF histogram

        :return: bin centers of r-dimension of histogram
        :rtype: np.ndarray(shape=nr, dtype=np.float32)
        """
        cdef float* r = self.thisptr.getR().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>r)
        return result

    def getT1(self):
        """
        Get the array of T1-values for the PCF histogram

        :return: bin centers of T1-dimension of histogram
        :rtype: np.ndarray(shape=nT1, dtype=np.float32)
        """
        cdef float* T1 = self.thisptr.getT1().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>T1)
        return result

    def getT2(self):
        """
        Get the array of T2-values for the PCF histogram

        :return: bin centers of T2-dimension of histogram
        :rtype: np.ndarray(shape=nT2, dtype=np.float32)
        """
        cdef float* T2 = self.thisptr.getT2().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT2()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>T2)
        return result

    def getInverseJacobian(self):
        """
        Get the inverse jacobian used in the pmft

        :return: Inverse Jacobian
        :rtype: np.ndarray(shape=(R, T2, T1), dtype=np.float32)
        """
        cdef float* inv_jac = self.thisptr.getInverseJacobian().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsT2()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>inv_jac)
        return result

    def getNBinsR(self):
        """
        Get the number of bins in the r-dimension of histogram

        :return: nr
        :rtype: unsigned int
        """
        cdef unsigned int r = self.thisptr.getNBinsR()
        return r

    def getNBinsT1(self):
        """
        Get the number of bins in the T1-dimension of histogram

        :return: nT1
        :rtype: unsigned int
        """
        cdef unsigned int T1 = self.thisptr.getNBinsT1()
        return T1

    def getNBinsT2(self):
        """
        Get the number of bins in the T2-dimension of histogram

        :return: nT2
        :rtype: unsigned int
        """
        cdef unsigned int T2 = self.thisptr.getNBinsT2()
        return T2

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

    def __cinit__(self, x_max, y_max, n_x, n_y, n_t):
        self.thisptr = new pmft.PMFTXYT(x_max, y_max, n_x, n_y, n_t)

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if (ref_points.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (ref_orientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if len(ref_points.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(ref_orientations.shape) != 1 or len(orientations.shape) != 1:
            raise ValueError("orientations must be a 2 dimensional array")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    <vec3[float]*>l_ref_points.data,
                                    <float*>l_ref_orientations.data,
                                    nRef,
                                    <vec3[float]*>l_points.data,
                                    <float*>l_orientations.data,
                                    nP)

    def compute(self, box, ref_points, ref_orientations, points, orientations):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTXYT.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getBinCounts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: np.ndarray(shape=(:math:`n_{\\theta}`, :math:`n_y`, :math:`n_x`), dtype=np.float32)
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: np.ndarray(shape=(:math:`n_{\\theta}`, :math:`n_y`, :math:`n_x`), dtype=np.float32)
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: np.ndarray(shape=(:math:`n_{\\theta}`, :math:`n_y`, :math:`n_x`), dtype=np.float32)
        """
        return -np.log(np.copy(self.getPCF()))

    def getX(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: np.ndarray(shape=:math:`n_x`, dtype=np.float32)
        """
        cdef float* x = self.thisptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>x)
        return result

    def getY(self):
        """
        Get the array of y-values for the PCF histogram

        :return: bin centers of y-dimension of histogram
        :rtype: np.ndarray(shape=:math:`n_y`, dtype=np.float32)
        """
        cdef float* y = self.thisptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>y)
        return result

    def getT(self):
        """
        Get the array of t-values for the PCF histogram

        :return: bin centers of t-dimension of histogram
        :rtype: np.ndarray(shape=:math:`n_t`, dtype=np.float32)
        """
        cdef float* t = self.thisptr.getT().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsT()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>t)
        return result

    def getJacobian(self):
        """
        Get the jacobian used in the pmft

        :return: Inverse Jacobian
        :rtype: float
        """
        cdef float j = self.thisptr.getJacobian()
        return j

    def getNBinsX(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: :math:`n_x`
        :rtype: unsigned int
        """
        cdef unsigned int x = self.thisptr.getNBinsX()
        return x

    def getNBinsY(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: :math:`n_y`
        :rtype: unsigned int
        """
        cdef unsigned int y = self.thisptr.getNBinsY()
        return y

    def getNBinsT(self):
        """
        Get the number of bins in the t-dimension of histogram

        :return: :math:`n_{\\theta}`
        :rtype: unsigned int
        """
        cdef unsigned int t = self.thisptr.getNBinsT()
        return t

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

    def __cinit__(self, x_max, y_max, n_x, n_y):
        self.thisptr = new pmft.PMFTXY2D(x_max, y_max, n_x, n_y)

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if (ref_points.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (ref_orientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if len(ref_points.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(ref_orientations.shape) != 1 or len(orientations.shape) != 1:
            raise ValueError("orientations must be a 1 dimensional array")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=1] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    <vec3[float]*>l_ref_points.data,
                                    <float*>l_ref_orientations.data,
                                    n_ref,
                                    <vec3[float]*>l_points.data,
                                    <float*>l_orientations.data,
                                    n_p)

    def compute(self, box, ref_points, ref_orientations, points, orientations):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTXY2D.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: np.ndarray(shape=(Ny, Nx), dtype=np.float32)
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: np.ndarray(shape=(Ny, Nx), dtype=np.float32)
        """
        return -np.log(np.copy(self.getPCF()))

    def getBinCounts(self):
        """
        Get the raw bin counts (non-normalized).

        :return: Bin Counts
        :rtype: np.ndarray(shape=(Ny, Nx), dtype=np.uint32)
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result

    def getX(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: np.ndarray(shape=nx, dtype=np.float32)
        """
        cdef float* x = self.thisptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>x)
        return result

    def getY(self):
        """
        Get the array of y-values for the PCF histogram


        :return: bin centers of y-dimension of histogram
        :rtype: np.ndarray(shape=ny, dtype=np.float32)
        """
        cdef float* y = self.thisptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>y)
        return result

    def getNBinsX(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: nx
        :rtype: unsigned int
        """
        cdef unsigned int x = self.thisptr.getNBinsX()
        return x

    def getNBinsY(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: ny
        :rtype: unsigned int
        """
        cdef unsigned int y = self.thisptr.getNBinsY()
        return y

    def getJacobian(self):
        """
        Get the jacobian

        :return: jacobian
        :rtype: float
        """
        cdef float j = self.thisptr.getJacobian()
        return j

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
    :type x_max: float
    :type y_max: float
    :type z_max: float
    :type n_x: unsigned int
    :type n_y: unsigned int
    :type n_z: unsigned int
    """
    cdef pmft.PMFTXYZ *thisptr

    def __cinit__(self, x_max, y_max, z_max, n_x, n_y, n_z):
        self.thisptr = new pmft.PMFTXYZ(x_max, y_max, z_max, n_x, n_y, n_z)

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations, face_orientations):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param face_orientations: orientations of particle faces to account for particle symmetry
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type face_orientations: np.ndarray(shape=((N), Nf, 4), dtype=np.float32)
        """
        if (ref_points.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (ref_orientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if (face_orientations.dtype != np.float32):
            raise ValueError("face_orientations must be a numpy float32 array")
        if len(ref_points.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(ref_orientations.shape) != 2 or len(orientations.shape) != 2:
            raise ValueError("orientations must be a 2 dimensional array")
        # handle multiple ways to input
        if (len(face_orientations.shape) < 2) or (len(face_orientations.shape) > 3):
            raise ValueError("points must be a 2 or 3 dimensional array")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        if ref_orientations.shape[1] != 4 or orientations.shape[1] != 4:
            raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
        if len(face_orientations) == 2:
            if face_orientations.shape[1] != 4:
                raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
            # need to broadcast into new array
            tmp_face_orientations = np.zeros(shape=(ref_points.shape[0], face_orientations.shape[0], face_orientations.shape[1]), dtype=np.float32)
            tmp_face_orientations[:] = face_orientations
            face_orientations = tmp_face_orientations
        elif face_orientations.shape[2] == 3:
            raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
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
                                    <vec3[float]*>l_ref_points.data,
                                    <quat[float]*>l_ref_orientations.data,
                                    nRef,
                                    <vec3[float]*>l_points.data,
                                    <quat[float]*>l_orientations.data,
                                    nP,
                                    <quat[float]*>l_face_orientations.data,
                                    nFaces)

    def compute(self, box, ref_points, ref_orientations, points, orientations, face_orientations):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param face_orientations: orientations of particle faces to account for particle symmetry
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type face_orientations: np.ndarray(shape=(N, Nf, 4), dtype=np.float32)
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations, face_orientations)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTXYZ.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getBinCounts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: np.ndarray(shape=(Nz, Ny, Nx), dtype=np.float32)
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsZ()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: np.ndarray(shape=(Nz, Ny, Nx), dtype=np.float32)
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsZ()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: np.ndarray(shape=(Nz, Ny, Nx), dtype=np.float32)
        """
        return -np.log(np.copy(self.getPCF()))

    def getX(self):
        """
        Get the array of x-values for the PCF histogram

        :return: bin centers of x-dimension of histogram
        :rtype: np.ndarray(shape=nx, dtype=np.float32)
        """
        cdef float* x = self.thisptr.getX().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>x)
        return result

    def getY(self):
        """
        Get the array of y-values for the PCF histogram

        :return: bin centers of y-dimension of histogram
        :rtype: np.ndarray(shape=ny, dtype=np.float32)
        """
        cdef float* y = self.thisptr.getY().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>y)
        return result

    def getZ(self):
        """
        Get the array of z-values for the PCF histogram

        :return: bin centers of z-dimension of histogram
        :rtype: np.ndarray(shape=nz, dtype=np.float32)
        """
        cdef float* z = self.thisptr.getZ().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsZ()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>z)
        return result

    def getNBinsX(self):
        """
        Get the number of bins in the x-dimension of histogram

        :return: nx
        :rtype: unsigned int
        """
        cdef unsigned int x = self.thisptr.getNBinsX()
        return x

    def getNBinsY(self):
        """
        Get the number of bins in the y-dimension of histogram

        :return: ny
        :rtype: unsigned int
        """
        cdef unsigned int y = self.thisptr.getNBinsY()
        return y

    def getNBinsZ(self):
        """
        Get the number of bins in the z-dimension of histogram

        :return: nz
        :rtype: unsigned int
        """
        cdef unsigned int z = self.thisptr.getNBinsZ()
        return z

    def getJacobian(self):
        """
        Get the jacobian

        :return: jacobian
        :rtype: float
        """
        cdef float j = self.thisptr.getJacobian()
        return j


cdef class PMFTRtheta:
    """Computes the PMFT [Cit2]_ for a given set of points.

    A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given :math:`R`, :math:`theta`,
    listed in the R and theta arrays. Theta is the magnitude of the smallest angle separating two particles.

    The values of R, theta to compute the pcf at are controlled by R_max, theta_max, n_R, n_theta parameters
    to the constructor. R_max, and theta_max determine the minimum/maximum distance at which to compute the pcf and
    n_R and n_theta is the number of bins in R and theta.

    .. note:: 3D: This calculation is defined for 3D systems only.

    .. moduleauthor:: Andrew Karas <askaras@umich.edu>

    :param R_max: maximum R distance at which to compute the pmft
    :param theta_max: maximum angle theta at which to compute the pmft
    :param n_R: number of bins in R
    :param n_theta: number of bins in theta
    :type R_max: float
    :type theta_max: float
    :type n_R: unsigned int
    :type n_theta: unsigned int
    """
    cdef pmft.PMFTRtheta *thisptr

    def __cinit__(self, R_max, theta_max, n_R, n_theta):
        self.thisptr = new pmft.PMFTRtheta(R_max, theta_max, n_R, n_theta)

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        """
        Resets the values of the pcf histograms in memory
        """
        self.thisptr.resetPCF()

    def accumulate(self, box, ref_points, ref_orientations, points, orientations, equivalent_orientations):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param equivalent_orientations: A set of quaternions that puts the particle into all symmetrically equivalent orientations
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type equivalent_orientations: np.ndarray(shape=((Nf, 4), dtype=np.float32)
        """
        if (ref_points.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (ref_orientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if (equivalent_orientations.dtype != np.float32):
            raise ValueError("equivalent_orientations must be a numpy float32 array")
        if len(ref_points.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(ref_orientations.shape) != 2 or len(orientations.shape) != 2:
            raise ValueError("orientations must be a 2 dimensional array")
        # handle multiple ways to input
        if equivalent_orientations.shape[1] != 4:
            raise ValueError("2nd dimension for equivalent_orientations must have 4 values: s, x, y, z")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        if ref_orientations.shape[1] != 4 or orientations.shape[1] != 4:
            raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef np.ndarray[float, ndim=2] l_equivalent_orientations = equivalent_orientations
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef unsigned int nQ = <unsigned int> equivalent_orientations.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    <vec3[float]*>l_ref_points.data,
                                    <quat[float]*>l_ref_orientations.data,
                                    nRef,
                                    <vec3[float]*>l_points.data,
                                    <quat[float]*>l_orientations.data,
                                    nP,
                                    <quat[float]*>l_equivalent_orientations.data,
                                    nQ)

    def compute(self, box, ref_points, ref_orientations, points, orientations, equivalent_orientations):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param equivalent_orientations: A set of quaternions that puts the particle into all symmetrically equivalent orientations
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type equivalent_orientations: np.ndarray(shape=((Nf, 4), dtype=np.float32)
        """
        self.thisptr.resetPCF()
        self.accumulate(box, ref_points, ref_orientations, points, orientations, equivalent_orientations)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTRtheta.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getBinCounts(self):
        """
        Get the raw bin counts.

        :return: Bin Counts
        :rtype: np.ndarray(shape=(Nr, Ntheta), dtype=np.float32)
        """
        cdef unsigned int* bin_counts = self.thisptr.getBinCounts().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBins_theta()
        cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32, <void*>bin_counts)
        return result


    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: np.ndarray(shape=(Nr, Ntheta), dtype=np.float32)
        """
        cdef float* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBins_theta()
        cdef np.ndarray[np.float32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32, <void*>pcf)
        return result

    def getPMFT(self):
        """
        Get the Potential of Mean Force and Torque.

        :return: PMFT
        :rtype: np.ndarray(shape=(Nr, Ntheta), dtype=np.float32)
        """
        return -np.log(np.copy(self.getPCF()))

    def getR(self):
        """
        Get the array of R-values for the PCF histogram

        :return: bin centers of R-dimension of histogram
        :rtype: np.ndarray(shape=nR, dtype=np.float32)
        """
        cdef float* R = self.thisptr.getR().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>R)
        return result

    def get_theta(self):
        """
        Get the array of theta-values for the PCF histogram

        :return: bin centers of theta-dimension of histogram
        :rtype: np.ndarray(shape=n_theta, dtype=np.float32)
        """
        cdef float* theta = self.thisptr.get_theta().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBins_theta()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>theta)
        return result

    def getNBinsR(self):
        """
        Get the number of bins in the R-dimension of histogram

        :return: nR
        :rtype: unsigned int
        """
        cdef unsigned int R = self.thisptr.getNBinsR()
        return R


    def getNBins_theta(self):
        """
        Get the number of bins in the theta-dimension of histogram

        :return: ntheta
        :rtype: unsigned int
        """
        cdef unsigned int theta = self.thisptr.getNBins_theta()
        return theta

    def getJacobian(self):
        """
        Get the jacobian

        :return: jacobian
        :rtype: float
        """
        cdef float j = self.thisptr.getJacobian()
        return j
