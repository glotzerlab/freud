
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._trajectory as _trajectory
cimport freud._pmft as pmft
from libc.string cimport memcpy
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
# DTYPE = np.float32
# ctypedef np.float32_t DTYPE_t

cdef class PMFTR12:
    """Computes the PMFT for a given set of points.

    A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given :math:`r`, :math:`\\theta_1`,
    :math:`\\theta_2` listed in the r, T1, and T2 arrays.

    The values of r, T1, T2 to compute the pcf at are controlled by rmax and nbins_r, nbins_T1, nbins_T2 parameters
    to the constructor. rmax determines the minimum/maximum r (:math:`\\min \\left( \\theta_1 \\right) =
    \\min \\left( \\theta_2 \\right) = 0`, (:math:`\\max \\left( \\theta_1 \\right) = \\max \\left( \\theta_2 \\right) = 2\\pi`)
    at which to compute the pcf and nbins_r, nbins_T1, nbins_T2 is the number of bins in r, T1, T2.

    .. note:: 2D: This calculation is defined for 2D systems only. However particle positions are still required to be \
    (x, y, 0)

    :param rMax: maximum distance at which to compute the pmft
    :param nr: number of bins in r
    :param nT1: number of bins in T1
    :param nT2: number of bins in T2
    :type rMax: float
    :type nr: unsigned int
    :type nT1: unsigned int
    :type nT2: unsigned int

    """
    cdef pmft.PMFTR12 *thisptr

    def __cinit__(self, rMax, nr, nT1, nT2):
        self.thisptr = new pmft.PMFTR12(rMax, nr, nT1, nT2)

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        self.thisptr.resetPCF()

    def accumulate(self, box, refPoints, refOrientations, points, orientations):
        if (refPoints.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (refOrientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if len(refPoints.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(refOrientations.shape) != 1 or len(orientations.shape) != 1:
            raise ValueError("orientations must be a 2 dimensional array")
        if len(refPoints.shape[1]) != 3 or len(points.shape[1]) != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints)
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points)
        cdef np.ndarray[float, ndim=1] l_refOrientations = np.ascontiguousarray(refOrientations)
        cdef np.ndarray[float, ndim=1] l_orientations = np.ascontiguousarray(orientations)
        cdef unsigned int nRef = <unsigned int> l_refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> l_points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    <vec3[float]*>&l_refPoints[0],
                                    <float*>&l_refOrientations[0],
                                    nRef,
                                    <vec3[float]*>&l_points[0],
                                    <float*>&l_orientations[0],
                                    nP)

    def reducePCF(self):
        self.thisptr.reducePCF()

    def getPCF(self):
        cdef unsigned int* pcf = self.thisptr.getPCF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsR(), self.thisptr.getNBinsT1(), self.thisptr.getNBinsT2()), dtype=np.int32)
        memcpy(&result[0], pcf, result.nbytes)
        return result

    def getR(self):
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsR()), dtype=np.float32)
        memcpy(&result[0], r, result.nbytes)
        return result

    def getT1(self):
        cdef float* T1 = self.thisptr.getT1().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsT1()), dtype=np.float32)
        memcpy(&result[0], T1, result.nbytes)
        return result

    def getT2(self):
        cdef float* T2 = self.thisptr.getT2().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsT2()), dtype=np.float32)
        memcpy(&result[0], T2, result.nbytes)
        return result

    def getNBinsR(self):
        cdef unsigned int r = self.thisptr.getNBinsR()
        return r

    def getNBinsT1(self):
        cdef unsigned int T1 = self.thisptr.getNBinsT1()
        return T1

    def getNBinsT2(self):
        cdef unsigned int T2 = self.thisptr.getNBinsT2()
        return T2

cdef class PMFXYZ:
    """
    Freud PMFTXYZ object. Wrapper for c++ pmft.PMFTXYZ()

    A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given :math:`x`, :math:`y`, :math:`z`,
    listed in the x, y, and z arrays.

    The values of x, y, z to compute the pcf at are controlled by xMax, yMax, zMax, nx, ny, and nz parameters
    to the constructor. xMax, yMax, and zMax determine the minimum/maximum distance at which to compute the pcf and
    nx, ny, nz is the number of bins in x, y, z.

    .. note:: 3D: This calculation is defined for 3D systems only.

    :param xMax: maximum x distance at which to compute the pmft
    :param yMax: maximum y distance at which to compute the pmft
    :param zMax: maximum z distance at which to compute the pmft
    :param nx: number of bins in x
    :param ny: number of bins in y
    :param nz: number of bins in z
    :type xMax: float
    :type yMax: float
    :type zMax: float
    :type nx: unsigned int
    :type ny: unsigned int
    :type nz: unsigned int
    """
    cdef pmft.PMFXYZ *thisptr

    def __cinit__(self, xMax, yMax, zMax, nx, ny, nz):
        self.thisptr = new pmft.PMFXYZ(xMax, yMax, zMax, nx, ny, nz)

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

    def accumulate(self, box, refPoints, refOrientations, points, orientations, faceOrientations):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param refPoints: reference points to calculate the local density
        :param refOrientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :param faceOrientations: orientations of particle faces to account for particle symmetry
        :type box: :py:meth:`freud.trajectory.Box`
        :type refPoints: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type refOrientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type refOrientations: np.ndarray(shape=(N, Nf, 4), dtype=np.float32)
        """
        if (refPoints.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (refOrientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if (faceOrientations.dtype != np.float32):
            raise ValueError("faceOrientations must be a numpy float32 array")
        if len(refPoints.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(refOrientations.shape) != 2 or len(orientations.shape) != 2:
            raise ValueError("orientations must be a 2 dimensional array")
        if len(faceOrientations.shape) != 3:
            raise ValueError("points must be a 3 dimensional array")
        if refPoints.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        if refOrientations.shape[1] != 4 or orientations.shape[1] != 4:
            raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
        if faceOrientations.shape[2] != 4:
            raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints.flatten())
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef np.ndarray[float, ndim=1] l_refOrientations = np.ascontiguousarray(refOrientations.flatten())
        cdef np.ndarray[float, ndim=1] l_orientations = np.ascontiguousarray(orientations.flatten())
        cdef np.ndarray[float, ndim=1] l_faceOrientations = np.ascontiguousarray(faceOrientations.flatten())
        cdef unsigned int nRef = <unsigned int> l_refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> l_points.shape[0]
        cdef unsigned int nFaces = <unsigned int> l_faceOrientations.shape[1]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    <vec3[float]*>&l_refPoints[0],
                                    <quat[float]*>&l_refOrientations[0],
                                    nRef,
                                    <vec3[float]*>&l_points[0],
                                    <quat[float]*>&l_orientations[0],
                                    nP,
                                    <quat[float]*>&l_faceOrientations[0],
                                    nFaces)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTXYZ.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getPCF(self, copy=False):
        """
        Get the positional correlation function.

        :param copy: Specify whether returned array will be a copy of the calculated data or not
        :type copy: bool
        :return: PCF
        :rtype: np.ndarray(shape=(Nz, Ny, Nx), dtype=np.float32)

        :todo: verify that the reshaping doesn't nuke everything, determine how to set the API
        """
        if copy:
            return self._getPCFCopy()
        else:
            return self._getPCFNoCopy()

    def _getPCFCopy(self):
        cdef unsigned int* pcf = self.thisptr.getPCF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsZ(), self.thisptr.getNBinsY(), self.thisptr.getNBinsX()), dtype=np.int32)
        memcpy(&result[0], pcf, result.nbytes)
        arrayShape = (self.thispth.getNBinsZ(), self.thisptr.getNBinsY(), self.thisptr.getNBinsX())
        pyResult = np.reshape(np.ascontiguousarray(result), arrayShape)
        return result

    def _getPCFNoCopy(self):
        cdef unsigned int* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsZ()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>pcf)
        return result

    def getX(self):
        cdef float* x = self.thisptr.getX().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsX()), dtype=np.float32)
        memcpy(&result[0], x, result.nbytes)
        return result

    def getY(self):
        cdef float* y = self.thisptr.getY().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsY()), dtype=np.float32)
        memcpy(&result[0], y, result.nbytes)
        return result

    def getZ(self):
        cdef float* z = self.thisptr.getZ().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsZ()), dtype=np.float32)
        memcpy(&result[0], z, result.nbytes)
        return result

    def getNBinsX(self):
        cdef unsigned int x = self.thisptr.getNBinsX()
        return x

    def getNBinsY(self):
        cdef unsigned int y = self.thisptr.getNBinsY()
        return y

    def getNBinsZ(self):
        cdef unsigned int z = self.thisptr.getNBinsZ()
        return z
