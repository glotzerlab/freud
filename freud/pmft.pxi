
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

    def accumulate(self, box, refPoints, refOrientations, points, orientations):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param refPoints: reference points to calculate the local density
        :param refOrientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :type box: :py:meth:`freud.trajectory.Box`
        :type refPoints: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type refOrientations: np.ndarray(shape=(N), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if (refPoints.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (refOrientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if len(refPoints.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(refOrientations.shape) != 1 or len(orientations.shape) != 1:
            raise ValueError("orientations must be a 2 dimensional array")
        if refPoints.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        # make sure that the angle is 0 to 2pi
        refOrientations = refOrientations % (2.0*np.pi)
        orientations = orientations % (2.0*np.pi)
        cdef np.ndarray l_refPoints = refPoints
        cdef np.ndarray l_points = points
        cdef np.ndarray l_refOrientations = refOrientations
        cdef np.ndarray l_orientations = orientations
        cdef unsigned int nRef = <unsigned int> refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    <vec3[float]*>l_refPoints.data,
                                    <float*>l_refOrientations.data,
                                    nRef,
                                    <vec3[float]*>l_points.data,
                                    <float*>l_orientations.data,
                                    nP)

    def compute(self, box, refPoints, refOrientations, points, orientations):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param refPoints: reference points to calculate the local density
        :param refOrientations: angles of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: angles of particles to use in calculation
        :type box: :py:meth:`freud.trajectory.Box`
        :type refPoints: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type refOrientations: np.ndarray(shape=(N), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        self.thisptr.resetPCF()
        self.accumulate(box, refPoints, refOrientations, points, orientations)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTR12.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getPCF(self):
        """
        Get the positional correlation function.

        :param copy: Specify whether returned array will be a copy of the calculated data or not
        :type copy: bool
        :return: PCF
        :rtype: np.ndarray(shape=(T1, T2, R), dtype=np.float32)

        :todo: check on the actual dimensions
        """
        cdef unsigned int* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsR()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsT2()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsT1()
        cdef np.ndarray[uint, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>pcf)
        return result

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

cdef class PMFXY2D:
    """
    Freud PMFXY2D object. Wrapper for c++ pmft.PMFXY2D()

    A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given :math:`x`, :math:`y`
    listed in the x and y arrays.

    The values of x and y to compute the pcf at are controlled by xMax, yMax, nx, and ny parameters
    to the constructor. xMax and yMax determine the minimum/maximum distance at which to compute the pcf and
    nx and ny are the number of bins in x and y.

    .. note:: 2D: This calculation is defined for 2D systems only.

    :param xMax: maximum x distance at which to compute the pmft
    :param yMax: maximum y distance at which to compute the pmft
    :param nx: number of bins in x
    :param ny: number of bins in y
    :type xMax: float
    :type yMax: float
    :type nx: unsigned int
    :type ny: unsigned int
    """
    cdef pmft.PMFXY2D *thisptr

    def __cinit__(self, xMax, yMax, nx, ny):
        self.thisptr = new pmft.PMFXY2D(xMax, yMax, nx, ny)

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

    def accumulate(self, box, refPoints, refOrientations, points, orientations):
        """
        Calculates the positional correlation function and adds to the current histogram.

        :param box: simulation box
        :param refPoints: reference points to calculate the local density
        :param refOrientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :type box: :py:meth:`freud.trajectory.Box`
        :type refPoints: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type refOrientations: np.ndarray(shape=(N), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N), dtype=np.float32)
        """
        if (refPoints.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if (refOrientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if len(refPoints.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(refOrientations.shape) != 1 or len(orientations.shape) != 1:
            raise ValueError("orientations must be a 1 dimensional array")
        if refPoints.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints.flatten())
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef np.ndarray[float, ndim=1] l_refOrientations = np.ascontiguousarray(refOrientations.flatten())
        cdef np.ndarray[float, ndim=1] l_orientations = np.ascontiguousarray(orientations.flatten())
        cdef unsigned int nRef = <unsigned int> refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box,
                                    <vec3[float]*>&l_refPoints[0],
                                    <float*>&l_refOrientations[0],
                                    nRef,
                                    <vec3[float]*>&l_points[0],
                                    <float*>&l_orientations[0],
                                    nP)

    def compute(self, box, refPoints, refOrientations, points, orientations):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

        :param box: simulation box
        :param refPoints: reference points to calculate the local density
        :param refOrientations: orientations of reference points to use in calculation
        :param points: points to calculate the local density
        :param orientations: orientations of particles to use in calculation
        :type box: :py:meth:`freud.trajectory.Box`
        :type refPoints: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type refOrientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type orientations: np.ndarray(shape=(N, 4), dtype=np.float32)
        """
        self.thisptr.resetPCF()
        self.accumulate(box, refPoints, refOrientations, points, orientations)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFXY2D.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: np.ndarray(shape=(Ny, Nx), dtype=np.float32)
        """
        cdef unsigned int* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32, <void*>pcf)
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

cdef class PMFXYZ:
    """
    Freud PMFXYZ object. Wrapper for c++ pmft.PMFXYZ()

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
        :type faceOrientations: np.ndarray(shape=(N, Nf, 4), dtype=np.float32)
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
        cdef unsigned int nRef = <unsigned int> refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef unsigned int nFaces = <unsigned int> faceOrientations.shape[1]
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

    def compute(self, box, refPoints, refOrientations, points, orientations, faceOrientations):
        """
        Calculates the positional correlation function for the given points. Will overwrite the current histogram.

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
        :type faceOrientations: np.ndarray(shape=(N, Nf, 4), dtype=np.float32)
        """
        self.thisptr.resetPCF()
        self.accumulate(box, refPoints, refOrientations, points, orientations, faceOrientations)

    def reducePCF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.pmft.PMFTXYZ.getPCF()`.
        """
        self.thisptr.reducePCF()

    def getPCF(self):
        """
        Get the positional correlation function.

        :return: PCF
        :rtype: np.ndarray(shape=(Nz, Ny, Nx), dtype=np.float32)
        """
        cdef unsigned int* pcf = self.thisptr.getPCF().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsZ()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsY()
        nbins[2] = <np.npy_intp>self.thisptr.getNBinsX()
        cdef np.ndarray[np.uint32_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_UINT32, <void*>pcf)
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
