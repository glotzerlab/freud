
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

cdef class PMFXYZ:
    """
    Freud PMFTXYZ object. Wrapper for c++ pmft.PMFTXYZ()"""
    cdef pmft.PMFXYZ *thisptr

    def __cinit__(self, xMax, yMax, zMax, nx, ny, nz):
        self.thisptr = new pmft.PMFXYZ(xMax, yMax, zMax, nx, ny, nz)

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        return BoxFromCPP(self.thisptr.getBox())

    def resetPCF(self):
        self.thisptr.resetPCF()

    def accumulate(self, box, refPoints, refOrientations, points, orientations, faceOrientations):
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
        if len(refPoints.shape[1]) != 3 or len(points.shape[1]) != 3:
            raise ValueError("2nd dimension for points must have 3 values: x, y, z")
        if len(refOrientations.shape[1]) != 4 or len(orientations.shape[1]) != 4:
            raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
        if len(faceOrientations.shape[2]) != 4:
            raise ValueError("2nd dimension for orientations must have 4 values: s, x, y, z")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints)
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points)
        cdef np.ndarray[float, ndim=1] l_refOrientations = np.ascontiguousarray(refOrientations)
        cdef np.ndarray[float, ndim=1] l_orientations = np.ascontiguousarray(orientations)
        cdef np.ndarray[float, ndim=1] l_faceOrientations = np.ascontiguousarray(faceOrientations)
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
        self.thisptr.reducePCF()

    def getPCF(self):
        cdef unsigned int* pcf = self.thisptr.getPCF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBinsZ(), self.thisptr.getNBinsY(), self.thisptr.getNBinsX()), dtype=np.int32)
        memcpy(&result[0], pcf, result.nbytes)
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
