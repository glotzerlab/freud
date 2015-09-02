
from freud.util._VectorMath cimport vec3
from freud.util._Boost cimport shared_array
cimport freud._trajectory as _trajectory
cimport freud._density as density
from libc.string cimport memcpy
from cython.operator cimport dereference as deref
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef class RDF:
    """
    Freud RDF object. Wrapper for c++ density.RDF().

    :param rmax: maximum distance to calculate
    :type rmax: float

    :param dr: distance between histogram bins
    :type dr: float
    """
    cdef density.RDF *thisptr

    def __cinit__(self, rmax, dr):
        self.thisptr = new density.RDF(rmax, dr)

    def __dealloc__(self):
        del self.thisptr

    def getBox(self):
        return BoxFromCPP(self.thisptr.getBox())

    def resetRDF(self):
        self.thisptr.resetRDF()

    def accumulate(self, box, refPoints, points):
        if (refPoints.dtype != DTYPE) or (points.dtype != DTYPE):
            raise ValueError("points must be a numpy float32 array")
        if len(refPoints.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(refPoints.shape[1]) != 3 or len(points.shape[1]) != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints)
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points)
        cdef unsigned int nRef = <unsigned int> l_refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> l_points.shape[0]
        # cdef _trajectory.Box* l_box
        # l_box = new _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>&l_refPoints[0], nRef, <vec3[float]*>&l_points[0], nP)

    def reduceRDF(self):
        self.thisptr.reduceRDF()

    def getRDF(self):
        cdef float* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getR(self):
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result

    def getNr(self):
        cdef float* Nr = self.thisptr.getNr().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], Nr, result.nbytes)
        return result

cdef class FloatCF:
    cdef density.CorrelationFunction[float] *thisptr

    def __cinit__(self, rmax, dr):
        self.thisptr = new density.CorrelationFunction[float](rmax, dr)

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, refPoints, refValues, points, values):
        if (refPoints.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if refPoints.ndim != 2 or points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(refPoints.shape[1]) != 3 or len(points.shape[1]) != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if (refValues.dtype != np.float32) or (values.dtype != np.float32):
            raise ValueError("values must be a numpy float32 array")
        if refValues.ndim != 1 or values.ndim != 1:
            raise ValueError("values must be a 1 dimensional array")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints)
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points)
        cdef np.ndarray[float, ndim=1] l_refValues = np.ascontiguousarray(refValues)
        cdef np.ndarray[float, ndim=1] l_values = np.ascontiguousarray(values)
        cdef unsigned int nRef = <unsigned int> l_refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> l_points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>&l_refPoints[0], <float*>&l_refValues[0], nRef, <vec3[float]*>&l_points[0], <float*>&l_values[0], nP)

    def getRDF(self):
        cdef float* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getBox(self):
        return BoxFromCPP(<trajectory.Box> self.thisptr.getBox())

    def resetCorrelationFunction(self):
        self.thisptr.resetCorrelationFunction()

    def compute(self, box, refPoints, refValues, points, values):
        self.thisptr.resetCorrelationFunction()
        self.accumulate(box, refPoints, refValues, points, values)

    def reduceCorrelationFunction(self):
        self.thisptr.reduceCorrelationFunction()

    def getCounts(self):
        cdef unsigned int* counts = <unsigned int*> self.thisptr.getCounts().get()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=np.uint32)
        memcpy(&result[0], counts, result.nbytes)
        return result

    def getR(self):
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result

cdef class ComplexCF:
    cdef density.CorrelationFunction[np.complex64_t] *thisptr

    def __cinit__(self, rmax, dr):
        self.thisptr = new density.CorrelationFunction[np.complex64_t](rmax, dr)

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, refPoints, refValues, points, values):
        if (refPoints.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if refPoints.ndim != 2 or points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if len(refPoints.shape[1]) != 3 or len(points.shape[1]) != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if (refValues.dtype != np.complex64) or (values.dtype != np.complex64):
            raise ValueError("values must be a numpy float32 array")
        if refValues.ndim != 1 or values.ndim != 1:
            raise ValueError("values must be a 1 dimensional array")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints)
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points)
        cdef np.ndarray[np.complex64_t, ndim=1] l_refValues = np.ascontiguousarray(refValues)
        cdef np.ndarray[np.complex64_t, ndim=1] l_values = np.ascontiguousarray(values)
        cdef unsigned int nRef = <unsigned int> l_refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> l_points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>&l_refPoints[0], <np.complex64_t*>&l_refValues[0], nRef, <vec3[float]*>&l_points[0], <np.complex64_t*>&l_values[0], nP)

    def getRDF(self):
        cdef np.complex64_t* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getBox(self):
        return BoxFromCPP(<trajectory.Box> self.thisptr.getBox())

    def resetCorrelationFunction(self):
        self.thisptr.resetCorrelationFunction()

    def compute(self, box, refPoints, refValues, points, values):
        self.thisptr.resetCorrelationFunction()
        self.accumulate(box, refPoints, refValues, points, values)

    def reduceCorrelationFunction(self):
        self.thisptr.reduceCorrelationFunction()

    def getCounts(self):
        cdef unsigned int* counts = <unsigned int*> self.thisptr.getCounts().get()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=np.uint32)
        memcpy(&result[0], counts, result.nbytes)
        return result

    def getR(self):
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result
