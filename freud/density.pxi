
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

cdef class GaussianDensity:
    """
    Freud GaussianDensity object. Wrapper for c++ density.GaussianDensity().

    :param width: number of pixels to make the image
    :type width: unsigned int

    :param width_x: number of pixels to make the image in x
    :type width_x: unsigned int

    :param width_y: number of pixels to make the image in y
    :type width_y: unsigned int

    :param width_z: number of pixels to make the image in z
    :type width_z: unsigned int

    :param r_cut: distance over which to blur
    :type r_cut: float

    :param sigma: sigma parameter for gaussian
    :type sigma: float
    """
    cdef density.GaussianDensity *thisptr

    def __cinit__(self, *args):
        if len(args) == 3:
            self.thisptr = new density.GaussianDensity(args[0], args[1], args[2])
        elif len(args) == 5:
            self.thisptr = new density.GaussianDensity(args[0], args[1], args[2],
                                                       args[3], args[4])
        else:
            raise TypeError('GaussianDensity takes exactly 3 or 5 arguments')

    def getBox(self):
        """
        return :py:meth:`freud.trajectory.Box()` object
        """
        return BoxFromCPP(self.thisptr.getBox())

    def accumulate(self, box, points):
        """
        Calculates the gaussian blur and adds it to the current GaussianDensity.
        """
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=1] l_points = points
        cdef unsigned int nP = l_points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>&l_points[0], nP)

    def compute(self, box, points):
        """
        Calculates the gaussian blur for the specified points. Will overwrite current GaussianDensity.
        """
        self.thisptr.resetDensity()
        self.accumulate(box, points)

    def getGaussianDensity(self):
        """
        returns the GaussianDensity.
        """
        cdef float* density = self.thisptr.getDensity().get()
        cdef _trajectory.Box l_box = self.thisptr.getBox()
        is2D = l_box.is2D()
        if is2D == True:
            arrayShape = (self.thisptr.getWidthY(), self.thisptr.getWidthX())
        else:
            arrayShape = (self.thispth.getWidthZ(), self.thisptr.getWidthY(), self.thisptr.getWidthX())
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=arrayShape, dtype=DTYPE)
        memcpy(&result[0], density, result.nbytes)
        return result

    def resetDensity(self):
        """
        resets the values of GaussianDensity in memory
        """
        self.thisptr.resetDensity()

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
        """
        return :py:meth:`freud.trajectory.Box()` object
        """
        return BoxFromCPP(self.thisptr.getBox())

    def resetRDF(self):
        """
        resets the values of RDF in memory
        """
        self.thisptr.resetRDF()

    def accumulate(self, box, refPoints, points):
        """
        Calculates the rdf and adds to the current rdf histogram.
        """
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
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>&l_refPoints[0], nRef, <vec3[float]*>&l_points[0], nP)

    def compute(self, box, refPoints, points):
        """
        Calculates the rdf for the specified points. Will overwrite the current histogram.
        """
        self.thisptr.resetRDF()
        self.accumulate(box, refPoints, points)

    def reduceRDF(self):
        """
        Reduces the histogram in the values over N processors to a single histogram.
        """
        self.thisptr.reduceRDF()

    def getRDF(self):
        """
        returns the RDF
        """
        cdef float* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getR(self):
        """
        returns the histogram bin center values
        """
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result

    def getNr(self):
        """
        returns the cumulative RDF.
        """
        cdef float* Nr = self.thisptr.getNr().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], Nr, result.nbytes)
        return result

cdef class FloatCF:
    """
    Freud FloatCF object. Wrapper for c++ density.FloatCF().

    :param r_max: distance over which to calculate
    :type r_max: float

    :param dr: bin size
    :type dr: float
    """
    cdef density.CorrelationFunction[float] *thisptr

    def __cinit__(self, rmax, dr):
        self.thisptr = new density.CorrelationFunction[float](rmax, dr)

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, refPoints, refValues, points, values):
        """
        Calculates the correlation function and adds to the current histogram.
        """
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
        """
        returns the RDF
        """
        cdef float* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getBox(self):
        """
        return :py:meth:`freud.trajectory.Box()` object
        """
        return BoxFromCPP(<trajectory.Box> self.thisptr.getBox())

    def resetCorrelationFunction(self):
        """
        resets the values of the correlation function histogram in memory
        """
        self.thisptr.resetCorrelationFunction()

    def compute(self, box, refPoints, refValues, points, values):
        """
        Calculates the correlation function for the given points. Will overwrite the current histogram.
        """
        self.thisptr.resetCorrelationFunction()
        self.accumulate(box, refPoints, refValues, points, values)

    def reduceCorrelationFunction(self):
        """
        Reduces the histogram in the values over N processors to a single histogram.
        """
        self.thisptr.reduceCorrelationFunction()

    def getCounts(self):
        """
        returns the counts of each bin
        """
        cdef unsigned int* counts = <unsigned int*> self.thisptr.getCounts().get()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=np.uint32)
        memcpy(&result[0], counts, result.nbytes)
        return result

    def getR(self):
        """
        returns the histogram bin center values
        """
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result

cdef class ComplexCF:
    """
    Freud ComplexCF object. Wrapper for c++ density.ComplexCF().

    :param r_max: distance over which to calculate
    :type r_max: float

    :param dr: bin size
    :type dr: float
    """
    cdef density.CorrelationFunction[np.complex64_t] *thisptr

    def __cinit__(self, rmax, dr):
        self.thisptr = new density.CorrelationFunction[np.complex64_t](rmax, dr)

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, refPoints, refValues, points, values):
        """
        Calculates the correlation function and adds to the current histogram.
        """
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
        """
        returns the RDF
        """
        cdef np.complex64_t* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getBox(self):
        """
        return :py:meth:`freud.trajectory.Box()` object
        """
        return BoxFromCPP(<trajectory.Box> self.thisptr.getBox())

    def resetCorrelationFunction(self):
        """
        resets the values of the correlation function histogram in memory
        """
        self.thisptr.resetCorrelationFunction()

    def compute(self, box, refPoints, refValues, points, values):
        """
        Calculates the correlation function for the given points. Will overwrite the current histogram.
        """
        self.thisptr.resetCorrelationFunction()
        self.accumulate(box, refPoints, refValues, points, values)

    def reduceCorrelationFunction(self):
        """
        Reduces the histogram in the values over N processors to a single histogram.
        """
        self.thisptr.reduceCorrelationFunction()

    def getCounts(self):
        """
        returns the counts of each bin
        """
        cdef unsigned int* counts = <unsigned int*> self.thisptr.getCounts().get()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=np.uint32)
        memcpy(&result[0], counts, result.nbytes)
        return result

    def getR(self):
        """
        returns the histogram bin center values
        """
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result
