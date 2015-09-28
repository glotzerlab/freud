
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

cdef class FloatCF:
    """Computes the pairwise correlation function <p*q>(r) between two sets of points with associated values p and q.

    Two sets of points and two sets of real values associated with those
    points are given. Computing the correlation function results in an
    array of the expected (average) product of all values at a given
    radial distance.

    The values of r to compute the correlation function at are
    controlled by the rmax and dr parameters to the constructor. rmax
    determines the maximum r at which to compute the correlation
    function and dr is the step size for each bin.

    2D: CorrelationFunction properly handles 2D boxes. As with everything
    else in freud, 2D points must be passed in as 3 component vectors
    x,y,0. Failing to set 0 in the third component will lead to
    undefined behavior.

    Self-correlation: It is often the case that we wish to compute the correlation
    function of a set of points with itself. If given the same arrays
    for both points and ref_points, we omit accumulating the
    self-correlation value in the first bin.

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
        if refPoints.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if (refValues.dtype != np.float32) or (values.dtype != np.float32):
            raise ValueError("values must be a numpy float32 array")
        if refValues.ndim != 1 or values.ndim != 1:
            raise ValueError("values must be a 1 dimensional array")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints.flatten())
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef np.ndarray[float, ndim=1] l_refValues = np.ascontiguousarray(refValues.flatten())
        cdef np.ndarray[float, ndim=1] l_values = np.ascontiguousarray(values.flatten())
        cdef unsigned int nRef = <unsigned int> refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>&l_refPoints[0], <float*>&l_refValues[0], nRef, <vec3[float]*>&l_points[0], <float*>&l_values[0], nP)

    def getRDF(self):
        """
        :return: expected (average) product of all values at a given radial distance
        :rtype: np.float32
        """
        cdef float* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getBox(self):
        """
        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
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
        :return: counts of each histogram bin
        :rtype: np.uint32
        """
        cdef unsigned int* counts = <unsigned int*> self.thisptr.getCounts().get()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=np.uint32)
        memcpy(&result[0], counts, result.nbytes)
        return result

    def getR(self):
        """
        :return: values of bin centers
        :rtype: np.float32
        """
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result

cdef class ComplexCF:
    """Computes the pairwise correlation function <p*q>(r) between two sets of points with associated values p and q.

    Two sets of points and two sets of complex values associated with those
    points are given. Computing the correlation function results in an
    array of the expected (average) product of all values at a given
    radial distance.

    The values of r to compute the correlation function at are
    controlled by the rmax and dr parameters to the constructor. rmax
    determines the maximum r at which to compute the correlation
    function and dr is the step size for each bin.

    2D: CorrelationFunction properly handles 2D boxes. As with everything
    else in freud, 2D points must be passed in as 3 component vectors
    x,y,0. Failing to set 0 in the third component will lead to
    undefined behavior.

    Self-correlation: It is often the case that we wish to compute the correlation
    function of a set of points with itself. If given the same arrays
    for both points and ref_points, we omit accumulating the
    self-correlation value in the first bin.

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
            raise TypeError("points must be a numpy float32 array")
        if refPoints.ndim != 2 or points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if refPoints.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if (refValues.dtype != np.complex64) or (values.dtype != np.complex64):
            raise TypeError("values must be a numpy complex64 array")
        if refValues.ndim != 1 or values.ndim != 1:
            raise ValueError("values must be a 1 dimensional array")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints.flatten())
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef np.ndarray[np.complex64_t, ndim=1] l_refValues = np.ascontiguousarray(refValues.flatten())
        cdef np.ndarray[np.complex64_t, ndim=1] l_values = np.ascontiguousarray(values.flatten())
        cdef unsigned int nRef = <unsigned int> refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>&l_refPoints[0], <np.complex64_t*>&l_refValues[0], nRef, <vec3[float]*>&l_points[0], <np.complex64_t*>&l_values[0], nP)

    def getRDF(self):
        """
        :return: expected (average) product of all values at a given radial distance
        :rtype: np.complex64
        """
        cdef np.complex64_t* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=np.complex64)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getBox(self):
        """
        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
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
        :return: counts of each histogram bin
        :rtype: np.uint32
        """
        cdef unsigned int* counts = <unsigned int*> self.thisptr.getCounts().get()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=np.uint32)
        memcpy(&result[0], counts, result.nbytes)
        return result

    def getR(self):
        """
        :return: values of bin centers
        :rtype: np.float32
        """
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result

cdef class GaussianDensity:
    """Computes the density of a system on a grid.

    Replaces particle positions with a gaussian blur and calculates the
    contribution from the grid based upon the distance of the grid cell from the center of the Gaussian. The dimensions
    of the image (grid) are set in the constructor.

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
        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def compute(self, box, points):
        """
        Calculates the gaussian blur for the specified points. Does not accumulate (will overwrite current image).
        """
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef unsigned int nP = points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*>&l_points[0], nP)

    def getGaussianDensity(self):
        """
        :return: Image (grid) with values of gaussian
        :rtype: np.float32
        """
        cdef float* density = self.thisptr.getDensity().get()
        cdef _trajectory.Box l_box = self.thisptr.getBox()
        is2D = l_box.is2D()
        if is2D == True:
            arrayShape = (self.thisptr.getWidthY(), self.thisptr.getWidthX())
        else:
            arrayShape = (self.thispth.getWidthZ(), self.thisptr.getWidthY(), self.thisptr.getWidthX())
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=arrayShape, dtype=DTYPE).flatten()
        memcpy(&result[0], density, result.nbytes)
        pyResult = np.reshape(np.ascontiguousarray(result), arrayShape)
        return pyResult

    def resetDensity(self):
        """
        resets the values of GaussianDensity in memory
        """
        self.thisptr.resetDensity()

cdef class LocalDensity:
    """ Computes the local density around a particle

    The density of the local environment is computed and averaged for a given set of reference points in a sea of
    data points. Providing the same points calculates them against themselves. Computing the local density results in
    an array listing the value of the local density around each reference point. Also available is the number of
    neighbors for each reference point, giving the user the ability to count the number of particles in that region.

    The values to compute the local density are set in the constructor. r_cut sets the maximum distance at which to
    calculate the local density. volume is the volume of a single particle. diameter is the diameter of the circumsphere
    of an individual particle.

    2D:
    RDF properly handles 2D boxes. Requires the points to be passed in [x, y, 0]. Failing to z=0 will lead to undefined
    behavior.

    :param r_cut: maximum distance over which to calculate the density
    :type r_cut: float

    :param volume: volume of a single particle
    :type volume: float

    :param diameter: diameter of particle circumsphere
    :type diameter: float
    """
    cdef density.LocalDensity *thisptr

    def __cinit__(self, r_cut, volume, diameter):
        self.thisptr = new density.LocalDensity(r_cut, volume, diameter)

    def getBox(self):
        """
        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
        """
        return BoxFromCPP(self.thisptr.getBox())

    def compute(self, box, refPoints, points):
        """
        Calculates the local density for the specified points. Does not accumulate (will overwrite current datat).
        """
        if (refPoints.dtype != DTYPE) or (points.dtype != DTYPE):
            raise ValueError("points must be a numpy float32 array")
        if len(refPoints.shape) != 2 or len(points.shape) != 2:
            raise ValueError("points must be a 2 dimensional array")
        if refPoints.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints.flatten())
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef unsigned int nRef = <unsigned int> refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*>&l_refPoints[0], nRef, <vec3[float]*>&l_points[0], nP)

    def getDensity(self):
        """
        :return: Density array for each particle
        :rtype: np.float32
        """
        cdef float* density = self.thisptr.getDensity().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNRef()), dtype=DTYPE)
        memcpy(&result[0], density, result.nbytes)
        return result

    def getNumNeighbors(self):
        """
        :return: Density array for each particle
        :rtype: np.float32
        """
        cdef float* neighbors = self.thisptr.getNumNeighbors().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNRef()), dtype=DTYPE)
        memcpy(&result[0], neighbors, result.nbytes)
        return result

cdef class RDF:
    """ Computes RDF for supplied data

    The RDF (g(r)) is computed and averaged for a given set of reference points in a sea of data points. Providing the
    same points calculates them against themselves. Computing the RDF results in an rdf array listing the value of the
    RDF at each given r, listed in the r array.

    The values of r to compute the rdf are set by the values of rmax, dr in the constructor. rmax sets the maximum
    distance at which to calculate the g(r) while dr determines the step size for each bin.

    2D:
    RDF properly handles 2D boxes. Requires the points to be passed in [x, y, 0]. Failing to z=0 will lead to undefined
    behavior.

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
        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
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
        if refPoints.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints.flatten())
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef unsigned int nRef = <unsigned int> refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
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
        :return: histogram of rdf values
        :rtype: np.float32
        """
        cdef float* rdf = self.thisptr.getRDF().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], rdf, result.nbytes)
        return result

    def getR(self):
        """
        :return: values of the histogram bin centers
        :rtype: np.float32
        """
        cdef float* r = self.thisptr.getR().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], r, result.nbytes)
        return result

    def getNr(self):
        """
        :return: histogram of cumulative rdf values
        :rtype: np.float32
        """
        cdef float* Nr = self.thisptr.getNr().get()
        cdef np.ndarray[float, ndim=1] result = np.zeros(shape=(self.thisptr.getNBins()), dtype=DTYPE)
        memcpy(&result[0], Nr, result.nbytes)
        return result
