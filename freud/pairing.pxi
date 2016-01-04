
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._trajectory as _trajectory
cimport freud._pairing as pairing
from libc.string cimport memcpy
from libcpp.complex cimport complex
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Pairing2D:
    """Compute pairs for the system of particles.

    :param rmax: distance over which to calculate
    :param k: number of neighbors to search
    :param compDotTol: value of the dot product below which a pair is determined
    :type rmax: float
    :type k: unsigned int
    :type compDotTol: float
    """
    cdef pairing.Pairing2D *thisptr

    def __cinit__(self, rmax, k, compDotTol):
        self.thisptr = new pairing.Pairing2D(rmax, k, compDotTol)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, orientations, compOrientations):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param points: reference points to calculate the local density
        :param orientations: orientations to use in computation
        :param compOrientations: possible orientations to check for bonds
        :type box: :py:meth:`freud.trajectory.Box`
        :type points: np.float32
        :type orientations: np.float32
        :type compOrientations: np.float32
        """
        if (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if (orientations.dtype != np.float32) or (compOrientations.dtype != np.float32):
            raise ValueError("values must be a numpy float32 array")
        if orientations.ndim != 1:
            raise ValueError("values must be a 1 dimensional array")
        if compOrientations.ndim != 2:
            raise ValueError("values must be a 2 dimensional array")
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef np.ndarray[float, ndim=1] l_compOrientations = np.ascontiguousarray(compOrientations.flatten())
        cdef np.ndarray[float, ndim=1] l_orientations = np.ascontiguousarray(orientations.flatten())
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef unsigned int nO = <unsigned int> compOrientations.shape[1]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.compute(l_box, <vec3[float]*>&l_points[0], <float*>&l_orientations[0], <float*>&l_compOrientations[0], nP, nO)

    def getMatch(self):
        """
        :return: match
        :rtype: np.uint32
        """
        cdef unsigned int *match = self.thisptr.getMatch().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32, <void*>match)
        return result

    def getPair(self):
        """
        :return: pair
        :rtype: np.uint32
        """
        cdef unsigned int *pair = self.thisptr.getPair().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_UINT32, <void*>pair)
        return result

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
        """
        return BoxFromCPP(<trajectory.Box> self.thisptr.getBox())
