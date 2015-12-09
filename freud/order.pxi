
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._trajectory as _trajectory
cimport freud._order as order
from libc.string cimport memcpy
import numpy as np
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class BondOrder:
    """Compute the bond order diagram for the system of particles.

    Create the 2D histogram containing the number of bonds formed through the surface of a unit sphere based on the
    equatorial (Theta) and azimuthal (Phi) *check on this* angles.

    :param r_max: distance over which to calculate
    :param k: order parameter i. to be removed
    :param n: number of neighbors to find
    :param nBinsT: number of theta bins
    :param nBinsP: number of phi bins
    :type r_max: float
    :type k: unsigned int
    :type n: unsigned int
    :type nBinsT: unsigned int
    :type nBinsP: unsigned int

    :todo: remove k, it is not used as such
    """
    cdef order.BondOrder *thisptr

    def __cinit__(self, rmax, k, n, nBinsT, nBinsP):
        self.thisptr = new order.BondOrder(rmax, k, n, nBinsT, nBinsP)

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, refPoints, refOrientations, points, orientations):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param refPoints: reference points to calculate the local density
        :param refOrientations: orientations to use in computation
        :param points: points to calculate the local density
        :param orientations: orientations to use in computation
        :type box: :py:meth:`freud.trajectory.Box`
        :type refPoints: np.float32
        :type refOrientations: np.float32
        :type points: np.float32
        :type orientations: np.float32
        """
        if (refPoints.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if refPoints.ndim != 2 or points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if refPoints.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if (refOrientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("values must be a numpy float32 array")
        if refOrientations.ndim != 2 or orientations.ndim != 2:
            raise ValueError("values must be a 1 dimensional array")
        if refOrientations.shape[1] != 4 or orientations.shape[1] != 4:
            raise ValueError("the 2nd dimension must have 3 values: q0, q1, q2, q3")
        cdef np.ndarray[float, ndim=1] l_refPoints = np.ascontiguousarray(refPoints.flatten())
        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
        cdef np.ndarray[float, ndim=1] l_refOrientations = np.ascontiguousarray(refOrientations.flatten())
        cdef np.ndarray[float, ndim=1] l_orientations = np.ascontiguousarray(orientations.flatten())
        cdef unsigned int nRef = <unsigned int> refPoints.shape[0]
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>&l_refPoints[0], <quat[float]*>&l_refOrientations[0], nRef, <vec3[float]*>&l_points[0], <quat[float]*>&l_orientations[0], nP)

    def getBondOrder(self):
        """
        :return: bond order
        :rtype: np.float32
        """
        cdef float *bod = self.thisptr.getBondOrder().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsPhi()
        nbins[1] = <np.npy_intp>self.thisptr.getNBinsTheta()
        cdef np.ndarray[float, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32, <void*>bod)
        return result

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.trajectory.Box()`
        """
        return BoxFromCPP(<trajectory.Box> self.thisptr.getBox())

    def resetBondOrder(self):
        """
        resets the values of the bond order in memory
        """
        self.thisptr.resetBondOrder()

    def compute(self, box, refPoints, refOrientations, points, orientations):
        """
        Calculates the bond order histogram. Will overwrite the current histogram.

        :param box: simulation box
        :param refPoints: reference points to calculate the local density
        :param refOrientations: orientations to use in computation
        :param points: points to calculate the local density
        :param orientations: orientations to use in computation
        :type box: :py:meth:`freud.trajectory.Box`
        :type refPoints: np.float32
        :type refOrientations: np.float32
        :type points: np.float32
        :type orientations: np.float32
        """
        self.thisptr.resetBondOrder()
        self.accumulate(box, refPoints, refOrientations, points, orientations)

    def reduceBondOrder(self):
        """
        Reduces the histogram in the values over N processors to a single histogram. This is called automatically by
        :py:meth:`freud.order.BondOrder.getBondOrder()`.
        """
        self.thisptr.reduceBondOrder()

    def getTheta(self):
        """
        :return: values of bin centers for Theta
        :rtype: np.float32
        """
        cdef float *theta = self.thisptr.getTheta().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsTheta()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>theta)
        return result

    def getPhi(self):
        """
        :return: values of bin centers for Phi
        :rtype: np.float32
        """
        cdef float *phi = self.thisptr.getPhi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNBinsPhi()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>phi)
        return result

    def getNBinsTheta(self):
        """
        Get the number of bins in the Theta-dimension of histogram

        :return: nTheta
        :rtype: unsigned int
        """
        cdef unsigned int nt = self.thisptr.getNBinsTheta()
        return nt

    def getNBinsPhi(self):
        """
        Get the number of bins in the Phi-dimension of histogram

        :return: nPhi
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNBinsPhi()
        return np

cdef class EntropicBonding:
    """Compute the entropic bonds each particle in the system.

    For each particle in the system determine which other particles are in which entropic bonding sites.

    :param xmax: +/- x distance to search for bonds
    :param ymax: +/- y distance to search for bonds
    :param nx: number of bins in x
    :param ny: number of bins in x
    :param nNeighbors: number of neighbors to find
    :param nBonds: number of bonds to populate per particle
    :param bondMap: 2D array containing the bond index for each x, y coordinate
    :type xmax: float
    :type ymax: float
    :type nx: unsigned int
    :type ny: unsigned int
    :type nNeighbors: unsigned int
    :type nBonds: unsigned int
    """
    cdef order.EntropicBonding *thisptr

    def __cinit__(self, xmax, ymax, nx, ny, nNeighbors, nBonds, bondMap):
        # should I extract from the bond map (nx, ny)
        # cdef np.ndarray[unsigned int, ndim=1] l_bondMap = np.ascontiguousarray(bondMap.flatten())
        # self.thisptr = new order.EntropicBonding(xmax, ymax, nx, ny, nNeighbors, nBonds)
        self.thisptr = new order.EntropicBonding()

    # def __dealloc__(self):
    #     del self.thisptr

    # def compute(self, box, points, orientations):
    #     """
    #     Calculates the correlation function and adds to the current histogram.

    #     :param box: simulation box
    #     :param points: points to calculate the bonding
    #     :param orientations: orientations as angles to use in computation
    #     :type box: :py:meth:`freud.trajectory.Box`
    #     :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
    #     :type orientations: np.ndarray(shape=(N), dtype=np.float32)
    #     """
    #     if points.dtype != np.float32:
    #         raise ValueError("points must be a numpy float32 array")
    #     if points.ndim != 2:
    #         raise ValueError("points must be a 2 dimensional array")
    #     if points.shape[1] != 3:
    #         raise ValueError("the 2nd dimension must have 3 values: x, y, z")
    #     if orientations.dtype != np.float32:
    #         raise ValueError("values must be a numpy float32 array")
    #     if orientations.ndim != 1:
    #         raise ValueError("values must be a 1 dimensional array")
    #     cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(points.flatten())
    #     cdef np.ndarray[float, ndim=1] l_orientations = np.ascontiguousarray(orientations.flatten())
    #     cdef unsigned int nP = <unsigned int> points.shape[0]
    #     cdef _trajectory.Box l_box = _trajectory.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
    #     with nogil:
    #         self.thisptr.compute(l_box, <vec3[float]*>&l_points[0], <float*>&l_orientations[0], nP)

    # def getBonds(self):
    #     """
    #     :return: particle bonds
    #     :rtype: np.float32
    #     """
    #     cdef unsigned int *bonds = self.thisptr.getBonds().get()
    #     cdef np.npy_intp nbins[2]
    #     nbins[0] = <np.npy_intp>self.thisptr.getNBinsY()
    #     nbins[1] = <np.npy_intp>self.thisptr.getNBinsX()
    #     cdef np.ndarray[float, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32, <void*>bonds)
    #     return result

    # def getBox(self):
    #     """
    #     Get the box used in the calculation

    #     :return: Freud Box
    #     :rtype: :py:meth:`freud.trajectory.Box()`
    #     """
    #     return BoxFromCPP(<trajectory.Box> self.thisptr.getBox())

    # def getNBinsX(self):
    #     """
    #     Get the number of bins in the x-dimension of histogram

    #     :return: nx
    #     :rtype: unsigned int
    #     """
    #     cdef unsigned int nx = self.thisptr.getNBinsX()
    #     return nx

    # def getNBinsY(self):
    #     """
    #     Get the number of bins in the y-dimension of histogram

    #     :return: ny
    #     :rtype: unsigned int
    #     """
    #     cdef unsigned int ny = self.thisptr.getNBinsY()
    #     return ny
