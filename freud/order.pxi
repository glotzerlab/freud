
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._box as _box
cimport freud._order as order
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
import numpy as np
cimport numpy as np
import time

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class BondOrder:
    """Compute the bond order diagram for the system of particles.

    Available Modues of Calculation:
    * If mode=bod (Bond Order Diagram): Create the 2D histogram containing the number of bonds formed through the \
    surface of a unit sphere based on the azimuthal (Theta) and polar (Phi) angles. This is the default.

    * If mode=lbod (Local Bond Order Diagram): Create the 2D histogram containing the number of bonds formed, rotated \
    into the local orientation of the central particle, through the surface of a unit sphere based on the azimuthal \
    (:math:`\\theta`) and polar (:math:`\\phi`) angles.

    * If mode=obcd (Orientation Bond Correlation Diagram): Create the 2D histogram containing the number of bonds formed, \
    rotated by the rotation that takes the orientation of neighboring particle j to the orientation of each particle i, \
    through the surface of a unit sphere based on the azimuthal (:math:`\\theta`) and polar (:math:`\\phi`) angles.

    * If mode=oocd (Orientation Orientation Correlation Diagram): Create the 2D histogram containing the directors of \
    neighboring particles (:math:`\\hat{z}` rotated by their quaternion), rotated into the local orientation of the \
    central particle, through the surface of a unit sphere based on the azimuthal (:math:`\\theta`) and polar \
    (:math:`\\phi`) angles.

    .. moduleauthor:: Erin Teich <erteich@umich.edu>

    :param r_max: distance over which to calculate
    :param k: order parameter i. to be removed
    :param n: number of neighbors to find
    :param n_bins_t: number of theta bins
    :param n_bins_p: number of phi bins
    :type r_max: float
    :type k: unsigned int
    :type n: unsigned int
    :type n_bins_t: unsigned int
    :type n_bins_p: unsigned int

    .. todo:: remove k, it is not used as such
    """
    cdef order.BondOrder *thisptr

    def __cinit__(self, float rmax, float k, unsigned int n, unsigned int n_bins_t, unsigned int n_bins_p):
        self.thisptr = new order.BondOrder(rmax, k, n, n_bins_t, n_bins_p)

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, np.ndarray[float, ndim=2] ref_points, np.ndarray[float, ndim=2] ref_orientations,
            np.ndarray[float, ndim=2] points, np.ndarray[float, ndim=2] orientations, str mode="bod"):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations to use in computation
        :param points: points to calculate the local density
        :param orientations: orientations to use in computation
        :param mode: mode to calc bond order. "bod", "lbod", "obcd", and "oocd"
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.float32
        :type ref_orientations: np.float32
        :type points: np.float32
        :type orientations: np.float32
        :type mode: str
        """
        if (ref_points.dtype != np.float32) or (points.dtype != np.float32):
            raise ValueError("points must be a numpy float32 array")
        if ref_points.ndim != 2 or points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if ref_points.shape[1] != 3 or points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        if (ref_orientations.dtype != np.float32) or (orientations.dtype != np.float32):
            raise ValueError("values must be a numpy float32 array")
        if ref_orientations.ndim != 2 or orientations.ndim != 2:
            raise ValueError("values must be a 1 dimensional array")
        if ref_orientations.shape[1] != 4 or orientations.shape[1] != 4:
            raise ValueError("the 2nd dimension must have 4 values: q0, q1, q2, q3")

        cdef unsigned int index = 0
        if mode == "bod":
            index = 0
        elif mode == "lbod":
            index = 1
        elif mode == "obcd":
            index = 2
        elif mode == "oocd":
            index = 3
        else:
            raise RuntimeError('Unknown BOD mode: {}. Options are: bod, lbod, obcd, oocd.'.format(mode))

        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int> ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(l_box, <vec3[float]*>l_ref_points.data, <quat[float]*>l_ref_orientations.data,
                n_ref, <vec3[float]*>l_points.data, <quat[float]*>l_orientations.data, n_p, index)

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
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def resetBondOrder(self):
        """
        resets the values of the bond order in memory
        """
        self.thisptr.resetBondOrder()

    def compute(self, box, np.ndarray[float, ndim=2] ref_points, np.ndarray[float, ndim=2] ref_orientations,
            np.ndarray[float, ndim=2] points, np.ndarray[float, ndim=2] orientations, str mode="bod"):
        """
        Calculates the bond order histogram. Will overwrite the current histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations to use in computation
        :param points: points to calculate the local density
        :param orientations: orientations to use in computation
        :param mode: mode to calc bond order. "bod", "lbod", "obcd", and "oocd"
        :type box: :py:meth:`freud.box.Box`
        :type ref_points: np.float32
        :type ref_orientations: np.float32
        :type points: np.float32
        :type orientations: np.float32
        :type mode: str
        """
        self.thisptr.resetBondOrder()
        self.accumulate(box, ref_points, ref_orientations, points, orientations, mode)

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

cdef class CubaticOrderParameter:
    """Compute the Cubatic Order Parameter [Cit1]_ for a system of particles using simulated annealing instead of \
    Newton-Raphson root finding.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param t_initial: Starting temperature
    :param t_final: Final temperature
    :param scale: Scaling factor to reduce temperature
    :param n_replicates: Number of replicate simulated annealing runs
    :param seed: random seed to use in calculations. If None, system time used
    :type t_initial: float
    :type t_final: float
    :type scale: float
    :type n_replicates: unsigned int
    :type seed: unsigned int

    """
    cdef order.CubaticOrderParameter *thisptr

    def __cinit__(self, t_initial, t_final, scale, n_replicates=1, seed=None):
        # run checks
        if (t_final >= t_initial):
            raise ValueError("t_final must be less than t_initial")
        if (scale >= 1.0):
            raise ValueError("scale must be less than 1")
        if seed is None:
            seed = int(time.time())
        elif not isinstance(seed, int):
            try:
                seed = int(seed)
            finally:
                print("supplied seed could not be used. using time as seed")
                seed = time.time()

        # for c++ code
        # create generalized rank four tensor, pass into c++
        cdef np.ndarray[float, ndim=2] kd = np.eye(3, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] dijkl = np.einsum("ij,kl->ijkl", kd, kd, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] dikjl = np.einsum("ik,jl->ijkl", kd, kd, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] diljk = np.einsum("il,jk->ijkl", kd, kd, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] r4 = dijkl+dikjl+diljk
        r4 *= (2.0/5.0)
        self.thisptr = new order.CubaticOrderParameter(t_initial, t_final, scale, <float*>r4.data, n_replicates, seed)

    def compute(self, orientations):
        """
        Calculates the per-particle and global OP

        :param box: simulation box
        :param orientations: orientations to calculate the order parameter
        :type box: :py:meth:`freud.box.Box`
        :type orientations: np.float32
        """
        if (orientations.dtype != np.float32):
            raise ValueError("orientations must be a numpy float32 array")
        if orientations.ndim != 2:
            raise ValueError("orientations must be a 2 dimensional array")
        if orientations.shape[1] != 4:
            raise ValueError("the 2nd dimension must have 4 values: q0, q1, q2, q3")
        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef unsigned int num_particles = <unsigned int> orientations.shape[0]

        with nogil:
            self.thisptr.compute(<quat[float]*>l_orientations.data, num_particles, 1)

    def get_t_initial(self):
        """
        :return: value of initial temperature
        :rtype: float
        """
        return self.thisptr.getTInitial()

    def get_t_final(self):
        """
        :return: value of final temperature
        :rtype: float
        """
        return self.thisptr.getTFinal()

    def get_scale(self):
        """
        :return: value of scale
        :rtype: float
        """
        return self.thisptr.getScale()

    def get_cubatic_order_parameter(self):
        """
        :return: Cubatic Order parameter
        :rtype: float
        """
        return self.thisptr.getCubaticOrderParameter()

    def get_orientation(self):
        """
        :return: orientation of global orientation
        :rtype: np.float32
        """
        cdef quat[float] q = self.thisptr.getCubaticOrientation()
        cdef np.npy_intp nbins[1]
        nbins[0] = 4
        # This should be updated/changed at some point
        # cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>&qij)
        cdef np.ndarray[float, ndim=1] result = np.array([q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)
        return result

    def get_particle_op(self):
        """
        :return: Cubatic Order parameter
        :rtype: float
        """
        cdef float * particle_op = self.thisptr.getParticleCubaticOrderParameter().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32,<void*>particle_op)
        return result

    def get_particle_tensor(self):
        """
        :return: Rank 4 tensor corresponding to each individual particle orientation
        :rtype: np.float32
        """
        cdef float *particle_tensor = self.thisptr.getParticleTensor().get()
        cdef np.npy_intp nbins[5]
        nbins[0] = <np.npy_intp>self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp>3
        nbins[2] = <np.npy_intp>3
        nbins[3] = <np.npy_intp>3
        nbins[4] = <np.npy_intp>3
        cdef np.ndarray[np.float32_t, ndim=5] result = np.PyArray_SimpleNewFromData(5, nbins, np.NPY_FLOAT32, <void*>particle_tensor)
        return result

    def get_global_tensor(self):
        """
        :return: Rank 4 tensor corresponding to each individual particle orientation
        :rtype: np.float32
        """
        cdef float *global_tensor = self.thisptr.getGlobalTensor().get()
        cdef np.npy_intp nbins[4]
        nbins[0] = <np.npy_intp>3
        nbins[1] = <np.npy_intp>3
        nbins[2] = <np.npy_intp>3
        nbins[3] = <np.npy_intp>3
        cdef np.ndarray[np.float32_t, ndim=4] result = np.PyArray_SimpleNewFromData(4, nbins, np.NPY_FLOAT32, <void*>global_tensor)
        return result

    def get_cubatic_tensor(self):
        """
        :return: Rank 4 tensor corresponding to each individual particle orientation
        :rtype: np.float32
        """
        cdef float *cubatic_tensor = self.thisptr.getCubaticTensor().get()
        cdef np.npy_intp nbins[4]
        nbins[0] = <np.npy_intp>3
        nbins[1] = <np.npy_intp>3
        nbins[2] = <np.npy_intp>3
        nbins[3] = <np.npy_intp>3
        cdef np.ndarray[np.float32_t, ndim=4] result = np.PyArray_SimpleNewFromData(4, nbins, np.NPY_FLOAT32, <void*>cubatic_tensor)
        return result

    def get_gen_r4_tensor(self):
        """
        :return: Rank 4 tensor corresponding to each individual particle orientation
        :rtype: np.float32
        """
        cdef float *gen_r4_tensor = self.thisptr.getGenR4Tensor().get()
        cdef np.npy_intp nbins[4]
        nbins[0] = <np.npy_intp>3
        nbins[1] = <np.npy_intp>3
        nbins[2] = <np.npy_intp>3
        nbins[3] = <np.npy_intp>3
        cdef np.ndarray[np.float32_t, ndim=4] result = np.PyArray_SimpleNewFromData(4, nbins, np.NPY_FLOAT32, <void*>gen_r4_tensor)
        return result

cdef class HexOrderParameter:
    """Calculates the x-atic order parameter for each particle in the system.

    The x-atic order parameter for a particle :math:`i` and its :math:`n` neighbors :math:`j` is given by:

    :math:`\\psi_k \\left( i \\right) = \\frac{1}{n} \\sum_j^n e^{k i \\phi_{ij}}`

    The parameter :math:`k` governs the symmetry of the order parameter while the parameter :math:`n` governs the number \
    of neighbors of particle :math:`i` to average over. :math:`\\phi_{ij}` is the angle between the vector \
     :math:`r_{ij}` and :math:`\\left( 1,0 \\right)`

    .. note:: 2D: This calculation is defined for 2D systems only. However particle positions are still required to be \
    (x, y, 0)

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param rmax: +/- r distance to search for neighbors
    :param k: symmetry of order parameter (:math:`k=6` is hexatic)
    :param n: number of neighbors (:math:`n=k` if :math:`n` not specified)
    :type rmax: float
    :type k: float
    :type n: unsigned int

    .. note:: While :math:`k` is a float, this is due to its use in calculations requiring floats. Passing in \
    non-integer values will result in undefined behavior
    """
    cdef order.HexOrderParameter *thisptr

    def __cinit__(self, rmax, k=float(6.0), n=int(0)):
        self.thisptr = new order.HexOrderParameter(rmax, k, n)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param points: points to calculate the order parameter
        :type box: :py:meth:`freud.box.Box`
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*>l_points.data, nP)

    def getPsi(self):
        """
        :return: order parameter
        :rtype: np.complex64
        """
        cdef float complex *psi = self.thisptr.getPsi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>psi)
        return result

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    def getK(self):
        """
        Get the symmetry of the order parameter

        :return: k
        :rtype: float

        .. note:: While :math:`k` is a float, this is due to its use in calculations requiring floats. Passing in \
        non-integer values will result in undefined behavior
        """
        cdef float k = self.thisptr.getK()
        return k

cdef class LocalDescriptors:
    """Compute a set of descriptors (a numerical "fingerprint") of a particle's local environment.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    :param nNeigh: Maximum number of neighbors to compute descriptors for
    :param lmax: Maximum spherical harmonic l to consider
    :param rmax: Initial guess of the maximum radius to looks for neighbors
    :param negative_m: True if we should also calculate Ylm for negative m
    :type box: :py:meth:`freud.box.Box()`
    :type nNeigh: unsigned int
    :type l: unsigned int
    :type rmax: float

    """
    cdef order.LocalDescriptors *thisptr

    def __cinit__(self, nNeigh, lmax, rmax, negative_m=True):
        self.thisptr = new order.LocalDescriptors(nNeigh, lmax, rmax, negative_m)

    def __dealloc__(self):
        del self.thisptr

    def computeNList(self, box, points_ref, points=None):
        """Compute the neighbor list for bonds from a set of source points to
        a set of destination points.

        :param nNeigh: Number of neighbors to compute with
        :param points_ref: source points to calculate the order parameter
        :param points: destination points to calculate the order parameter
        :type points_ref: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32) or None

        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        if points_ref.dtype != np.float32:
            raise ValueError("points_ref must be a numpy float32 array")
        if points_ref.ndim != 2:
            raise ValueError("points_ref must be a 2 dimensional array")
        if points_ref.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")

        if points is None:
            points = points_ref

        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")

        cdef np.ndarray[float, ndim=2] l_points_ref = points_ref
        cdef unsigned int nRef = <unsigned int> points_ref.shape[0]
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        with nogil:
            self.thisptr.computeNList(l_box, <vec3[float]*>l_points_ref.data,
                                      nRef, <vec3[float]*>l_points.data, nP)

    def compute(self, box, unsigned int nNeigh, points_ref, points=None):
        """Calculates the local descriptors of bonds from a set of source
        points to a set of destination points.

        :param nNeigh: Number of neighbors to compute with
        :param points_ref: source points to calculate the order parameter
        :param points: destination points to calculate the order parameter
        :type points_ref: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32) or None

        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        if points_ref.dtype != np.float32:
            raise ValueError("points_ref must be a numpy float32 array")
        if points_ref.ndim != 2:
            raise ValueError("points_ref must be a 2 dimensional array")
        if points_ref.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")

        if points is None:
            points = points_ref

        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")

        cdef np.ndarray[float, ndim=2] l_points_ref = points_ref
        cdef unsigned int nRef = <unsigned int> points_ref.shape[0]
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        with nogil:
            self.thisptr.compute(l_box, nNeigh, <vec3[float]*>l_points_ref.data,
                                 nRef, <vec3[float]*>l_points.data, nP)

    def getSph(self):
        """
        Get a reference to the last computed spherical harmonic array

        :return: order parameter
        :rtype: np.complex64
        """
        cdef float complex *sph = self.thisptr.getSph().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        nbins[1] = <np.npy_intp>self.thisptr.getNNeigh()
        nbins[2] = <np.npy_intp>self.thisptr.getSphWidth()
        cdef np.ndarray[np.complex64_t, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_COMPLEX64, <void*>sph)
        return result

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    def getNNeigh(self):
        """
        Get the number of neighbors

        :return: n
        :rtype: unsigned int

        """
        cdef unsigned int n = self.thisptr.getNNeigh()
        return n

    def getLMax(self):
        """
        Get the maximum spherical harmonic l to calculate for

        :return: l
        :rtype: unsigned int

        """
        cdef unsigned int l = self.thisptr.getLMax()
        return l

    def getRMax(self):
        """
        Get the cutoff radius

        :return: r
        :rtype: float

        """
        cdef float r = self.thisptr.getRMax()
        return r

cdef class TransOrderParameter:
    """Compute the translational order parameter for each particle

    .. moduleauthor:: Michael Engel <engelmm@umich.edu>

    :param rmax: +/- r distance to search for neighbors
    :param k: symmetry of order parameter (:math:`k=6` is hexatic)
    :param n: number of neighbors (:math:`n=k` if :math:`n` not specified)
    :type rmax: float
    :type k: float
    :type n: unsigned int

    """
    cdef order.TransOrderParameter *thisptr

    def __cinit__(self, rmax, k=6.0, n=0):
        self.thisptr = new order.TransOrderParameter(rmax, k, n)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, orientations):
        """
        Calculates the local descriptors.

        :param box: simulation box
        :param points: points to calculate the order parameter
        :type box: :py:meth:`freud.box.Box`
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        with nogil:
            self.thisptr.compute(l_box, <vec3[float]*>l_points.data, nP)

    def getDr(self):
        """
        Get a reference to the last computed spherical harmonic array

        :return: order parameter
        :rtype: np.complex64
        """
        cdef float complex *dr = self.thisptr.getDr().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>dr)
        return result

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class LocalQl:
    """Compute the local Steinhardt rotationally invariant Ql [Cit4]_ order parameter for a set of points.

    Implements the local rotationally invariant Ql order parameter described by Steinhardt. For a particle i, \
    we calculate the average :math:`Q_l` by summing the spherical harmonics between particle :math:`i` and its \
    neighbors :math:`j` in a local region: \
    :math:`\\overline{Q}_{lm}(i) = \\frac{1}{N_b} \\displaystyle\\sum_{j=1}^{N_b} Y_{lm}(\\theta(\\vec{r}_{ij}),\
    \\phi(\\vec{r}_{ij}))`

    This is then combined in a rotationally invariant fashion to remove local orientational order as follows:
    :math:`Q_l(i)=\\sqrt{\\frac{4\pi}{2l+1} \\displaystyle\\sum_{m=-l}^{l} |\\overline{Q}_{lm}|^2 }`

    For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)

    Added first/second shell combined average Ql order parameter for a set of points:

    * Variation of the Steinhardt Ql order parameter
    * For a particle i, we calculate the average Q_l by summing the spherical harmonics between particle i and its \
    neighbors j and the neighbors k of neighbor j in a local region

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>

    :param box: simulation box
    :param rmax: Cutoff radius for the local order parameter. Values near first minima of the rdf are recommended
    :param l: Spherical harmonic quantum number l.  Must be a positive number
    :param rmin: can look at only the second shell or some arbitrary rdf region
    :type box: :py:meth:`freud.box.Box`
    :type rmax: float
    :type l: unsigned int
    :type rmin: float

    .. todo:: move box to compute, this is old API
    """
    cdef order.LocalQl *thisptr

    def __cinit__(self, box, rmax, l, rmin=0):
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new order.LocalQl(l_box, rmax, l, rmin)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)

    def computeAve(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAve(<vec3[float]*>l_points.data, nP)

    def computeNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeNorm(<vec3[float]*>l_points.data, nP)

    def computeAveNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAve(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAveNorm(<vec3[float]*>l_points.data, nP)

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def setBox(self, box):
        """
        Reset the simulation box

        :param box: simulation box
        :type box: :py:meth:`freud.box.Box`
        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)

    def getQl(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getAveQl(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getAveQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getQlNorm(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getQlNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getQlAveNorm(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getQlAveNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class LocalQlNear:
    """Compute the local Steinhardt rotationally invariant Ql order parameter [Cit4]_ for a set of points.

    Implements the local rotationally invariant Ql order parameter described by Steinhardt. For a particle i, \
    we calculate the average :math:`Q_l` by summing the spherical harmonics between particle :math:`i` and its \
    neighbors :math:`j` in a local region: \
    :math:`\\overline{Q}_{lm}(i) = \\frac{1}{N_b} \\displaystyle\\sum_{j=1}^{N_b} Y_{lm}(\\theta(\\vec{r}_{ij}),\
    \\phi(\\vec{r}_{ij}))`

    This is then combined in a rotationally invariant fashion to remove local orientational order as follows:
    :math:`Q_l(i)=\\sqrt{\\frac{4\pi}{2l+1} \\displaystyle\\sum_{m=-l}^{l} |\\overline{Q}_{lm}|^2 }`

    For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)

    Added first/second shell combined average Ql order parameter for a set of points:

    * Variation of the Steinhardt Ql order parameter
    * For a particle i, we calculate the average Q_l by summing the spherical harmonics between particle i and its \
    neighbors j and the neighbors k of neighbor j in a local region

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>

    :param box: simulation box
    :param rmax: Cutoff radius for the local order parameter. Values near first minima of the rdf are recommended
    :param l: Spherical harmonic quantum number l.  Must be a positive number
    :param kn: number of nearest neighbors. must be a positive integer
    :type box: :py:meth:`freud.box.Box`
    :type rmax: float
    :type l: unsigned int
    :type kn: unsigned int

    .. todo:: move box to compute, this is old API
    """
    cdef order.LocalQlNear *thisptr

    def __cinit__(self, box, rmax, l, kn=12):
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new order.LocalQlNear(l_box, rmax, l, kn)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)

    def computeAve(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAve(<vec3[float]*>l_points.data, nP)

    def computeNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeNorm(<vec3[float]*>l_points.data, nP)

    def computeAveNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAve(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAveNorm(<vec3[float]*>l_points.data, nP)

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def setBox(self, box):
        """
        Reset the simulation box

        :param box: simulation box
        :type box: :py:meth:`freud.box.Box`
        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)

    def getQl(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getAveQl(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getAveQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getQlNorm(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getQlNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getQlAveNorm(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getQlAveNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class LocalWl:
    """Compute the local Steinhardt rotationally invariant :math:`W_l` order parameter [Cit4]_ for a set of points.

    Implements the local rotationally invariant :math:`W_l` order parameter described by Steinhardt that can aid in distinguishing \
    between FCC, HCP, and BCC.

    For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)

    Added first/second shell combined average :math:`W_l` order parameter for a set of points:

    * Variation of the Steinhardt :math:`W_l` order parameter
    * For a particle i, we calculate the average :math:`W_l` by summing the spherical harmonics between particle i and its \
    neighbors j and the neighbors k of neighbor j in a local region

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>

    :param box: simulation box
    :param rmax: Cutoff radius for the local order parameter. Values near first minima of the rdf are recommended
    :param l: Spherical harmonic quantum number l.  Must be a positive number
    :type box: :py:meth:`freud.box.Box`
    :type rmax: float
    :type l: unsigned int

    .. todo:: move box to compute, this is old API
    """
    cdef order.LocalWl *thisptr

    def __cinit__(self, box, rmax, l):
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new order.LocalWl(l_box, rmax, l)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)

    def computeAve(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAve(<vec3[float]*>l_points.data, nP)

    def computeNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeNorm(<vec3[float]*>l_points.data, nP)

    def computeAveNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAve(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAveNorm(<vec3[float]*>l_points.data, nP)

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def setBox(self, box):
        """
        Reset the simulation box

        :param box: simulation box
        :type box: :py:meth:`freud.box.Box`
        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)

    def getQl(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getWl(self):
        """
        Get a reference to the last computed Wl for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.complex64
        """
        cdef float complex *Wl = self.thisptr.getWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Wl)
        return result

    def getAveWl(self):
        """
        Get a reference to the last computed Wl for each particle.  Returns NaN instead of Wl for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float complex *Wl = self.thisptr.getAveWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Wl)
        return result

    def getWlNorm(self):
        """
        Get a reference to the last computed Wl for each particle.  Returns NaN instead of Wl for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float complex *Wl = self.thisptr.getWlNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Wl)
        return result

    def getWlAveNorm(self):
        """
        Get a reference to the last computed Wl for each particle.  Returns NaN instead of Wl for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float complex *Wl = self.thisptr.getAveNormWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Wl)
        return result

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class LocalWlNear:
    """Compute the local Steinhardt rotationally invariant :math:`W_l` order parameter [Cit4]_ for a set of points.

    Implements the local rotationally invariant :math:`W_l` order parameter described by Steinhardt that can aid in distinguishing \
    between FCC, HCP, and BCC.

    For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)

    Added first/second shell combined average :math:`W_l` order parameter for a set of points:

    * Variation of the Steinhardt :math:`W_l` order parameter
    * For a particle i, we calculate the average :math:`W_l` by summing the spherical harmonics between particle i and its \
    neighbors j and the neighbors k of neighbor j in a local region

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>

    :param box: simulation box
    :param rmax: Cutoff radius for the local order parameter. Values near first minima of the rdf are recommended
    :param l: Spherical harmonic quantum number l.  Must be a positive number
    :param kn: Number of nearest neighbors. Must be a positive number
    :type box: :py:meth:`freud.box.Box`
    :type rmax: float
    :type l: unsigned int
    :type kn: unsigned int

    .. todo:: move box to compute, this is old API
    """
    cdef order.LocalWlNear *thisptr

    def __cinit__(self, box, rmax, l, kn=12):
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new order.LocalWlNear(l_box, rmax, l, kn)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)

    def computeAve(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAve(<vec3[float]*>l_points.data, nP)

    def computeNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeNorm(<vec3[float]*>l_points.data, nP)

    def computeAveNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAve(<vec3[float]*>l_points.data, nP)
        self.thisptr.computeAveNorm(<vec3[float]*>l_points.data, nP)

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def setBox(self, box):
        """
        Reset the simulation box

        :param box: simulation box
        :type box: :py:meth:`freud.box.Box`
        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)

    def getQl(self):
        """
        Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float *Ql = self.thisptr.getQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[float, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*>Ql)
        return result

    def getWl(self):
        """
        Get a reference to the last computed Wl for each particle.  Returns NaN instead of Ql for particles with no neighbors.

        :return: order parameter
        :rtype: np.complex64
        """
        cdef float complex *Wl = self.thisptr.getWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Wl)
        return result

    def getWlNorm(self):
        """
        Get a reference to the last computed Wl for each particle.  Returns NaN instead of Wl for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float complex *Wl = self.thisptr.getWlNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Wl)
        return result

    def getAveWl(self):
        """
        Get a reference to the last computed Wl for each particle.  Returns NaN instead of Wl for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float complex *Wl = self.thisptr.getAveWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Wl)
        return result

    def getWlAveNorm(self):
        """
        Get a reference to the last computed Wl for each particle.  Returns NaN instead of Wl for particles with no neighbors.

        :return: order parameter
        :rtype: np.float32
        """
        cdef float complex *Wl = self.thisptr.getWlAveNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Wl)
        return result

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class SolLiq:
    """Computes dot products of :math:`Q_{lm}` between particles and uses these for clustering.

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>

    :param box: simulation box
    :param rmax: Cutoff radius for the local order parameter. Values near first minima of the rdf are recommended
    :param Qthreshold: Value of dot product threshold when evaluating :math:`Q_{lm}^*(i) Q_{lm}(j)` to determine \
    if a neighbor pair is a solid-like bond. (For :math:`l=6`, 0.7 generally good for FCC or BCC structures)
    :param Sthreshold: Minimum required number of adjacent solid-link bonds for a particle to be considered solid-like \
    for clustering. (For :math:`l=6`, 6-8 generally good for FCC or BCC structures)
    :param l: Choose spherical harmonic :math:`Q_l`.  Must be positive and even.
    :type box: :py:meth:`freud.box.Box`
    :type rmax: float
    :type Qthreshold: float
    :type Sthreshold: unsigned int
    :type l: unsigned int

    .. todo:: move box to compute, this is old API
    """
    cdef order.SolLiq *thisptr

    def __cinit__(self, box, rmax, Qthreshold, Sthreshold, l):
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new order.SolLiq(l_box, rmax, Qthreshold, Sthreshold, l)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)

    def computeSolLiqVariant(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.computeSolLiqVariant(<vec3[float]*>l_points.data, nP)

    def computeSolLiqNoNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.computeSolLiqNoNorm(<vec3[float]*>l_points.data, nP)

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def setClusteringRadius(self, rcutCluster):
        """
        Reset the clustering radius

        :param rcutCluster: radius for the cluster finding
        :type rcutCluster: float
        """
        self.thisptr.setClusteringRadius(rcutCluster)

    def setBox(self, box):
        """
        Reset the simulation box

        :param box: simulation box
        :type box: :py:meth:`freud.box.Box`
        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)

    def getLargestClusterSize(self):
        """
        Returns the largest cluster size. Must compute sol-liq first

        :return: largest cluster size
        :rtype: unsigned int
        """
        cdef unsigned int clusterSize = self.thisptr.getLargestClusterSize()
        return clusterSize

    def getClusterSizes(self):
        """
        Returns the largest cluster size. Must compute sol-liq first

        :return: largest cluster size
        :rtype: np.uint32

        .. todo:: unsure of the best way to pass back...as this doesn't do what I want
        """
        cdef vector[unsigned int] clusterSizes = self.thisptr.getClusterSizes()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNumClusters()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32, <void*>&clusterSizes)
        return result

    def getQlmi(self):
        """
        Get a reference to the last computed Qlmi for each particle.

        :return: order parameter
        :rtype: np.complex64
        """
        cdef float complex *Qlmi = self.thisptr.getQlmi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Qlmi)
        return result

    def getClusters(self):
        """
        Get a reference to the last computed set of solid-like cluster indices for each particle

        :return: clusters
        :rtype: np.uint32
        """
        cdef unsigned int *clusters = self.thisptr.getClusters().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32, <void*>clusters)
        return result

    def getNumberOfConnections(self):
        """
        Get a reference to the number of connections per particle

        :return: clusters
        :rtype: np.uint32
        """
        cdef unsigned int *connections = self.thisptr.getNumberOfConnections().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32, <void*>connections)
        return result

    def getQldot_ij(self):
        """
        Get a reference to the qldot_ij values

        :return: largest cluster size
        :rtype: np.uint32

        .. todo:: figure out the size of this cause apparently its size is just its size
        """
        cdef vector[float complex] Qldot = self.thisptr.getQldot_ij()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNumClusters()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>&Qldot)
        return result

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class SolLiqNear:
    """Computes dot products of :math:`Q_{lm}` between particles and uses these for clustering.

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>

    :param box: simulation box
    :param rmax: Cutoff radius for the local order parameter. Values near first minima of the rdf are recommended
    :param Qthreshold: Value of dot product threshold when evaluating :math:`Q_{lm}^*(i) Q_{lm}(j)` to determine \
    if a neighbor pair is a solid-like bond. (For :math:`l=6`, 0.7 generally good for FCC or BCC structures)
    :param Sthreshold: Minimum required number of adjacent solid-link bonds for a particle to be considered solid-like \
    for clustering. (For :math:`l=6`, 6-8 generally good for FCC or BCC structures)
    :param l: Choose spherical harmonic :math:`Q_l`.  Must be positive and even.
    :param kn: Number of nearest neighbors. Must be a positive number
    :type box: :py:meth:`freud.box.Box`
    :type rmax: float
    :type Qthreshold: float
    :type Sthreshold: unsigned int
    :type l: unsigned int
    :type kn: unsigned int

    .. todo:: move box to compute, this is old API
    """
    cdef order.SolLiqNear *thisptr

    def __cinit__(self, box, rmax, Qthreshold, Sthreshold, l, kn=12):
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new order.SolLiqNear(l_box, rmax, Qthreshold, Sthreshold, l, kn)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.compute(<vec3[float]*>l_points.data, nP)

    def computeSolLiqVariant(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.computeSolLiqVariant(<vec3[float]*>l_points.data, nP)

    def computeSolLiqNoNorm(self, points):
        """Compute the local rotationally invariant Ql order parameter.

        :param points: points to calculate the order parameter
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension must have 3 values: x, y, z")
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        self.thisptr.computeSolLiqNoNorm(<vec3[float]*>l_points.data, nP)

    def getBox(self):
        """
        Get the box used in the calculation

        :return: Freud Box
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())

    def setClusteringRadius(self, rcutCluster):
        """
        Reset the clustering radius

        :param rcutCluster: radius for the cluster finding
        :type rcutCluster: float
        """
        self.thisptr.setClusteringRadius(rcutCluster)

    def setBox(self, box):
        """
        Reset the simulation box

        :param box: simulation box
        :type box: :py:meth:`freud.box.Box`
        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)

    def getLargestClusterSize(self):
        """
        Returns the largest cluster size. Must compute sol-liq first

        :return: largest cluster size
        :rtype: unsigned int
        """
        cdef unsigned int clusterSize = self.thisptr.getLargestClusterSize()
        return clusterSize

    def getClusterSizes(self):
        """
        Returns the largest cluster size. Must compute sol-liq first

        :return: largest cluster size
        :rtype: np.uint32

        .. todo:: unsure of the best way to pass back...as this doesn't do what I want
        """
        cdef vector[unsigned int] clusterSizes = self.thisptr.getClusterSizes()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNumClusters()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32, <void*>&clusterSizes)
        return result

    def getQlmi(self):
        """
        Get a reference to the last computed Qlmi for each particle.

        :return: order parameter
        :rtype: np.complex64
        """
        cdef float complex *Qlmi = self.thisptr.getQlmi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>Qlmi)
        return result

    def getClusters(self):
        """
        Get a reference to the last computed set of solid-like cluster indices for each particle

        :return: clusters
        :rtype: np.uint32
        """
        cdef unsigned int *clusters = self.thisptr.getClusters().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32, <void*>clusters)
        return result

    def getNumberOfConnections(self):
        """
        Get a reference to the number of connections per particle

        :return: clusters
        :rtype: np.uint32
        """
        cdef unsigned int *connections = self.thisptr.getNumberOfConnections().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32, <void*>connections)
        return result

    def getQldot_ij(self):
        """
        Get a reference to the qldot_ij values

        :return: largest cluster size
        :rtype: np.uint32

        .. todo:: figure out the size of this cause apparently its size is just its size
        """
        cdef vector[float complex] Qldot = self.thisptr.getQldot_ij()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp>self.thisptr.getNumClusters()
        cdef np.ndarray[np.complex64_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64, <void*>&Qldot)
        return result

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class MatchEnv:
    """Clusters particles according to whether their local environments match or not, according to various shape \
    matching metrics.

    .. moduleauthor:: Erin Teich <erteich@umich.edu>

    :param box: simulation box
    :param rmax: Cutoff radius for the local order parameter. Values near first minima of the rdf are recommended
    """
    cdef order.MatchEnv *thisptr

    def __cinit__(self, box, rmax, k):
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new order.MatchEnv(l_box, rmax, k)

    def __dealloc__(self):
        del self.thisptr

    def setBox(self, box):
        """
        Reset the simulation box

        :param box: simulation box
        :type box: :py:meth:`freud.box.Box`
        """
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)

    def cluster(self, points, threshold, hard_r=False):
        """Determine clusters of particles with matching environments.

        :param points: particle positions
        :param threshold: maximum magnitude of the vector difference between two vectors, below which you call them matching
        :param hard_r: if true, only add the neighbor particles to each particle's environment if they fall within the threshold of m_rmaxsq
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type threshold: np.float32
        :type hard_r: bool
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension of points must have 3 values: x, y, z")

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        self.thisptr.cluster(<vec3[float]*>l_points.data, nP, threshold, hard_r)

    def matchMotif(self, points, ref_points, threshold, hard_r=False):
        """Determine clusters of particles that match the motif provided by ref_points.

        :param points: particle positions
        :param ref_points: vectors that make up the motif against which we are matching
        :param threshold: maximum magnitude of the vector difference between two vectors, below which you call them matching
        :param hard_r: if true, only add the neighbor particles to each particle's environment if they fall within the threshold of m_rmaxsq
        :type points: np.ndarray(shape=(N, 3), dtype=np.float32)
        :type ref_points: np.ndarray(shape=(num_neigh, 3), dtype=np.float32)
        :type threshold: np.float32
        :type hard_r: bool
        """
        if points.dtype != np.float32:
            raise ValueError("points must be a numpy float32 array")
        if points.ndim != 2:
            raise ValueError("points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise ValueError("the 2nd dimension of points must have 3 values: x, y, z")
        if ref_points.dtype != np.float32:
            raise ValueError("ref_points must be a numpy float32 array")
        if ref_points.ndim != 2:
            raise ValueError("ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise ValueError("the 2nd dimension of ref_points must have 3 values: x, y, z")

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=2] l_ref_points = ref_points
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef unsigned int nRef = <unsigned int> ref_points.shape[0]

        self.thisptr.matchMotif(<vec3[float]*>l_points.data, nP, <vec3[float]*>l_ref_points.data, nRef, threshold, hard_r)

    def isSimilar(self, ref_points1, ref_points2, threshold):
        """Test if the motif provided by ref_points1 is similar to the motif provided by ref_points2.

        :param ref_points1: vectors that make up motif 1
        :param ref_points2: vectors that make up motif 2
        :param threshold: maximum magnitude of the vector difference between two vectors, below which you call them matching
        :type ref_points1: np.ndarray(shape=(num_neigh, 3), dtype=np.float32)
        :type ref_points2: np.ndarray(shape=(num_neigh, 3), dtype=np.float32)
        :type threshold: np.float32
        """
        if ref_points1.dtype != np.float32:
            raise ValueError("ref_points1 must be a numpy float32 array")
        if ref_points1.ndim != 2:
            raise ValueError("ref_points1 must be a 2 dimensional array")
        if ref_points1.shape[1] != 3:
            raise ValueError("the 2nd dimension of ref_points1 must have 3 values: x, y, z")
        if ref_points2.dtype != np.float32:
            raise ValueError("ref_points2 must be a numpy float32 array")
        if ref_points2.ndim != 2:
            raise ValueError("ref_points2 must be a 2 dimensional array")
        if ref_points2.shape[1] != 3:
            raise ValueError("the 2nd dimension of ref_points2 must have 3 values: x, y, z")

        cdef np.ndarray[float, ndim=2] l_ref_points1 = np.copy(ref_points1)
        cdef np.ndarray[float, ndim=2] l_ref_points2 = np.copy(ref_points2)
        cdef unsigned int nRef1 = <unsigned int> ref_points1.shape[0]
        cdef unsigned int nRef2 = <unsigned int> ref_points2.shape[0]
        cdef float threshold_sq = threshold*threshold

        if nRef1 != nRef2:
            raise ValueError("the number of vectors in ref_points1 must MATCH the number of vectors in ref_points2")

        cdef map[unsigned int, unsigned int] vec_map = self.thisptr.isSimilar(<vec3[float]*>l_ref_points1.data, <vec3[float]*>l_ref_points2.data, nRef1, threshold_sq)
        return vec_map

    def getClusters(self):
        """
        Get a reference to the particles, indexed into clusters according to their matching local environments

        :return: clusters
        :rtype: np.uint32
        """
        cdef unsigned int *clusters = self.thisptr.getClusters().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32, <void*>clusters)
        return result

    def getEnvironment(self, i):
        """
        Returns the set of vectors defining the environment indexed by i

        :param i: environment index
        :type i: unsigned int
        :return: the array of vectors
        :rtype: list[list[float, float, float]]
        """
        cdef vec3[float] *environment = self.thisptr.getEnvironment(i).get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp>self.thisptr.getNumNeighbors()
        nbins[1] = 3
        cdef np.ndarray[float, ndim=2] result = np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32, <void*>environment)
        return result

    def getTotEnvironment(self):
        """
        Returns the entire m_Np by m_k by 3 matrix of all environments for all particles

        :return: the array of vectors
        :rtype: list[list[list[float, float, float]]]
        """
        cdef vec3[float] *tot_environment = self.thisptr.getTotEnvironment().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp>self.thisptr.getNP()
        nbins[1] = <np.npy_intp>self.thisptr.getNumNeighbors()
        nbins[2] = 3
        cdef np.ndarray[float, ndim=3] result = np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32, <void*>tot_environment)
        return result

    def getNP(self):
        """
        Get the number of particles

        :return: np
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    def getNumClusters(self):
        """
        Get the number of clusters

        :return: num_clust
        :rtype: unsigned int
        """
        cdef unsigned int num_clust = self.thisptr.getNumClusters()
        return num_clust

cdef class Pairing2D:
    """Compute pairs for the system of particles.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param rmax: distance over which to calculate
    :param k: number of neighbors to search
    :param compDotTol: value of the dot product below which a pair is determined
    :type rmax: float
    :type k: unsigned int
    :type compDotTol: float
    """
    cdef order.Pairing2D *thisptr

    def __cinit__(self, rmax, k, compDotTol):
        self.thisptr = new order.Pairing2D(rmax, k, compDotTol)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, orientations, compOrientations):
        """
        Calculates the correlation function and adds to the current histogram.

        :param box: simulation box
        :param points: reference points to calculate the local density
        :param orientations: orientations to use in computation
        :param compOrientations: possible orientations to check for bonds
        :type box: :py:meth:`freud.box.Box`
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
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef np.ndarray[float, ndim=2] l_compOrientations = compOrientations
        cdef np.ndarray[float, ndim=1] l_orientations = orientations
        cdef unsigned int nP = <unsigned int> points.shape[0]
        cdef unsigned int nO = <unsigned int> compOrientations.shape[1]
        cdef _box.Box l_box = _box.Box(box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(), box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.compute(l_box, <vec3[float]*>l_points.data, <float*>l_orientations.data, <float*>l_compOrientations.data, nP, nO)

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
        :rtype: :py:meth:`freud.box.Box()`
        """
        return BoxFromCPP(<box.Box> self.thisptr.getBox())
