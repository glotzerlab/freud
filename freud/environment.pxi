# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import numpy as np
import time
import warnings
warnings.simplefilter('always', DeprecationWarning)
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
cimport freud._box as _box
cimport freud._order as order
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class BondOrder:
    """
    Compute the bond order diagram for the system of particles.

    Available modes of calculation:

    * If :code:`mode='bod'` (Bond Order Diagram, *default*):
      Create the 2D histogram containing the number of bonds formed through
      the surface of a unit sphere based on the azimuthal
      :math:`\\left( \\theta \\right)` and polar
      :math:`\\left( \\phi \\right)` angles.

    * If :code:`mode='lbod'` (Local Bond Order Diagram):
      Create the 2D histogram containing the number of bonds formed, rotated
      into the local orientation of the central particle, through the surface
      of a unit sphere based on the azimuthal :math:`\\left( \\theta \\right)`
      and polar :math:`\\left( \\phi \\right)` angles.

    * If :code:`mode='obcd'` (Orientation Bond Correlation Diagram):
      Create the 2D histogram containing the number of bonds formed, rotated
      by the rotation that takes the orientation of neighboring particle j to
      the orientation of each particle i, through the surface of a unit sphere
      based on the azimuthal :math:`\\left( \\theta \\right)` and polar
      :math:`\\left( \\phi \\right)` angles.

    * If :code:`mode='oocd'` (Orientation Orientation Correlation Diagram):
      Create the 2D histogram containing the directors of neighboring particles
      (:math:`\\hat{z}` rotated by their quaternion), rotated into the local
      orientation of the central particle, through the surface of a unit
      sphere based on the azimuthal :math:`\\left( \\theta \\right)` and
      polar :math:`\\left( \\phi \\right)` angles.

    .. moduleauthor:: Erin Teich <erteich@umich.edu>

    :param float r_max: distance over which to calculate
    :param k: order parameter i. to be removed
    :param n: number of neighbors to find
    :param n_bins_t: number of theta bins
    :param n_bins_p: number of phi bins
    :type k: unsigned int
    :type n: unsigned int
    :type n_bins_t: unsigned int
    :type n_bins_p: unsigned int

    .. todo:: remove k, it is not used as such
    """
    cdef order.BondOrder * thisptr
    cdef num_neigh
    cdef rmax

    def __cinit__(self, float rmax, float k, unsigned int n,
                  unsigned int n_bins_t, unsigned int n_bins_p):
        self.thisptr = new order.BondOrder(rmax, k, n, n_bins_t, n_bins_p)
        self.rmax = rmax
        self.num_neigh = n

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, ref_points, ref_orientations, points,
                   orientations, str mode="bod", nlist=None):
        """Calculates the correlation function and adds to the current
        histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations to use in computation
        :param points: points to calculate the local density
        :param orientations: orientations to use in computation
        :param mode: mode to calc bond order. "bod", "lbod", "obcd", and "oocd"
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                      find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`,
                          shape=(:math:`N_{particles}`, 3),
                          dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`,
                                shape=(:math:`N_{particles}`, 4),
                                dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                      shape=(:math:`N_{particles}`, 3),
                      dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 4),
                            dtype= :class:`numpy.float32`
        :type mode: str
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_orientations must be a 2 dimensional array")
        if ref_orientations.shape[1] != 4:
            raise TypeError('ref_orientations should be an Nx4 array')

        orientations = freud.common.convert_array(
                orientations, 2, dtype=np.float32, contiguous=True,
                dim_message="orientations must be a 2 dimensional array")
        if orientations.shape[1] != 4:
            raise TypeError('orientations should be an Nx4 array')

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
            raise RuntimeError(
                ('Unknown BOD mode: {}. Options are:'
                    'bod, lbod, obcd, oocd.').format(mode))

        defaulted_nlist = make_default_nlist_nn(
            box, ref_points, points, self.num_neigh, nlist, None, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef np.ndarray[float, ndim = 2] l_ref_points = ref_points
        cdef np.ndarray[float, ndim = 2] l_points = points
        cdef np.ndarray[float, ndim = 2] l_ref_orientations = ref_orientations
        cdef np.ndarray[float, ndim = 2] l_orientations = orientations
        cdef unsigned int n_ref = <unsigned int > ref_points.shape[0]
        cdef unsigned int n_p = <unsigned int > points.shape[0]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.accumulate(
                    l_box, nlist_ptr,
                    < vec3[float]*>l_ref_points.data,
                    < quat[float]*>l_ref_orientations.data,
                    n_ref,
                    < vec3[float]*>l_points.data,
                    < quat[float]*>l_orientations.data,
                    n_p,
                    index)
        return self

    @property
    def bond_order(self):
        """Bond order.
        """
        return self.getBondOrder()

    def getBondOrder(self):
        """Get the bond order.

        :return: bond order
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{\\phi}, N_{\\theta} \\right)`,
                dtype= :class:`numpy.float32`
        """
        cdef float * bod = self.thisptr.getBondOrder().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.thisptr.getNBinsPhi()
        nbins[1] = <np.npy_intp > self.thisptr.getNBinsTheta()
        cdef np.ndarray[float, ndim= 2] result = np.PyArray_SimpleNewFromData(
                2, nbins, np.NPY_FLOAT32, < void*>bod)
        return result

    @property
    def box(self):
        """Box used in the calculation.
        """
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        :return: freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(< box.Box > self.thisptr.getBox())

    def resetBondOrder(self):
        """Resets the values of the bond order in memory.
        """
        self.thisptr.reset()

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                mode="bod", nlist=None):
        """Calculates the bond order histogram. Will overwrite the current
        histogram.

        :param box: simulation box
        :param ref_points: reference points to calculate the local density
        :param ref_orientations: orientations to use in computation
        :param points: points to calculate the local density
        :param orientations: orientations to use in computation
        :param mode: mode to calc bond order. "bod", "lbod", "obcd", and "oocd"
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                      find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_points: :class:`numpy.ndarray`,
                          shape= :math:`\\left(N_{particles}, 3 \\right)`,
                          dtype= :class:`numpy.float32`
        :type ref_orientations: :class:`numpy.ndarray`,
                                shape= :math:`\\left(N_{particles}, 4\\right)`,
                                dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3 \\right)`,
                      dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`,
                            shape= :math:`\\left(N_{particles}, 4 \\right)`,
                            dtype= :class:`numpy.float32`
        :type mode: str
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        self.thisptr.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, mode, nlist)
        return self

    def reduceBondOrder(self):
        """Reduces the histogram in the values over N processors to a single
        histogram. This is called automatically by
        :py:meth:`freud.order.BondOrder.getBondOrder()`.
        """
        self.thisptr.reduceBondOrder()

    def getTheta(self):
        """
        :return: values of bin centers for Theta
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{\\theta} \\right)`,
                dtype= :class:`numpy.float32`
        """
        cdef float * theta = self.thisptr.getTheta().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.thisptr.getNBinsTheta()
        cdef np.ndarray[np.float32_t, ndim= 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>theta)
        return result

    def getPhi(self):
        """
        :return: values of bin centers for Phi
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{\\phi} \\right)`,
                dtype= :class:`numpy.float32`
        """
        cdef float * phi = self.thisptr.getPhi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.thisptr.getNBinsPhi()
        cdef np.ndarray[np.float32_t, ndim= 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>phi)
        return result

    def getNBinsTheta(self):
        """Get the number of bins in the Theta-dimension of histogram.

        :return: :math:`N_{\\theta}`
        :rtype: unsigned int
        """
        cdef unsigned int nt = self.thisptr.getNBinsTheta()
        return nt

    def getNBinsPhi(self):
        """Get the number of bins in the Phi-dimension of histogram.

        :return: :math:`N_{\\phi}`
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNBinsPhi()
        return np

cdef class LocalDescriptors:
    """Compute a set of descriptors (a numerical "fingerprint") of a particle's
    local environment.

    The resulting spherical harmonic array will be a complex-valued
    array of shape `(num_bonds, num_sphs)`. Spherical harmonic
    calculation can be restricted to some number of nearest neighbors
    through the `num_neighbors` argument; if a particle has more bonds
    than this number, the last one or more rows of bond spherical
    harmonics for each particle will not be set.

    .. moduleauthor:: Matthew Spellings <mspells@umich.edu>

    :param num_neighbors: Maximum number of neighbors to compute descriptors
                          for
    :param lmax: Maximum spherical harmonic :math:`l` to consider
    :param float rmax: Initial guess of the maximum radius to looks for
                       neighbors
    :param bool negative_m: True if we should also calculate :math:`Y_{lm}` for
                            negative :math:`m`
    :type num_neighbors: unsigned int
    :type lmax: unsigned int
    """
    cdef order.LocalDescriptors * thisptr
    cdef num_neigh
    cdef rmax

    known_modes = {'neighborhood': order.LocalNeighborhood,
                   'global': order.Global,
                   'particle_local': order.ParticleLocal}

    def __cinit__(self, num_neighbors, lmax, rmax, negative_m=True):
        self.thisptr = new order.LocalDescriptors(
                num_neighbors, lmax, rmax, negative_m)
        self.num_neigh = num_neighbors
        self.rmax = rmax

    def __dealloc__(self):
        del self.thisptr

    def computeNList(self, box, points_ref, points=None):
        """Compute the neighbor list for bonds from a set of source points to
        a set of destination points.

        :param points_ref: source points to calculate the order parameter
        :param points: destination points to calculate the order parameter
        :type points_ref: :class:`numpy.ndarray`,
                          shape= :math:`\\left(N_{particles}, 3 \\right)`,
                          dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3 \\right)`,
                      dtype= :class:`numpy.float32`
        """
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        points_ref = freud.common.convert_array(
                points_ref, 2, dtype=np.float32, contiguous=True,
                dim_message="points_ref must be a 2 dimensional array")
        if points_ref.shape[1] != 3:
            raise TypeError('points_ref should be an Nx3 array')

        if points is None:
            points = points_ref

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef np.ndarray[float, ndim = 2] l_points_ref = points_ref
        cdef unsigned int nRef = <unsigned int > points_ref.shape[0]
        cdef np.ndarray[float, ndim = 2] l_points = points
        cdef unsigned int nP = <unsigned int > points.shape[0]
        with nogil:
            self.thisptr.computeNList(l_box, < vec3[float]*>l_points_ref.data,
                                      nRef, < vec3[float]*>l_points.data, nP)
        return self

    def compute(self, box, unsigned int num_neighbors, points_ref, points=None,
                orientations=None, mode='neighborhood', nlist=None):
        """Calculates the local descriptors of bonds from a set of source
        points to a set of destination points.

        :param num_neighbors: Number of neighbors to compute with or to limit to,
                     if the neighbor list is precomputed
        :param points_ref: source points to calculate the order parameter
        :param points: destination points to calculate the order parameter
        :param orientations: Orientation of each reference point
        :param mode: Orientation mode to use for environments, either
                     'neighborhood' to use the orientation of the local
                     neighborhood, 'particle_local' to use the given
                     particle orientations, or 'global' to not rotate
                     environments
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                      find bonds or 'precomputed' if using :py:meth:`~.computeNList`
        :type points_ref: :class:`numpy.ndarray`,
                          shape= :math:`\\left(N_{particles}, 3 \\right)`,
                          dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3 \\right)`,
                      dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`,
                            shape= :math:`\\left(N_{particles}, 4 \\right)`,
                            dtype= :class:`numpy.float32` or None
        :type mode: str
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        if mode not in self.known_modes:
            raise RuntimeError(
                'Unknown LocalDescriptors orientation mode: {}'.format(mode))

        points_ref = freud.common.convert_array(
                points_ref, 2, dtype=np.float32, contiguous=True,
                dim_message="points_ref must be a 2 dimensional array")
        if points_ref.shape[1] != 3:
            raise TypeError('points_ref should be an Nx3 array')

        if points is None:
            points = points_ref

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef np.ndarray[float, ndim = 2] l_orientations = orientations
        if mode == 'particle_local':
            if orientations is None:
                raise RuntimeError(
                    ('Orientations must be given to orient LocalDescriptors '
                        'with particles\' orientations'))

            orientations = freud.common.convert_array(
                    orientations, 2, dtype=np.float32, contiguous=True,
                    dim_message="orientations must be a 2 dimensional array")
            if orientations.shape[1] != 4:
                raise TypeError('orientations should be an Nx4 array')

            if orientations.shape[0] != points_ref.shape[0]:
                raise ValueError(
                    "orientations must have the same size as points_ref")

            l_orientations = orientations

        cdef np.ndarray[float, ndim = 2] l_points_ref = points_ref
        cdef unsigned int nRef = <unsigned int > points_ref.shape[0]
        cdef np.ndarray[float, ndim = 2] l_points = points
        cdef unsigned int nP = <unsigned int > points.shape[0]
        cdef order.LocalDescriptorOrientation l_mode

        l_mode = self.known_modes[mode]

        self.num_neigh = num_neighbors

        cdef NeighborList nlist_
        cdef locality.NeighborList *nlist_ptr
        if nlist == 'precomputed':
            nlist_ptr = NULL
        else:
            defaulted_nlist = make_default_nlist_nn(
                box, points_ref, points, self.num_neigh, nlist, True, self.rmax)
            nlist_ = defaulted_nlist[0]
            nlist_ptr = nlist_.get_ptr()

        with nogil:
            self.thisptr.compute(
                    l_box, nlist_ptr, num_neighbors,
                    < vec3[float]*>l_points_ref.data,
                    nRef, < vec3[float]*>l_points.data, nP,
                    < quat[float]*>l_orientations.data, l_mode)
        return self

    @property
    def sph(self):
        """A reference to the last computed spherical harmonic array.
        """
        return self.getSph()

    def getSph(self):
        """Get a reference to the last computed spherical harmonic array.

        :return: order parameter
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{bonds}, \\text{SphWidth} \\right)`, \
                dtype= :class:`numpy.complex64`
        """
        cdef float complex * sph = self.thisptr.getSph().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.thisptr.getNSphs()
        nbins[1] = <np.npy_intp > self.thisptr.getSphWidth()
        cdef np.ndarray[np.complex64_t, ndim= 2
                        ] result = np.PyArray_SimpleNewFromData(
                                2, nbins, np.NPY_COMPLEX64, < void*>sph)
        return result

    @property
    def num_particles(self):
        """Get the number of particles.
        """
        return self.getNP()

    def getNP(self):
        """Get the number of particles.

        :return: :math:`N_{particles}`
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    @property
    def num_neighbors(self):
        """Get the number of neighbors.
        """
        return self.getNSphs()

    def getNSphs(self):
        """Get the number of neighbors.

        :return: :math:`N_{neighbors}`
        :rtype: unsigned int

        """
        cdef unsigned int n = self.thisptr.getNSphs()
        return n

    @property
    def l_max(self):
        """Get the maximum spherical harmonic :math:`l` to calculate for.
        """
        return self.getLMax()

    def getLMax(self):
        """Get the maximum spherical harmonic :math:`l` to calculate for.

        :return: :math:`l`
        :rtype: unsigned int

        """
        cdef unsigned int l_max = self.thisptr.getLMax()
        return l_max

    @property
    def r_max(self):
        """Get the cutoff radius.
        """
        return self.getRMax()

    def getRMax(self):
        """Get the cutoff radius.

        :return: :math:`r`
        :rtype: float

        """
        cdef float r = self.thisptr.getRMax()
        return r

cdef class MatchEnv:
    """Clusters particles according to whether their local environments match
    or not, according to various shape matching metrics.

    .. moduleauthor:: Erin Teich <erteich@umich.edu>

    :param box: Simulation box
    :param float rmax: Cutoff radius for cell list and clustering algorithm.
                       Values near first minimum of the RDF are recommended.
    :param k: Number of nearest neighbors taken to define the local environment
              of any given particle.
    :type box: :class:`freud.box.Box`
    :type k: unsigned int
    """
    cdef order.MatchEnv * thisptr
    cdef rmax
    cdef num_neigh
    cdef m_box

    def __cinit__(self, box, rmax, k):
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new order.MatchEnv(l_box, rmax, k)

        self.rmax = rmax
        self.num_neigh = k
        self.m_box = box

    def __dealloc__(self):
        del self.thisptr

    def setBox(self, box):
        """Reset the simulation box.

        :param box: simulation box
        :type box: :py:class:`freud.box.Box`
        """
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)
        self.m_box = box

    def cluster(self, points, threshold, hard_r=False, registration=False,
                global_search=False, env_nlist=None, nlist=None):
        """Determine clusters of particles with matching environments.

        :param points: particle positions
        :param float threshold: maximum magnitude of the vector difference
                                between two vectors, below which they are
                                "matching"
        :param bool hard_r: If True, add all particles that fall within the
                            threshold of m_rmaxsq to the environment
        :param bool registration: If True, first use brute force registration to
                                  orient one set of environment vectors with
                                  respect to the other set such that it
                                  minimizes the RMSD between the two sets.
        :param bool global_search: If True, do an exhaustive search wherein the
                                   environments of every single pair of
                                   particles in the simulation are compared.
                                   If False, only compare the environments of
                                   neighboring particles.
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                      find neighbors of every particle, to compare environments
        :param env_nlist: :py:class:`freud.locality.NeighborList` object to use
                          to find the environment of every particle
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3\\right)`,
                      dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        :type env_nlist: :py:class:`freud.locality.NeighborList`
        """
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        # keeping the below syntax seems to be crucial for passing unit tests
        cdef np.ndarray[float, ndim = 1] l_points = np.ascontiguousarray(
                points.flatten())
        cdef unsigned int nP = <unsigned int > points.shape[0]

        cdef locality.NeighborList * nlist_ptr
        cdef NeighborList nlist_
        cdef locality.NeighborList *env_nlist_ptr
        cdef NeighborList env_nlist_
        if hard_r:
            defaulted_nlist = make_default_nlist(
                self.m_box, points, points, self.rmax, nlist, True)
            nlist_ = defaulted_nlist[0]
            nlist_ptr = nlist_.get_ptr()

            defaulted_env_nlist = make_default_nlist(self.m_box, points, points, self.rmax, env_nlist, True)
            env_nlist_ = defaulted_env_nlist[0]
            env_nlist_ptr = env_nlist_.get_ptr()
        else:
            defaulted_nlist = make_default_nlist_nn(
                self.m_box, points, points, self.num_neigh, nlist,
                None, self.rmax)
            nlist_ = defaulted_nlist[0]
            nlist_ptr = nlist_.get_ptr()

            defaulted_env_nlist = make_default_nlist_nn(self.m_box, points, points, self.num_neigh, env_nlist, None, self.rmax)
            env_nlist_ = defaulted_env_nlist[0]
            env_nlist_ptr = env_nlist_.get_ptr()

        # keeping the below syntax seems to be crucial for passing unit tests
        self.thisptr.cluster(
                env_nlist_ptr, nlist_ptr, < vec3[float]*> & l_points[0], nP,
                threshold, hard_r, registration, global_search)

    def matchMotif(self, points, refPoints, threshold, registration=False,
                   nlist=None):
        """Determine clusters of particles that match the motif provided by
        refPoints.

        :param points: particle positions
        :param refPoints: vectors that make up the motif against which we are
                          matching
        :param float threshold: maximum magnitude of the vector difference
                                between two vectors, below which they are
                                considered "matching"
        :param bool registration: If true, first use brute force registration
                                  to orient one set of environment vectors with
                                  respect to the other set such that it
                                  minimizes the RMSD between the two sets.
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                      find bonds
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3\\right)`,
                      dtype= :class:`numpy.float32`
        :type refPoints: :class:`numpy.ndarray`,
                         shape= :math:`\\left(N_{neighbors}, 3\\right)`,
                         dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        refPoints = freud.common.convert_array(
                refPoints, 2, dtype=np.float32, contiguous=True,
                dim_message="refPoints must be a 2 dimensional array")
        if refPoints.shape[1] != 3:
            raise TypeError('refPoints should be an Nx3 array')

        # keeping the below syntax seems to be crucial for passing unit tests
        cdef np.ndarray[float, ndim = 1] l_points = np.ascontiguousarray(
                points.flatten())
        cdef np.ndarray[float, ndim = 1] l_refPoints = np.ascontiguousarray(
                refPoints.flatten())
        cdef unsigned int nP = <unsigned int > points.shape[0]
        cdef unsigned int nRef = <unsigned int > refPoints.shape[0]

        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, None, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        # keeping the below syntax seems to be crucial for passing unit tests
        self.thisptr.matchMotif(
                nlist_ptr, < vec3[float]*> & l_points[0], nP,
                < vec3[float]*> & l_refPoints[0], nRef, threshold,
                registration)

    def minRMSDMotif(self, points, refPoints, registration=False, nlist=None):
        """Rotate (if registration=True) and permute the environments of all
        particles to minimize their RMSD wrt the motif provided by refPoints.

        :param points: particle positions
        :param refPoints: vectors that make up the motif against which we are
                          matching
        :param bool registration: If true, first use brute force registration to
                                  orient one set of environment vectors with
                                  respect to the other set such that it
                                  minimizes the RMSD between the two sets.
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                      find bonds
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3\\right)`,
                      dtype= :class:`numpy.float32`
        :type refPoints: :class:`numpy.ndarray`,
                         shape= :math:`\\left(N_{neighbors}, 3\\right)`,
                         dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        :return: vector of minimal RMSD values, one value per particle.
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}\\right)`,
                dtype= :class:`numpy.float32`
        """
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        refPoints = freud.common.convert_array(
                refPoints, 2, dtype=np.float32, contiguous=True,
                dim_message="refPoints must be a 2 dimensional array")
        if refPoints.shape[1] != 3:
            raise TypeError('refPoints should be an Nx3 array')

        # keeping the below syntax seems to be crucial for passing unit tests
        cdef np.ndarray[float, ndim = 1] l_points = np.ascontiguousarray(
                points.flatten())
        cdef np.ndarray[float, ndim = 1] l_refPoints = np.ascontiguousarray(
                refPoints.flatten())
        cdef unsigned int nP = <unsigned int > points.shape[0]
        cdef unsigned int nRef = <unsigned int > refPoints.shape[0]

        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, None, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        # keeping the below syntax seems to be crucial for passing unit tests
        cdef vector[float] min_rmsd_vec = self.thisptr.minRMSDMotif(
                nlist_ptr, < vec3[float]*> & l_points[0], nP,
                < vec3[float]*> & l_refPoints[0], nRef, registration)

        return min_rmsd_vec

    def isSimilar(self, refPoints1, refPoints2, threshold, registration=False):
        """Test if the motif provided by refPoints1 is similar to the motif
        provided by refPoints2.

        :param refPoints1: vectors that make up motif 1
        :param refPoints2: vectors that make up motif 2
        :param float threshold: maximum magnitude of the vector difference
                                between two vectors, below which they are
                                considered "matching"
        :param bool registration: If true, first use brute force registration to
                                  orient one set of environment vectors with
                                  respect to the other set such that it
                                  minimizes the RMSD between the two sets.
        :type refPoints1: :class:`numpy.ndarray`,
                          shape= :math:`\\left(N_{particles}, 3\\right)`,
                          dtype= :class:`numpy.float32`
        :type refPoints2: :class:`numpy.ndarray`,
                          shape= :math:`\\left(N_{particles}, 3\\right)`,
                          dtype= :class:`numpy.float32`
        :return: a doublet that gives the rotated (or not) set of refPoints2,
                    and the mapping between the vectors of refPoints1 and
                    refPoints2 that will make them correspond to each other.
                    empty if they do not correspond to each other.
        :rtype: tuple[( :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}, 3\\right)`,
                dtype= :class:`numpy.float32`), map[int, int]]
        """
        refPoints1 = freud.common.convert_array(
                refPoints1, 2, dtype=np.float32, contiguous=True,
                dim_message="refPoints1 must be a 2 dimensional array")
        if refPoints1.shape[1] != 3:
            raise TypeError('refPoints1 should be an Nx3 array')

        refPoints2 = freud.common.convert_array(
                refPoints2, 2, dtype=np.float32, contiguous=True,
                dim_message="refPoints2 must be a 2 dimensional array")
        if refPoints2.shape[1] != 3:
            raise TypeError('refPoints2 should be an Nx3 array')

        # keeping the below syntax seems to be crucial for passing unit tests
        cdef np.ndarray[float, ndim = 1] l_refPoints1 = np.copy(
                np.ascontiguousarray(refPoints1.flatten()))
        cdef np.ndarray[float, ndim = 1] l_refPoints2 = np.copy(
                np.ascontiguousarray(refPoints2.flatten()))
        cdef unsigned int nRef1 = <unsigned int > refPoints1.shape[0]
        cdef unsigned int nRef2 = <unsigned int > refPoints2.shape[0]
        cdef float threshold_sq = threshold*threshold

        if nRef1 != nRef2:
            raise ValueError(
                ("the number of vectors in refPoints1 must MATCH the number of"
                    "vectors in refPoints2"))

        # keeping the below syntax seems to be crucial for passing unit tests
        cdef map[unsigned int, unsigned int] vec_map = self.thisptr.isSimilar(
                < vec3[float]*>&l_refPoints1[0],
                < vec3[float]*>&l_refPoints2[0],
                nRef1, threshold_sq, registration)
        cdef np.ndarray[float, ndim = 2] rot_refPoints2 = np.reshape(
                l_refPoints2, (nRef2, 3))
        return [rot_refPoints2, vec_map]

    def minimizeRMSD(self, refPoints1, refPoints2, registration=False):
        """Get the somewhat-optimal RMSD between the set of vectors refPoints1
        and the set of vectors refPoints2.

        :param refPoints1: vectors that make up motif 1
        :param refPoints2: vectors that make up motif 2
        :param registration: if true, first use brute force registration to
                                orient one set of environment vectors with
                                respect to the other set such that it minimizes
                                the RMSD between the two sets
        :type refPoints1: :class:`numpy.ndarray`,
                            shape= :math:`\\left(N_{particles}, 3\\right)`,
                            dtype= :class:`numpy.float32`
        :type refPoints2: :class:`numpy.ndarray`,
                            shape= :math:`\\left(N_{particles}, 3\\right)`,
                            dtype= :class:`numpy.float32`
        :type registration: bool
        :return: a triplet that gives the associated min_rmsd, rotated (or not)
                    set of refPoints2, and the mapping between the vectors of
                    refPoints1 and refPoints2 that somewhat minimizes the RMSD.
        :rtype: tuple[float, ( :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}, 3\\right)`,
                dtype= :class:`numpy.float32`), map[int, int]]
        """
        refPoints1 = freud.common.convert_array(
                refPoints1, 2, dtype=np.float32, contiguous=True,
                dim_message="refPoints1 must be a 2 dimensional array")
        if refPoints1.shape[1] != 3:
            raise TypeError('refPoints1 should be an Nx3 array')

        refPoints2 = freud.common.convert_array(
                refPoints2, 2, dtype=np.float32, contiguous=True,
                dim_message="refPoints2 must be a 2 dimensional array")
        if refPoints2.shape[1] != 3:
            raise TypeError('refPoints2 should be an Nx3 array')

        # keeping the below syntax seems to be crucial for passing unit tests
        cdef np.ndarray[float, ndim = 1] l_refPoints1 = np.copy(
                np.ascontiguousarray(refPoints1.flatten()))
        cdef np.ndarray[float, ndim = 1] l_refPoints2 = np.copy(
                np.ascontiguousarray(refPoints2.flatten()))
        cdef unsigned int nRef1 = <unsigned int > refPoints1.shape[0]
        cdef unsigned int nRef2 = <unsigned int > refPoints2.shape[0]

        if nRef1 != nRef2:
            raise ValueError(
                ("the number of vectors in refPoints1 must MATCH the number of"
                    "vectors in refPoints2"))

        cdef float min_rmsd = -1
        # keeping the below syntax seems to be crucial for passing unit tests
        cdef map[unsigned int, unsigned int] results_map = \
            self.thisptr.minimizeRMSD(
                    < vec3[float]*>&l_refPoints1[0],
                    < vec3[float]*>&l_refPoints2[0],
                    nRef1, min_rmsd, registration)
        cdef np.ndarray[float, ndim = 2] rot_refPoints2 = np.reshape(
                l_refPoints2, (nRef2, 3))
        return [min_rmsd, rot_refPoints2, results_map]

    def getClusters(self):
        """Get a reference to the particles, indexed into clusters according to
        their matching local environments

        :return: clusters
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}\\right)`,
                dtype= :class:`numpy.uint32`
        """
        cdef unsigned int * clusters = self.thisptr.getClusters().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp > self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim= 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_UINT32, < void*>clusters)
        return result

    def getEnvironment(self, i):
        """Returns the set of vectors defining the environment indexed by i.

        :param i: environment index
        :type i: unsigned int
        :return: the array of vectors
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{neighbors}, 3\\right)`,
                dtype= :class:`numpy.float32`
        """
        cdef vec3[float] * environment = self.thisptr.getEnvironment(i).get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.thisptr.getMaxNumNeighbors()
        nbins[1] = 3
        cdef np.ndarray[float, ndim= 2
                        ] result = np.PyArray_SimpleNewFromData(
                                2, nbins, np.NPY_FLOAT32, < void*>environment)
        return result

    @property
    def tot_environment(self):
        """Returns the entire m_Np by m_maxk by 3 matrix of all environments
        for all particles.
        """
        return self.getTotEnvironment()

    def getTotEnvironment(self):
        """Returns the entire m_Np by m_maxk by 3 matrix of all environments
        for all particles.

        :return: the array of vectors
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}, N_{neighbors}, 3\\right)`,
                dtype= :class:`numpy.float32`
        """
        cdef vec3[float] * tot_environment = self.thisptr.getTotEnvironment(
                ).get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp > self.thisptr.getNP()
        nbins[1] = <np.npy_intp > self.thisptr.getMaxNumNeighbors()
        nbins[2] = 3
        cdef np.ndarray[float, ndim= 3
                        ] result = np.PyArray_SimpleNewFromData(
                                3, nbins, np.NPY_FLOAT32,
                                < void*>tot_environment)
        return result

    @property
    def num_particles(self):
        """Get the number of particles.
        """
        return self.getNP()

    def getNP(self):
        """Get the number of particles.

        :return: :math:`N_{particles}`
        :rtype: unsigned int
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    @property
    def num_clusters(self):
        """Get the number of clusters.
        """
        return self.getNumClusters()

    def getNumClusters(self):
        """Get the number of clusters.

        :return: :math:`N_{clusters}`
        :rtype: unsigned int
        """
        cdef unsigned int num_clust = self.thisptr.getNumClusters()
        return num_clust

cdef class Pairing2D:
    """
    Compute pairs for the system of particles.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    .. deprecated:: 0.8.2
       Use :py:mod:`freud.bond` instead.

    :param float rmax: distance over which to calculate
    :param k: number of neighbors to search
    :param float compDotTol: value of the dot product below which a pair is
                             determined
    :type k: unsigned int
    """
    cdef order.Pairing2D * thisptr
    cdef rmax
    cdef num_neigh

    def __cinit__(self, rmax, k, compDotTol):
        warnings.warn("This class is deprecated, use freud.bond instead!", DeprecationWarning)
        self.thisptr = new order.Pairing2D(rmax, k, compDotTol)
        self.rmax = rmax
        self.num_neigh = k

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, orientations, compOrientations, nlist=None):
        """Calculates the correlation function and adds to the current
        histogram.

        :param box: simulation box
        :param points: reference points to calculate the local density
        :param orientations: orientations to use in computation
        :param compOrientations: possible orientations to check for bonds
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                      find bonds
        :type box: :py:class:`freud.box.Box`
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3\\right)`,
                      dtype= :class:`numpy.float32`
        :type orientations: :class:`numpy.ndarray`,
                            shape= :math:`\\left(N_{particles}\\right)`,
                            dtype= :class:`numpy.float32`
        :type compOrientations: :class:`numpy.ndarray`,
                                shape= :math:`\\left(N_{particles}\\right)`,
                                dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 1, dtype=np.float32, contiguous=True,
                dim_message="orientations must be a 1 dimensional array")

        compOrientations = freud.common.convert_array(
                compOrientations, 2, dtype=np.float32, contiguous=True,
                dim_message="compOrientations must be a 2 dimensional array")

        cdef np.ndarray[float, ndim = 2] l_points = points
        cdef np.ndarray[float, ndim = 2] l_compOrientations = compOrientations
        cdef np.ndarray[float, ndim = 1] l_orientations = orientations
        cdef unsigned int nP = <unsigned int > points.shape[0]
        cdef unsigned int nO = <unsigned int > compOrientations.shape[1]
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())

        defaulted_nlist = make_default_nlist_nn(
            box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        self.thisptr.compute(
                l_box, nlist_ptr, < vec3[float]*>l_points.data,
                < float*>l_orientations.data, < float*>l_compOrientations.data,
                nP, nO)
        return self

    @property
    def match(self):
        """Match.
        """
        return self.getMatch()

    def getMatch(self):
        """Get the match.

        :return: match
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}\\right)`,
                dtype= :class:`numpy.uint32`
        """
        cdef unsigned int * match = self.thisptr.getMatch().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.thisptr.getNumParticles()
        cdef np.ndarray[np.uint32_t, ndim= 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_UINT32, < void*>match)
        return result

    @property
    def pair(self):
        """Pair.
        """
        return self.getPair()

    def getPair(self):
        """Get the pair.

        :return: pair
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}\\right)`,
                dtype= :class:`numpy.uint32`
        """
        cdef unsigned int * pair = self.thisptr.getPair().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.thisptr.getNumParticles()
        cdef np.ndarray[np.uint32_t, ndim= 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_UINT32, < void*>pair)
        return result

    @property
    def box(self):
        """Get the box used in the calculation.
        """
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        :return: freud Box
        :rtype: :py:class:`freud.box.Box`
        """
        return BoxFromCPP(< box.Box > self.thisptr.getBox())

cdef class AngularSeparation:
    """Calculates the minimum angles of separation between particles and
    references.

    .. moduleauthor:: Erin Teich & Andrew Karas

    """
    cdef order.AngularSeparation * thisptr
    cdef num_neigh
    cdef rmax
    cdef nlist_

    def __cinit__(self, rmax, n):
        self.thisptr = new order.AngularSeparation()
        self.rmax = rmax
        self.num_neigh = int(n)
        self.nlist_ = None

    def __dealloc__(self):
        del self.thisptr

    @property
    def nlist(self):
        return self.nlist_

    def computeNeighbor(self, box, ref_ors, ors, ref_points, points,
                        equiv_quats, nlist=None):
        """Calculates the minimum angles of separation between ref_ors and ors,
        checking for underlying symmetry as encoded in equiv_quats.

        :param box: simulation box
        :param ref_ors: orientations to calculate the order parameter
        :param ref_points: points to calculate the order parameter
        :param ors: orientations (neighbors of ref_ors) to calculate the order
                    parameter
        :param points: points (neighbors of ref_points) to calculate the order
                       parameter
        :param equiv_quats: the set of all equivalent quaternions that takes
                            the particle as it is defined to some global
                            reference orientation. Important: equiv_quats must
                            include both q and -q, for all included quaternions
        :param nlist: :py:class:`freud.locality.NeighborList` object to use to
                      find bonds
        :type box: :py:class:`freud.box.Box`
        :type ref_ors: :class:`numpy.ndarray`,
                       shape= :math:`\\left(N_{particles}, 4 \\right)`,
                       dtype= :class:`numpy.float32`
        :type ref_points: :class:`numpy.ndarray`,
                          shape= :math:`\\left(N_{particles}, 3 \\right)`,
                          dtype= :class:`numpy.float32`
        :type ors: :class:`numpy.ndarray`,
                   shape= :math:`\\left(N_{particles}, 4 \\right)`,
                   dtype= :class:`numpy.float32`
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3 \\right)`,
                      dtype= :class:`numpy.float32`
        :type equiv_quats: :class:`numpy.ndarray`,
                           shape= :math:`\\left(N_{equiv}, 4 \\right)`,
                           dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
        """
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_points must be a 2 dimensional array")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        ref_ors = freud.common.convert_array(
                ref_ors, 2, dtype=np.float32, contiguous=True,
                dim_message="ref_ors must be a 2 dimensional array")
        if ref_ors.shape[1] != 4:
            raise TypeError('ref_ors should be an Nx4 array')

        ors = freud.common.convert_array(
                ors, 2, dtype=np.float32, contiguous=True,
                dim_message="ors must be a 2 dimensional array")
        if ors.shape[1] != 4:
            raise TypeError('ors should be an Nx4 array')

        equiv_quats = freud.common.convert_array(
                equiv_quats, 2, dtype=np.float32, contiguous=True,
                dim_message="equiv_quats must be a 2 dimensional array")
        if equiv_quats.shape[1] != 4:
            raise TypeError('equiv_quats should be an N_equiv x 4 array')

        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())

        defaulted_nlist = make_default_nlist_nn(
            box, ref_points, points, self.num_neigh, nlist, None, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()
        self.nlist_ = nlist_

        cdef np.ndarray[float, ndim = 2] l_ref_ors = ref_ors
        cdef np.ndarray[float, ndim = 2] l_ors = ors
        cdef np.ndarray[float, ndim = 2] l_equiv_quats = equiv_quats

        cdef unsigned int nRef = <unsigned int > ref_ors.shape[0]
        cdef unsigned int nP = <unsigned int > ors.shape[0]
        cdef unsigned int nEquiv = <unsigned int > equiv_quats.shape[0]

        with nogil:
            self.thisptr.computeNeighbor(
                    nlist_ptr,
                    < quat[float]*>l_ref_ors.data,
                    < quat[float]*>l_ors.data,
                    < quat[float]*>l_equiv_quats.data,
                    nRef, nP, nEquiv)
        return self

    def computeGlobal(self, global_ors, ors, equiv_quats):
        """Calculates the minimum angles of separation between global_ors and
        ors, checking for underlying symmetry as encoded in equiv_quats.

        :param global_ors: global reference orientations to calculate the order
                            parameter
        :param ors: orientations to calculate the order parameter
        :param equiv_quats: the set of all equivalent quaternions that takes
                            the particle as it is defined to some global
                            reference orientation. Important: equiv_quats must
                            include both q and -q, for all included quaternions
        :type ref_ors: :class:`numpy.ndarray`,
                        shape= :math:`\\left(N_{particles}, 4 \\right)`,
                        dtype= :class:`numpy.float32`
        :type ors: :class:`numpy.ndarray`,
                    shape= :math:`\\left(N_{particles}, 4 \\right)`,
                    dtype= :class:`numpy.float32`
        :type equiv_quats: :class:`numpy.ndarray`,
                            shape= :math:`\\left(N_{equiv}, 4 \\right)`,
                            dtype= :class:`numpy.float32`
        """
        global_ors = freud.common.convert_array(
                global_ors, 2, dtype=np.float32, contiguous=True,
                dim_message="global_ors must be a 2 dimensional array")
        if global_ors.shape[1] != 4:
            raise TypeError('global_ors should be an Nx4 array')

        ors = freud.common.convert_array(
                ors, 2, dtype=np.float32, contiguous=True,
                dim_message="ors must be a 2 dimensional array")
        if ors.shape[1] != 4:
            raise TypeError('ors should be an Nx4 array')

        equiv_quats = freud.common.convert_array(
                equiv_quats, 2, dtype=np.float32, contiguous=True,
                dim_message="equiv_quats must be a 2 dimensional array")
        if equiv_quats.shape[1] != 4:
            raise TypeError('equiv_quats should be an N_equiv x 4 array')

        cdef np.ndarray[float, ndim = 2] l_global_ors = global_ors
        cdef np.ndarray[float, ndim = 2] l_ors = ors
        cdef np.ndarray[float, ndim = 2] l_equiv_quats = equiv_quats

        cdef unsigned int nGlobal = <unsigned int > global_ors.shape[0]
        cdef unsigned int nP = <unsigned int > ors.shape[0]
        cdef unsigned int nEquiv = <unsigned int > equiv_quats.shape[0]

        with nogil:
            self.thisptr.computeGlobal(
                    < quat[float]*>l_global_ors.data,
                    < quat[float]*>l_ors.data,
                    < quat[float]*>l_equiv_quats.data,
                    nGlobal, nP, nEquiv)
        return self

    def getNeighborAngles(self):
        """
        :return: angles in radians
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{reference}, N_{neighbors} \\right)`,
                dtype= :class:`numpy.float32`
        """

        cdef float * neigh_ang = self.thisptr.getNeighborAngles().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > len(self.nlist)
        cdef np.ndarray[float, ndim= 1
                        ] result = np.PyArray_SimpleNewFromData(
                                1, nbins, np.NPY_FLOAT32, < void*>neigh_ang)
        return result

    def getGlobalAngles(self):
        """
        :return: angles in radians
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}, N_{global} \\right)`,
                dtype= :class:`numpy.float32`
        """

        cdef float * global_ang = self.thisptr.getGlobalAngles().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp > self.thisptr.getNP()
        nbins[1] = <np.npy_intp > self.thisptr.getNglobal()
        cdef np.ndarray[float, ndim= 2
                        ] result = np.PyArray_SimpleNewFromData(
                                2, nbins, np.NPY_FLOAT32, < void*>global_ang)
        return result

    def getNP(self):
        """Get the number of particles used in computing the last set.

        :return: :math:`N_{particles}`
        :rtype: unsigned int

        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    def getNReference(self):
        """Get the number of reference particles used in computing the neighbor
        angles.

        :return: :math:`N_{particles}`
        :rtype: unsigned int
        """
        cdef unsigned int nref = self.thisptr.getNref()
        return nref

    def getNGlobal(self):
        """Get the number of global orientations to check against.

        :return: :math:`N_{global orientations}`
        :rtype: unsigned int
        """
        cdef unsigned int nglobal = self.thisptr.getNglobal()
        return nglobal
