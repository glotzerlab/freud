# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import numpy as np
import time
import warnings
from .errors import FreudDeprecationWarning
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
cimport freud._box as _box
cimport freud._environment as environment
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

    Args:
        r_max (float): distance over which to calculate
        k (unsigned int): order parameter i. to be removed
        n (unsigned int): number of neighbors to find
        n_bins_t (unsigned int): number of theta bins
        n_bins_p (unsigned int): number of phi bins

    .. todo:: remove k, it is not used as such
    """
    cdef environment.BondOrder * thisptr
    cdef num_neigh
    cdef rmax

    def __cinit__(self, float rmax, float k, unsigned int n,
                  unsigned int n_bins_t, unsigned int n_bins_p):
        self.thisptr = new environment.BondOrder(rmax, k, n, n_bins_t, n_bins_p)
        self.rmax = rmax
        self.num_neigh = n

    def __dealloc__(self):
        del self.thisptr

    def accumulate(self, box, ref_points, ref_orientations, points,
                   orientations, str mode="bod", nlist=None):
        """Calculates the correlation function and adds to the current
        histogram.

        Args:
          box (class:`freud.box:Box`): simulation box
          ref_points (class:`numpy.ndarray`,
                      shape=(:math:`N_{particles}`, 3),
                      dtype= :class:`numpy.float32`): reference points to
                      calculate bonds
          ref_orientations (class:`numpy.ndarray`,
                           shape=(:math:`N_{particles}`, 4),
                           dtype= :class:`numpy.float32`): reference orientations to
                           calculate bonds
          points (class:`numpy.ndarray`,
                  shape=(:math:`N_{particles}`, 3),
                  dtype= :class:`numpy.float32`): points to calculate the bonding
          orientations (class:`numpy.ndarray`,
                       shape=(:math:`N_{particles}`, 3),
                       dtype= :class:`numpy.float32`): orientations to calculate the bonding
          mode (str): mode to calc bond order. "bod", "lbod", "obcd", and "oocd" (Default value = "bod")
          nlist(class:`freud.locality.NeighborList`): NeighborList to use to find bonds (Default value = None)
                                                      find bonds (Default value = None)

        """
        box = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        ref_orientations = freud.common.convert_array(
                ref_orientations, 2, dtype=np.float32, contiguous=True,
                array_name="ref_orientations")
        if ref_orientations.shape[1] != 4:
            raise TypeError('ref_orientations should be an Nx4 array')

        orientations = freud.common.convert_array(
                orientations, 2, dtype=np.float32, contiguous=True,
                array_name="orientations")
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
        """Bond order."""
        return self.getBondOrder()

    def getBondOrder(self):
        """Get the bond order.

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{\\phi}, N_{\\theta} \\right)`,
          dtype= :class:`numpy.float32`: bond order

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
        """Box used in the calculation."""
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
          py:class:`freud.box.Box`: freud Box

        """
        return BoxFromCPP(< box.Box > self.thisptr.getBox())

    def resetBondOrder(self):
        """Resets the values of the bond order in memory."""
        self.thisptr.reset()

    def compute(self, box, ref_points, ref_orientations, points, orientations,
                mode="bod", nlist=None):
        """Calculates the bond order histogram. Will overwrite the current
        histogram.

        Args:
          box (class:`freud.box:Box`): simulation box
          ref_points (class:`numpy.ndarray`,
                      shape=(:math:`N_{particles}`, 3),
                      dtype= :class:`numpy.float32`): reference points to
                      calculate bonds
          ref_orientations (class:`numpy.ndarray`,
                           shape=(:math:`N_{particles}`, 4),
                           dtype= :class:`numpy.float32`): reference orientations to
                           calculate bonds
          points (class:`numpy.ndarray`,
                  shape=(:math:`N_{particles}`, 3),
                  dtype= :class:`numpy.float32`): points to calculate the bonding
          orientations (class:`numpy.ndarray`,
                       shape=(:math:`N_{particles}`, 3),
                       dtype= :class:`numpy.float32`): orientations to calculate the bonding
          mode (str): mode to calc bond order. "bod", "lbod", "obcd", and "oocd" (Default value = "bod")
          nlist(class:`freud.locality.NeighborList`): NeighborList to use to find bonds (Default value = None)
                                                      find bonds (Default value = None)

        """
        self.thisptr.reset()
        self.accumulate(box, ref_points, ref_orientations,
                        points, orientations, mode, nlist)
        return self

    def reduceBondOrder(self):
        """Reduces the histogram in the values over N processors to a single
        histogram. This is called automatically by
        :py:meth:`freud.environment.BondOrder.getBondOrder()`.

        """
        self.thisptr.reduceBondOrder()

    def getTheta(self):
        """

        Returns:
          class:`numpy.ndarray`,
        shape= :math:`\\left(N_{\\theta} \\right)`,
        dtype= :class:`numpy.float32`: values of bin centers for Theta

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

        Returns:
          class:`numpy.ndarray`,
        shape= :math:`\\left(N_{\\phi} \\right)`,
        dtype= :class:`numpy.float32`: values of bin centers for Phi

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

        Returns:
          unsigned int: math:`N_{\\theta}`

        """
        cdef unsigned int nt = self.thisptr.getNBinsTheta()
        return nt

    def getNBinsPhi(self):
        """Get the number of bins in the Phi-dimension of histogram.

        Returns:
          unsigned int: math:`N_{\\phi}`

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

    Args:
        num_neighbors (unsigned int): Maximum number of neighbors to compute descriptors
                              for
        lmax (unsigned int): Maximum spherical harmonic :math:`l` to consider
        rmax (float): Initial guess of the maximum radius to looks for
                           neighbors
        negative_m (bool): True if we should also calculate :math:`Y_{lm}` for
                                negative :math:`m`
    """
    cdef environment.LocalDescriptors * thisptr
    cdef num_neigh
    cdef rmax

    known_modes = {'neighborhood': environment.LocalNeighborhood,
                   'global': environment.Global,
                   'particle_local': environment.ParticleLocal}

    def __cinit__(self, num_neighbors, lmax, rmax, negative_m=True):
        self.thisptr = new environment.LocalDescriptors(
                num_neighbors, lmax, rmax, negative_m)
        self.num_neigh = num_neighbors
        self.rmax = rmax

    def __dealloc__(self):
        del self.thisptr

    def computeNList(self, box, points_ref, points=None):
        """Compute the neighbor list for bonds from a set of source points to
        a set of destination points.

        Args:
          box (class:`freud.box:Box`): simulation box
          points_ref (class:`numpy.ndarray`,
                      shape=(:math:`N_{particles}`, 3),
                      dtype= :class:`numpy.float32`): source points to calculate the
                      order parameter
          points (class:`numpy.ndarray`,
                 shape=(:math:`N_{particles}`, 3),
                 dtype= :class:`numpy.float32`): destination points to calculate the
                 order parameter (Default value = None)

        """
        box = freud.common.convert_box(box)
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        points_ref = freud.common.convert_array(
                points_ref, 2, dtype=np.float32, contiguous=True,
                array_name="points_ref")
        if points_ref.shape[1] != 3:
            raise TypeError('points_ref should be an Nx3 array')

        if points is None:
            points = points_ref

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
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

        Args:
          box (class:`freud.box:Box`): simulation box
          num_neighbors (unsigned int): Number of neighbors to compute with or to
                                        limit to, if the neighbor list is precomputed
          points_ref (class:`numpy.ndarray`,
                      shape=(:math:`N_{particles}`, 3),
                      dtype= :class:`numpy.float32`): source points to calculate the
                      order parameter
          points (class:`numpy.ndarray`,
                 shape=(:math:`N_{particles}`, 3),
                 dtype= :class:`numpy.float32`): destination points to calculate the
                 order parameter (Default value = None)
          orientations (class:`numpy.ndarray`,
                       shape=(:math:`N_{particles}`, 4),
                       dtype= :class:`numpy.float32`): Orientation of each reference
                                                       point (Default value = None)
          mode (str): Orientation mode to use for environments, either
                     'neighborhood' to use the orientation of the local
                     neighborhood, 'particle_local' to use the given particle
                     orientations, or 'global' to not rotate environments
                     (Default value = 'neighborhood')
          nlist (py:class:`freud.locality.NeighborList`): Neighborlist to use to
                    find bonds or 'precomputed' if using :py:meth:`~.computeNList`
                    (Default value = None)
        """
        box = freud.common.convert_box(box)
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        if mode not in self.known_modes:
            raise RuntimeError(
                'Unknown LocalDescriptors orientation mode: {}'.format(mode))

        points_ref = freud.common.convert_array(
                points_ref, 2, dtype=np.float32, contiguous=True,
                array_name="points_ref")
        if points_ref.shape[1] != 3:
            raise TypeError('points_ref should be an Nx3 array')

        if points is None:
            points = points_ref

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
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
                    array_name="orientations")
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
        cdef environment.LocalDescriptorOrientation l_mode

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
        """A reference to the last computed spherical harmonic array."""
        return self.getSph()

    def getSph(self):
        """Get a reference to the last computed spherical harmonic array.

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{bonds}, \\text{SphWidth} \\right)`, \
          dtype= :class:`numpy.complex64`: order parameter

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
        """Get the number of particles."""
        return self.getNP()

    def getNP(self):
        """Get the number of particles.

        Returns:
          unsigned int: math:`N_{particles}`

        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    @property
    def num_neighbors(self):
        """Get the number of neighbors."""
        return self.getNSphs()

    def getNSphs(self):
        """Get the number of neighbors.

        Returns:
          unsigned int: math:`N_{neighbors}`

        """
        cdef unsigned int n = self.thisptr.getNSphs()
        return n

    @property
    def l_max(self):
        """Get the maximum spherical harmonic :math:`l` to calculate for."""
        return self.getLMax()

    def getLMax(self):
        """Get the maximum spherical harmonic :math:`l` to calculate for.

        Returns:
          unsigned int: math:`l`

        """
        cdef unsigned int l_max = self.thisptr.getLMax()
        return l_max

    @property
    def r_max(self):
        """Get the cutoff radius."""
        return self.getRMax()

    def getRMax(self):
        """Get the cutoff radius.

        Returns:
          float: math:`r`

        """
        cdef float r = self.thisptr.getRMax()
        return r

cdef class MatchEnv:
    """Clusters particles according to whether their local environments match
    or not, according to various shape matching metrics.

    .. moduleauthor:: Erin Teich <erteich@umich.edu>

    box (:class:`freud.box.Box`): Simulation box
    rmax (float): Cutoff radius for cell list and clustering algorithm.
                  Values near first minimum of the RDF are recommended.
    k (unsigned int): Number of nearest neighbors taken to define the local environment
                      of any given particle.
    """
    cdef environment.MatchEnv * thisptr
    cdef rmax
    cdef num_neigh
    cdef m_box

    def __cinit__(self, box, rmax, k):
        box = freud.common.convert_box(box)
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new environment.MatchEnv(l_box, rmax, k)

        self.rmax = rmax
        self.num_neigh = k
        self.m_box = box

    def __dealloc__(self):
        del self.thisptr

    def setBox(self, box):
        """Reset the simulation box.

        Args:
          box(py:class:`freud.box.Box`): simulation box

        """
        box = freud.common.convert_box(box)
        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)
        self.m_box = box

    def cluster(self, points, threshold, hard_r=False, registration=False,
                global_search=False, env_nlist=None, nlist=None):
        """Determine clusters of particles with matching environments.

        Args:
          points (class:`numpy.ndarray`,
                 shape=(:math:`N_{particles}`, 3),
                 dtype= :class:`numpy.float32`): destination points to calculate the
                 order parameter (Default value = None)
          threshold (float): maximum magnitude of the vector difference between
                             two vectors, below which they are "matching"
          hard_r (bool): If True, add all particles that fall within the
                         threshold of m_rmaxsq to the environment
          registration (bool): If True, first use brute force registration to
                               orient one set of environment vectors with
                               respect to the other set such that it
                               minimizes the RMSD between the two sets.
          global_search (bool): If True, do an exhaustive search wherein the
                                environments of every single pair of
                                particles in the simulation are compared.
                                If False, only compare the environments of
                                neighboring particles.
          env_nlist (py:class:`freud.locality.NeighborList`): Neighborlist to use to
                    find the environment of every particle (Default value = None)
          nlist (py:class:`freud.locality.NeighborList`): Neighborlist to use to find
                    neighbors of every particle, to compare environments (Default value = None)

        """
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
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

        Args:
          points (class:`numpy.ndarray`,
                 shape=(:math:`N_{particles}`, 3),
                 dtype= :class:`numpy.float32`): particle positions
          refPoints (class:`numpy.ndarray`,
                    shape=(:math:`N_{particles}`, 3),
                    dtype= :class:`numpy.float32`): vectors that make up the motif
                                                    against which we are matching
          threshold (float): maximum magnitude of the vector difference
                             between two vectors, below which they are
                             considered "matching"
          registration (bool): If true, first use brute force registration
                               to orient one set of environment vectors with
                               respect to the other set such that it
                               minimizes the RMSD between the two sets (Default value = False).
          nlist (py:class:`freud.locality.NeighborList`): Neighborlist to use to
                    find bonds (Default value = None)

        """
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        refPoints = freud.common.convert_array(
                refPoints, 2, dtype=np.float32, contiguous=True,
                array_name="refPoints")
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

        Args:
          points (class:`numpy.ndarray`,
                 shape=(:math:`N_{particles}`, 3),
                 dtype= :class:`numpy.float32`): particle positions
          refPoints (class:`numpy.ndarray`,
                    shape=(:math:`N_{particles}`, 3),
                    dtype= :class:`numpy.float32`): vectors that make up the motif
                                                    against which we are matching
          registration (bool): If true, first use brute force registration
                               to orient one set of environment vectors with
                               respect to the other set such that it
                               minimizes the RMSD between the two sets (Default value = False).
          nlist (py:class:`freud.locality.NeighborList`): Neighborlist to use to
                    find bonds (Default value = None)
        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{particles}\\right)`,
          dtype= :class:`numpy.float32`: vector of minimal RMSD values, one value per particle.

        """
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        refPoints = freud.common.convert_array(
                refPoints, 2, dtype=np.float32, contiguous=True,
                array_name="refPoints")
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

        Args:
          refPoints1 (class:`numpy.ndarray`,
                     shape=(:math:`N_{particles}`, 3),
                     dtype= :class:`numpy.float32`): vectors that make up motif 1
          refPoints2 (class:`numpy.ndarray`,
                     shape=(:math:`N_{particles}`, 3),
                     dtype= :class:`numpy.float32`): vectors that make up motif 2
          threshold (float): maximum magnitude of the vector difference
                             between two vectors, below which they are
                             considered "matching"
          registration (bool): If true, first use brute force registration
                               to orient one set of environment vectors with
                               respect to the other set such that it
                               minimizes the RMSD between the two sets (Default value = False).

        Returns:
          tuple[(:class:`numpy.ndarray`,
                   shape= :math:`\\left(N_{particles}, 3\\right)`,
                   dtype= :class:`numpy.float32`),
                map[int, int]]: a doublet that gives the rotated (or not) set of refPoints2,
                                and the mapping between the vectors of refPoints1 and
                                refPoints2 that will make them correspond to each other.
                                empty if they do not correspond to each other.

        """
        refPoints1 = freud.common.convert_array(
                refPoints1, 2, dtype=np.float32, contiguous=True,
                array_name="refPoints1")
        if refPoints1.shape[1] != 3:
            raise TypeError('refPoints1 should be an Nx3 array')

        refPoints2 = freud.common.convert_array(
                refPoints2, 2, dtype=np.float32, contiguous=True,
                array_name="refPoints2")
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

        Args:
          refPoints1 (class:`numpy.ndarray`,
                     shape=(:math:`N_{particles}`, 3),
                     dtype= :class:`numpy.float32`): vectors that make up motif 1
          refPoints2 (class:`numpy.ndarray`,
                     shape=(:math:`N_{particles}`, 3),
                     dtype= :class:`numpy.float32`): vectors that make up motif 2
          registration (bool): If true, first use brute force registration
                               to orient one set of environment vectors with
                               respect to the other set such that it
                               minimizes the RMSD between the two sets (Default value = False).

        Returns:
          tuple[float,
                (:class:`numpy.ndarray`,
                 shape= :math:`\\left(N_{particles}, 3\\right)`,
                 dtype= :class:`numpy.float32`),
                map[int, int]]: a triplet that gives the associated min_rmsd, rotated (or not)
                                set of refPoints2, and the mapping between the vectors of
                                refPoints1 and refPoints2 that somewhat minimizes the RMSD.

        """
        refPoints1 = freud.common.convert_array(
                refPoints1, 2, dtype=np.float32, contiguous=True,
                array_name="refPoints1")
        if refPoints1.shape[1] != 3:
            raise TypeError('refPoints1 should be an Nx3 array')

        refPoints2 = freud.common.convert_array(
                refPoints2, 2, dtype=np.float32, contiguous=True,
                array_name="refPoints2")
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

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{particles}\\right)`,
          dtype= :class:`numpy.uint32`: clusters

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

        Args:
          i (unsigned int): environment index

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{neighbors}, 3\\right)`,
          dtype= :class:`numpy.float32`: the array of vectors

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

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{particles}, N_{neighbors}, 3\\right)`,
          dtype= :class:`numpy.float32`: the array of vectors

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
        """Get the number of particles."""
        return self.getNP()

    def getNP(self):
        """Get the number of particles.

        Returns:
          unsigned int: math:`N_{particles}`

        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    @property
    def num_clusters(self):
        """Get the number of clusters."""
        return self.getNumClusters()

    def getNumClusters(self):
        """Get the number of clusters.

        Returns:
          unsigned int: math:`N_{clusters}`

        """
        cdef unsigned int num_clust = self.thisptr.getNumClusters()
        return num_clust

cdef class Pairing2D:
    """
    Compute pairs for the system of particles.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    .. deprecated:: 0.8.2
       Use :py:mod:`freud.bond` instead.

    Args:
        rmax (float): distance over which to calculate
        k (unsigned int): number of neighbors to search
        compDotTol (float): value of the dot product below which a pair is
                            determined
    """
    cdef environment.Pairing2D * thisptr
    cdef rmax
    cdef num_neigh

    def __cinit__(self, rmax, k, compDotTol):
        warnings.warn("This class is deprecated, use freud.bond instead!", FreudDeprecationWarning)
        self.thisptr = new environment.Pairing2D(rmax, k, compDotTol)
        self.rmax = rmax
        self.num_neigh = k

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, orientations, compOrientations, nlist=None):
        """Calculates the correlation function and adds to the current
        histogram.

        Args:
          box (class:`freud.box:Box`): simulation box
          points (class:`numpy.ndarray`,
                  shape=(:math:`N_{particles}`, 3),
                  dtype= :class:`numpy.float32`): reference points to calculate the local density
          orientations (class:`numpy.ndarray`,
                        shape=(:math:`N_{particles}`, 4),
                        dtype= :class:`numpy.float32`): orientations to use in computation
          compOrientations (class:`numpy.ndarray`,
                            shape=(:math:`N_{particles}`, 4),
                            dtype= :class:`numpy.float32`): possible orientations to check
                                                            for bonds
          nlist (py:class:`freud.locality.NeighborList`): Neighborlist to use to
              find bonds (Default value = None)

        """
        box = freud.common.convert_box(box)
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        orientations = freud.common.convert_array(
                orientations, 1, dtype=np.float32, contiguous=True,
                array_name="orientations")

        compOrientations = freud.common.convert_array(
                compOrientations, 2, dtype=np.float32, contiguous=True,
                array_name="compOrientations")

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
        """The match."""
        return self.getMatch()

    def getMatch(self):
        """Get the match.

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{particles}\\right)`,
          dtype= :class:`numpy.uint32`: match

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
        """Pair."""
        return self.getPair()

    def getPair(self):
        """Get the pair.

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{particles}\\right)`,
          dtype= :class:`numpy.uint32`: pair

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
        """Get the box used in the calculation."""
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
          py:class:`freud.box.Box`: freud Box

        """
        return BoxFromCPP(< box.Box > self.thisptr.getBox())

cdef class AngularSeparation:
    """Calculates the minimum angles of separation between particles and
    references.

    .. moduleauthor:: Erin Teich
    .. moduleauthor:: Andrew Karas

    """
    cdef environment.AngularSeparation * thisptr
    cdef num_neigh
    cdef rmax
    cdef nlist_

    def __cinit__(self, rmax, n):
        self.thisptr = new environment.AngularSeparation()
        self.rmax = rmax
        self.num_neigh = int(n)
        self.nlist_ = None

    def __dealloc__(self):
        del self.thisptr

    @property
    def nlist(self):
        """ """
        return self.nlist_

    def computeNeighbor(self, box, ref_ors, ors, ref_points, points,
                        equiv_quats, nlist=None):
        """Calculates the minimum angles of separation between ref_ors and ors,
        checking for underlying symmetry as encoded in equiv_quats.

        Args:
          box (class:`freud.box:Box`): simulation box
          orientations (class:`numpy.ndarray`,
                       shape=(:math:`N_{particles}`, 3),
                       dtype= :class:`numpy.float32`): orientations to calculate
                       the order parameter
          ref_orientations (class:`numpy.ndarray`,
                           shape=(:math:`N_{particles}`, 4),
                           dtype= :class:`numpy.float32`): reference orientations to
                           calculate the order parameter
          ref_points (class:`numpy.ndarray`,
                      shape=(:math:`N_{particles}`, 3),
                      dtype= :class:`numpy.float32`): reference points to
                      calculate the order parameter
          points (class:`numpy.ndarray`,
                  shape=(:math:`N_{particles}`, 3),
                  dtype= :class:`numpy.float32`): points to calculate the
                  the order parameter
          mode (str): mode to calc bond order. "bod", "lbod", "obcd", and "oocd" (Default value = "bod")
          nlist(class:`freud.locality.NeighborList`): NeighborList to use to find bonds (Default value = None)
                                                      find bonds (Default value = None)
          equiv_quats (class:`numpy.ndarray`,
                       shape=(:math:`N_{particles}`, 4),
                       dtype= :class:`numpy.float32`): the set of all equivalent quaternions that takes the particle as it is defined to some global reference orientation. Important: equiv_quats must include both q and -q, for all included quaternions
          nlist (py:class:`freud.locality.NeighborList`): Neighborlist to use to find bonds
                                                          (Default value = None)

        """
        box = freud.common.convert_box(box)
        ref_points = freud.common.convert_array(
                ref_points, 2, dtype=np.float32, contiguous=True,
                array_name="ref_points")
        if ref_points.shape[1] != 3:
            raise TypeError('ref_points should be an Nx3 array')

        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        ref_ors = freud.common.convert_array(
                ref_ors, 2, dtype=np.float32, contiguous=True,
                array_name="ref_ors")
        if ref_ors.shape[1] != 4:
            raise TypeError('ref_ors should be an Nx4 array')

        ors = freud.common.convert_array(
                ors, 2, dtype=np.float32, contiguous=True,
                array_name="ors")
        if ors.shape[1] != 4:
            raise TypeError('ors should be an Nx4 array')

        equiv_quats = freud.common.convert_array(
                equiv_quats, 2, dtype=np.float32, contiguous=True,
                array_name="equiv_quats")
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

        Args:
          ors (class:`numpy.ndarray`,
               shape=(:math:`N_{particles}`, 3),
               dtype= :class:`numpy.float32`): orientations to calculate
               the order parameter
          global_ors (class:`numpy.ndarray`,
                      shape=(:math:`N_{particles}`, 4),
                      dtype= :class:`numpy.float32`): reference orientations to
                      calculate the order parameter
          equiv_quats (class:`numpy.ndarray`,
                       shape=(:math:`N_{particles}`, 4),
                       dtype= :class:`numpy.float32`): the set of all equivalent
                       quaternions that takes the particle as it is defined to
                       some global reference orientation. Important: equiv_quats
                       must include both q and -q, for all included quaternions

        """
        global_ors = freud.common.convert_array(
                global_ors, 2, dtype=np.float32, contiguous=True,
                array_name="global_ors")
        if global_ors.shape[1] != 4:
            raise TypeError('global_ors should be an Nx4 array')

        ors = freud.common.convert_array(
                ors, 2, dtype=np.float32, contiguous=True,
                array_name="ors")
        if ors.shape[1] != 4:
            raise TypeError('ors should be an Nx4 array')

        equiv_quats = freud.common.convert_array(
                equiv_quats, 2, dtype=np.float32, contiguous=True,
                array_name="equiv_quats")
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

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{reference}, N_{neighbors} \\right)`,
          dtype= :class:`numpy.float32`: angles in radians

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

        Returns:
          class:`numpy.ndarray`,
          shape= :math:`\\left(N_{particles}, N_{global} \\right)`,
          dtype= :class:`numpy.float32`: angles in radians

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

        Returns:
          unsigned int: math:`N_{particles}`

        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    def getNReference(self):
        """Get the number of reference particles used in computing the neighbor
        angles.

        Returns:
          unsigned int: math:`N_{particles}`

        """
        cdef unsigned int nref = self.thisptr.getNref()
        return nref

    def getNGlobal(self):
        """Get the number of global orientations to check against.

        Returns:
          unsigned int: math:`N_{global orientations}`

        """
        cdef unsigned int nglobal = self.thisptr.getNglobal()
        return nglobal
