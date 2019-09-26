# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.environment` module contains functions which characterize the
local environments of particles in the system. These methods use the positions
and orientations of particles in the local neighborhood of a given particle to
characterize the particle environment.
"""

import freud.common
import numpy as np
import warnings
import freud.locality

from freud.common cimport Compute
from freud.locality cimport PairCompute, SpatialHistogram
from freud.util cimport vec3, quat
from libcpp.vector cimport vector
from libcpp.map cimport map
from cython.operator cimport dereference
cimport freud.box
cimport freud._environment
cimport freud.locality
cimport freud.util

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class BondOrder(SpatialHistogram):
    R"""Compute the bond orientational order diagram for the system of
    particles.

    The bond orientational order diagram (BOOD) is a way of studying the
    average local environments experienced by particles. In a BOOD, a particle
    and its nearest neighbors (determined by either a prespecified number of
    neighbors or simply a cutoff distance) are treated as connected by a bond
    joining their centers. All of the bonds in the system are then binned by
    their azimuthal (:math:`\theta`) and polar (:math:`\phi`) angles to
    indicate the location of a particle's neighbors relative to itself. The
    distance between the particle and its neighbor is only important when
    determining whether it is counted as a neighbor, but is not part of the
    BOOD; as such, the BOOD can be viewed as a projection of all bonds onto the
    unit sphere. The resulting 2D histogram provides insight into how particles
    are situated relative to one-another in a system.

    This class provides access to the classical BOOD as well as a few useful
    variants. These variants can be accessed *via* the :code:`mode` arguments
    to the :meth:`~BondOrder.compute` or :meth:`~BondOrder.accumulate`
    methods. Available modes of calculation are:

    * :code:`'bod'` (Bond Order Diagram, *default*):
      This mode constructs the default BOOD, which is the 2D histogram
      containing the number of bonds formed through each azimuthal
      :math:`\left( \theta \right)` and polar :math:`\left( \phi \right)`
      angle.

    * :code:`'lbod'` (Local Bond Order Diagram):
      In this mode, a particle's neighbors are rotated into the local frame of
      the particle before the BOOD is calculated, *i.e.* the directions of
      bonds are determined relative to the orientation of the particle rather
      than relative to the global reference frame. An example of when this mode
      would be useful is when a system is composed of multiple grains of the
      same crystal; the normal BOOD would show twice as many peaks as expected,
      but using this mode, the bonds would be superimposed.

    * :code:`'obcd'` (Orientation Bond Correlation Diagram):
      This mode aims to quantify the degree of orientational as well as
      translational ordering. As a first step, the rotation that would align a
      particle's neighbor with the particle is calculated. Then, the neighbor
      is rotated **around the central particle** by that amount, which actually
      changes the direction of the bond. One example of how this mode could be
      useful is in identifying plastic crystals, which exhibit translational
      but not orientational ordering. Normally, the BOOD for a plastic crystal
      would exhibit clear structure since there is translational order, but
      with this mode, the neighbor positions would actually be modified,
      resulting in an isotropic (disordered) BOOD.

    * :code:`'oocd'` (Orientation Orientation Correlation Diagram):
      This mode is substantially different from the other modes. Rather than
      compute the histogram of neighbor bonds, this mode instead computes a
      histogram of the directors of neighboring particles, where the director
      is defined as the basis vector :math:`\hat{z}` rotated by the neighbor's
      quaternion. The directors are then rotated into the central particle's
      reference frame. This mode provides insight into the local orientational
      environment of particles, indicating, on average, how a particle's
      neighbors are oriented.

    Args:
        r_max (float):
            Distance over which to calculate.
        bins (unsigned int or sequence of length 2):
            If an unsigned int, the number of bins in :math:`\theta` and
            :math:`\phi`. If a sequence of two integers, interpreted as
            :code:`(num_bins_theta, num_bins_phi)`.

    Attributes:
        bond_order (:math:`\left(N_{\phi}, N_{\theta} \right)` :class:`numpy.ndarray`):
            Bond order.
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        theta (:math:`\left(N_{\theta} \right)` :class:`numpy.ndarray`):
            The values of bin centers for :math:`\theta`.
        phi (:math:`\left(N_{\phi} \right)` :class:`numpy.ndarray`):
            The values of bin centers for :math:`\phi`.
        n_bins_theta (unsigned int):
            The number of bins in the :math:`\theta` dimension.
        n_bins_phi (unsigned int):
            The number of bins in the :math:`\phi` dimension.

    """  # noqa: E501
    cdef freud._environment.BondOrder * thisptr

    def __cinit__(self, bins):
        try:
            n_bins_theta, n_bins_phi = bins
        except TypeError:
            n_bins_theta = n_bins_phi = bins

        self.thisptr = self.histptr = new freud._environment.BondOrder(
            n_bins_theta, n_bins_phi)

    def __dealloc__(self):
        del self.thisptr

    @property
    def default_query_args(self):
        raise NotImplementedError('No default query arguments for BondOrder.')

    @Compute._compute()
    def accumulate(self, box, points, orientations, query_points=None,
                   query_orientations=None, str mode="bod", neighbors=None):
        R"""Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate bonds.
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Reference orientations used to calculate bonds.
            query_points ((:math:`N_{query_points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate bonds. Uses :code:`points` if not
                provided or :code:`None`.
                (Default value = :code:`None`).
            query_orientations ((:math:`N_{query_points}`, 4) :class:`numpy.ndarray`, optional):
                Orientations used to calculate bonds. Uses
                :code:`orientations` if not provided or :code:`None`.
                (Default value = :code:`None`).
            mode (str, optional):
                Mode to calculate bond order. Options are :code:`'bod'`,
                :code:`'lbod'`, :code:`'obcd'`, or :code:`'oocd'`
                (Default value = :code:`'bod'`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef:
            freud.box.Box b
            freud.locality.NeighborQuery nq
            freud.locality.NlistptrWrapper nlistptr
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        b, nq, nlistptr, qargs, l_query_points, num_query_points = \
            self.preprocess_arguments_new(box, points, query_points, neighbors)
        if query_orientations is None:
            query_orientations = orientations

        orientations = freud.common.convert_array(
            orientations, shape=(nq.points.shape[0], 4))
        query_orientations = freud.common.convert_array(
            query_orientations, shape=(num_query_points, 4))

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

        cdef const float[:, ::1] l_orientations = orientations
        cdef const float[:, ::1] l_query_orientations = query_orientations

        self.thisptr.accumulate(
            nq.get_ptr(),
            <quat[float]*> &l_orientations[0, 0],
            <vec3[float]*> &l_query_points[0, 0],
            <quat[float]*> &l_query_orientations[0, 0],
            num_query_points,
            index, nlistptr.get_ptr(), dereference(qargs.thisptr))
        return self

    @Compute._computed_property()
    def bond_order(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getBondOrder(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property()
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    @Compute._reset
    def reset(self):
        R"""Resets the values of the bond order in memory."""
        self.thisptr.reset()

    @Compute._compute()
    def compute(self, box, points, orientations, query_points=None,
                query_orientations=None, mode="bod", neighbors=None):
        R"""Calculates the bond order histogram. Will overwrite the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate bonds.
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Reference orientations used to calculate bonds.
            query_points ((:math:`N_{query_points}`, 3) :class:`numpy.ndarray`, optional):
                query_points used to calculate bonds. Uses :code:`points` if not
                provided or :code:`None`.
                (Default value = :code:`None`).
            query_orientations ((:math:`N_{query_points}`, 4) :class:`numpy.ndarray`, optional):
                Orientations used to calculate bonds. Uses
                :code:`orientations` if not provided or :code:`None`.
                (Default value = :code:`None`).
            mode (str, optional):
                Mode to calculate bond order. Options are :code:`'bod'`,
                :code:`'lbod'`, :code:`'obcd'`, or :code:`'oocd'`
                (Default value = :code:`'bod'`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(box, points, orientations,
                        query_points, query_orientations, mode, neighbors)
        return self

    def __repr__(self):
        return ("freud.environment.{cls}( bins=({bins}))".format(
            cls=type(self).__name__,
            bins=', '.join([str(b) for b in self.nbins])))


cdef class LocalDescriptors(PairCompute):
    R"""Compute a set of descriptors (a numerical "fingerprint") of a particle's
    local environment.

    The resulting spherical harmonic array will be a complex-valued
    array of shape `(num_bonds, num_sphs)`. Spherical harmonic
    calculation can be restricted to some number of nearest neighbors
    through the `num_neighbors` argument; if a particle has more bonds
    than this number, the last one or more rows of bond spherical
    harmonics for each particle will not be set.

    Args:
        num_neighbors (unsigned int):
            Maximum number of neighbors to compute descriptors for.
        l_max (unsigned int):
            Maximum spherical harmonic :math:`l` to consider.
        r_max (float):
            Initial guess of the maximum radius to looks for neighbors.
        negative_m (bool, optional):
            True if we should also calculate :math:`Y_{lm}` for negative
            :math:`m`. (Default value = :code:`True`)

    Attributes:
        sph (:math:`\left(N_{bonds}, \text{SphWidth} \right)` :class:`numpy.ndarray`):
            A reference to the last computed spherical harmonic array.
        num_particles (unsigned int):
            The number of points passed to the last call to :meth:`~.compute`.
        num_sphs (unsigned int):
            The last number of spherical harmonics computed. This is equal to
            the number of bonds in the last computation, which is at most the
            number of `points` multiplied by the lower of the `num_neighbors`
            arguments passed to the last compute call or the constructor (it
            may be less if there are not enough neighbors for every particle).
        l_max (unsigned int):
            The maximum spherical harmonic :math:`l` to calculate for.
        r_max (float):
            The cutoff radius.
    """  # noqa: E501
    cdef freud._environment.LocalDescriptors * thisptr
    cdef num_neighbors
    cdef r_max
    cdef negative_m

    known_modes = {'neighborhood': freud._environment.LocalNeighborhood,
                   'global': freud._environment.Global,
                   'particle_local': freud._environment.ParticleLocal}

    def __cinit__(self, num_neighbors, l_max, r_max, negative_m=True):
        self.thisptr = new freud._environment.LocalDescriptors(
            l_max, negative_m)
        self.num_neighbors = num_neighbors
        self.r_max = r_max
        self.negative_m = negative_m

    def __dealloc__(self):
        del self.thisptr

    @Compute._compute()
    def compute(self, box, points,
                query_points=None, orientations=None, mode='neighborhood',
                neighbors=None):
        R"""Calculates the local descriptors of bonds from a set of source
        points to a set of destination points.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            num_neighbors (unsigned int):
                Number of nearest neighbors to compute with or to limit to, if the
                neighbor list is precomputed.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Source points to calculate the order parameter.
            query_points ((:math:`N_{query_points}`, 3) :class:`numpy.ndarray`, optional):
                Destination points to calculate the order parameter
                (Default value = :code:`None`).
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`, optional):
                Orientation of each point (Default value = :code:`None`).
            mode (str, optional):
                Orientation mode to use for environments, either
                :code:`'neighborhood'` to use the orientation of the local
                neighborhood, :code:`'particle_local'` to use the given
                particle orientations, or :code:`'global'` to not rotate
                environments (Default value = :code:`'neighborhood'`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = :code:`None`).
        """  # noqa: E501
        cdef:
            freud.box.Box b
            freud.locality.NeighborQuery nq
            freud.locality.NlistptrWrapper nlistptr
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        b, nq, nlistptr, qargs, l_query_points, num_query_points = \
            self.preprocess_arguments_new(box, points, query_points, neighbors)

        # The l_orientations_ptr is only used for 'particle_local' mode.
        cdef const float[:, ::1] l_orientations
        cdef quat[float] *l_orientations_ptr = NULL
        if mode == 'particle_local':
            if orientations is None:
                raise RuntimeError(
                    ('Orientations must be given to orient LocalDescriptors '
                        'with particles\' orientations'))

            orientations = freud.common.convert_array(
                orientations, shape=(points.shape[0], 4))

            l_orientations = orientations
            l_orientations_ptr = <quat[float]*> &l_orientations[0, 0]

        if mode not in self.known_modes:
            raise RuntimeError(
                'Unknown LocalDescriptors orientation mode: {}'.format(mode))
        cdef freud._environment.LocalDescriptorOrientation l_mode
        l_mode = self.known_modes[mode]

        self.thisptr.compute(
            nq.get_ptr(),
            <vec3[float]*> &l_query_points[0, 0], num_query_points,
            l_orientations_ptr, l_mode,
            nlistptr.get_ptr(), dereference(qargs.thisptr))
        return self

    @Compute._computed_property()
    def nlist(self):
        return freud.locality.nlist_from_cnlist(self.thisptr.getNList())

    @Compute._computed_property()
    def sph(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getSph(),
            freud.util.arr_type_t.COMPLEX_FLOAT)

    @Compute._computed_property()
    def num_particles(self):
        return self.thisptr.getNPoints()

    @Compute._computed_property()
    def num_sphs(self):
        return self.thisptr.getNSphs()

    @property
    def l_max(self):
        return self.thisptr.getLMax()

    def __repr__(self):
        return ("freud.environment.{cls}(num_neighbors={num_neighbors}, "
                "l_max={l_max}, r_max={r_max}, "
                "negative_m={negative_m})").format(
                    cls=type(self).__name__, num_neighbors=self.num_neighbors,
                    l_max=self.l_max, r_max=self.r_max,
                    negative_m=self.negative_m)


cdef class MatchEnv(Compute):
    R"""Clusters particles according to whether their local environments match
    or not, according to various shape matching metrics.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        r_max (float):
            Cutoff radius for cell list and clustering algorithm. Values near
            the first minimum of the RDF are recommended.
        num_neighbors (unsigned int):
            Number of nearest neighbors taken to define the local environment
            of any given particle.

    Attributes:
        tot_environment (:math:`\left(N_{particles}, N_{neighbors}, 3\right)` :class:`numpy.ndarray`):
            All environments for all particles.
        num_particles (unsigned int):
            The number of particles.
        num_clusters (unsigned int):
            The number of clusters.
        clusters (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The per-particle index indicating cluster membership.
    """  # noqa: E501
    cdef freud._environment.MatchEnv * thisptr
    cdef r_max
    cdef num_neighbors
    cdef m_box

    def __cinit__(self, box, r_max, num_neighbors):
        cdef freud.box.Box b = freud.common.convert_box(box)

        self.thisptr = new freud._environment.MatchEnv(
            dereference(b.thisptr), r_max, num_neighbors)

        self.r_max = r_max
        self.num_neighbors = num_neighbors
        self.m_box = box

    def __dealloc__(self):
        del self.thisptr

    @Compute._compute()
    def cluster(self, points, threshold, hard_r=False, registration=False,
                global_search=False, env_nlist=None, nlist=None):
        R"""Determine clusters of particles with matching environments.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Destination points to calculate the order parameter.
            threshold (float):
                Maximum magnitude of the vector difference between two vectors,
                below which they are "matching."
            hard_r (bool):
                If True, exclude all particles that fall beyond the threshold
                of :code:`r_max` from the environment.
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets.
                (Default value = :code:`False`)
            global_search (bool, optional):
                If True, do an exhaustive search wherein the environments of
                every single pair of particles in the simulation are compared.
                If False, only compare the environments of neighboring
                particles. (Default value = :code:`False`)
            env_nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find the environment of every particle
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find neighbors of every particle, to
                compare environments (Default value = :code:`None`).
        """
        points = freud.common.convert_array(points, shape=(None, 3))

        cdef const float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        cdef freud.locality.NeighborList nlist_
        cdef freud.locality.NeighborList env_nlist_
        if hard_r:
            nlist_ = freud.locality.make_default_nlist(
                self.m_box, points, None, dict(r_max=self.r_max), nlist)

            env_nlist_ = freud.locality.make_default_nlist(
                self.m_box, points, None, dict(r_max=self.r_max), env_nlist)
        else:
            nlist_ = freud.locality.make_default_nlist(
                self.m_box, points, None,
                dict(num_neighbors=self.num_neighbors, r_guess=self.r_max),
                nlist)

            env_nlist_ = freud.locality.make_default_nlist(
                self.m_box, points, None,
                dict(num_neighbors=self.num_neighbors, r_guess=self.r_max),
                env_nlist)

        self.thisptr.cluster(
            env_nlist_.get_ptr(), nlist_.get_ptr(),
            <vec3[float]*> &l_points[0, 0], nP, threshold,
            registration, global_search)
        return self

    @Compute._compute()
    def matchMotif(self, points, ref_points, threshold, registration=False,
                   nlist=None):
        R"""Determine clusters of particles that match the motif provided by
        ref_points.

        Args:
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up the motif against which we are matching.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Particle positions.
            threshold (float):
                Maximum magnitude of the vector difference between two vectors,
                below which they are considered "matching."
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets
                (Default value = False).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """
        points = freud.common.convert_array(points, shape=(None, 3))
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))

        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(
            points.flatten())
        cdef np.ndarray[float, ndim=1] l_ref_points = np.ascontiguousarray(
            ref_points.flatten())
        cdef unsigned int nP = l_points.shape[0]
        cdef unsigned int nRef = l_ref_points.shape[0]

        cdef freud.locality.NeighborList nlist_
        nlist_ = freud.locality.make_default_nlist(
            self.m_box, points, None,
            dict(num_neighbors=self.num_neighbors, r_guess=self.r_max), nlist)

        self.thisptr.matchMotif(
            nlist_.get_ptr(), <vec3[float]*> &l_points[0], nP,
            <vec3[float]*> &l_ref_points[0], nRef, threshold,
            registration)

    @Compute._compute()
    def minRMSDMotif(self, ref_points, points, registration=False, nlist=None):
        R"""Rotate (if registration=True) and permute the environments of all
        particles to minimize their RMSD with respect to the motif provided by
        ref_points.

        Args:
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up the motif against which we are matching.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Particle positions.
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets
                (Default value = :code:`False`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        Returns:
            :math:`\left(N_{particles}\right)` :class:`numpy.ndarray`:
                Vector of minimal RMSD values, one value per particle.

        """
        points = freud.common.convert_array(points, shape=(None, 3))
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))

        cdef np.ndarray[float, ndim=1] l_points = np.ascontiguousarray(
            points.flatten())
        cdef np.ndarray[float, ndim=1] l_ref_points = np.ascontiguousarray(
            ref_points.flatten())
        cdef unsigned int nP = l_points.shape[0]
        cdef unsigned int nRef = l_ref_points.shape[0]

        cdef freud.locality.NeighborList nlist_
        nlist_ = freud.locality.make_default_nlist(
            self.m_box, points, None,
            dict(num_neighbors=self.num_neighbors, r_guess=self.r_max), nlist)

        cdef vector[float] min_rmsd_vec = self.thisptr.minRMSDMotif(
            nlist_.get_ptr(), <vec3[float]*> &l_points[0], nP,
            <vec3[float]*> &l_ref_points[0], nRef, registration)

        return min_rmsd_vec

    def isSimilar(self, ref_points, points,
                  threshold, registration=False):
        R"""Test if the motif provided by ref_points is similar to the motif
        provided by points.

        Args:
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up motif 1.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up motif 2.
            threshold (float):
                Maximum magnitude of the vector difference between two vectors,
                below which they are considered "matching."
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets
                (Default value = :code:`False`).

        Returns:
            tuple ((:math:`\left(N_{particles}, 3\right)` :class:`numpy.ndarray`), map[int, int]):
                A doublet that gives the rotated (or not) set of
                :code:`points`, and the mapping between the vectors of
                :code:`ref_points` and :code:`points` that will make them
                correspond to each other. Empty if they do not correspond to
                each other.
        """  # noqa: E501
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))
        points = freud.common.convert_array(points, shape=(None, 3))

        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points = points
        cdef unsigned int nRef1 = l_ref_points.shape[0]
        cdef unsigned int nRef2 = l_points.shape[0]
        cdef float threshold_sq = threshold*threshold

        if nRef1 != nRef2:
            raise ValueError(
                ("The number of vectors in ref_points must MATCH"
                 "the number of vectors in points"))

        cdef map[unsigned int, unsigned int] vec_map = self.thisptr.isSimilar(
            <vec3[float]*> &l_ref_points[0, 0],
            <vec3[float]*> &l_points[0, 0],
            nRef1, threshold_sq, registration)
        return [np.asarray(l_points), vec_map]

    def minimizeRMSD(self, ref_points, points, registration=False):
        R"""Get the somewhat-optimal RMSD between the set of vectors ref_points
        and the set of vectors points.

        Args:
            ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up motif 1.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up motif 2.
            registration (bool, optional):
                If true, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets
                (Default value = :code:`False`).

        Returns:
            tuple (float, (:math:`\left(N_{particles}, 3\right)` :class:`numpy.ndarray`), map[int, int]):
                A triplet that gives the associated min_rmsd, rotated (or not)
                set of points, and the mapping between the vectors of
                ref_points and points that somewhat minimizes the RMSD.
        """  # noqa: E501
        ref_points = freud.common.convert_array(ref_points, shape=(None, 3))
        points = freud.common.convert_array(points, shape=(None, 3))

        cdef const float[:, ::1] l_ref_points = ref_points
        cdef const float[:, ::1] l_points = points
        cdef unsigned int nRef1 = l_ref_points.shape[0]
        cdef unsigned int nRef2 = l_points.shape[0]

        if nRef1 != nRef2:
            raise ValueError(
                ("The number of vectors in ref_points must MATCH"
                 "the number of vectors in points"))

        cdef float min_rmsd = -1
        cdef map[unsigned int, unsigned int] results_map = \
            self.thisptr.minimizeRMSD(
                <vec3[float]*> &l_ref_points[0, 0],
                <vec3[float]*> &l_points[0, 0],
                nRef1, min_rmsd, registration)
        return [min_rmsd, np.asarray(l_points), results_map]

    @Compute._computed_property()
    def clusters(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusters(),
            freud.util.arr_type_t.UNSIGNED_INT)

    @Compute._computed_method()
    def getEnvironment(self, i):
        R"""Returns the set of vectors defining the environment indexed by i.

        Args:
            i (unsigned int): Environment index.

        Returns:
            :math:`\left(N_{neighbors}, 3\right)` :class:`numpy.ndarray`:
            The array of vectors.
        """
        env = self.thisptr.getEnvironment(i)
        return np.asarray([[p.x, p.y, p.z] for p in env])

    @Compute._computed_property()
    def tot_environment(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getTotEnvironment(),
            freud.util.arr_type_t.FLOAT, 3)

    @Compute._computed_property()
    def num_particles(self):
        return self.thisptr.getNP()

    @Compute._computed_property()
    def num_clusters(self):
        return self.thisptr.getNumClusters()

    def __repr__(self):
        return ("freud.environment.{cls}(box={box}, "
                "r_max={r_max}, num_neighbors={num_neighbors})").format(
                    cls=type(self).__name__, box=self.m_box.__repr__(),
                    r_max=self.r_max, num_neighbors=self.num_neighbors)

    @Compute._computed_method()
    def plot(self, ax=None):
        """Plot cluster distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        try:
            values, counts = np.unique(self.clusters, return_counts=True)
        except ValueError:
            return None
        else:
            return freud.plot.clusters_plot(
                values, counts, num_clusters_to_plot=10, ax=ax)

    def _repr_png_(self):
        import freud.plot
        try:
            return freud.plot.ax_to_bytes(self.plot())
        except AttributeError:
            return None


cdef class AngularSeparationNeighbor(PairCompute):
    R"""Calculates the minimum angles of separation between particles and
    references.

    Args:
        r_max (float):
            Cutoff radius for cell list and clustering algorithm. Values near
            the first minimum of the RDF are recommended.
        num_neighbors (int):
            The number of neighbors.

    Attributes:
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list.
        neighbor_angles (:math:`\left(N_{bonds}\right)` :class:`numpy.ndarray`):
            The neighbor angles in radians. **This field is only populated
            after** :meth:`~.computeNeighbor` **is called.** The angles
            are stored in the order of the neighborlist object.
        global_angles (:math:`\left(N_{particles}, N_{global} \right)` :class:`numpy.ndarray`):
            The global angles in radians. **This field is only populated
            after** :meth:`~.computeGlobal` **is called.** The angles
            are stored in the order of the neighborlist object.
    """  # noqa: E501
    cdef freud._environment.AngularSeparationNeighbor * thisptr

    def __cinit__(self):
        self.thisptr = new freud._environment.AngularSeparationNeighbor()

    def __dealloc__(self):
        del self.thisptr

    @Compute._compute()
    def compute(self, box, points, orientations, query_points=None,
                query_orientations=None,
                equiv_orientations=np.array([[1, 0, 0, 0]]),
                neighbors=None):
        R"""Calculates the minimum angles of separation between :code:`orientations`
        and :code:`query_orientations`, checking for underlying symmetry as encoded
        in :code:`equiv_orientations`. The result is stored in the :code:`neighbor_angles`
        class attribute.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate the order parameter.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Reference orientations used to calculate the order parameter.
            query_points ((:math:`N_{query_points}`, 3) :class:`numpy.ndarray`):
                Points used to calculate the order parameter. Uses :code:`points`
                if not provided or :code:`None`.
            query_orientations ((:math:`N_{query_points}`, 4) :class:`numpy.ndarray`):
                query_orientations used to calculate the order parameter. Uses :code:`orientations`
                if not provided or :code:`None`.
            equiv_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`, optional):
                The set of all equivalent quaternions that takes the particle
                as it is defined to some global reference orientation.
                Important: :code:`equiv_orientations` must include both :math:`q` and
                :math:`-q`, for all included quaternions.
                (Default value = :code:`[[1, 0, 0, 0]]`)
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef:
            freud.box.Box b
            freud.locality.NeighborQuery nq
            freud.locality.NlistptrWrapper nlistptr
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        b, nq, nlistptr, qargs, l_query_points, num_query_points = \
            self.preprocess_arguments_new(box, points, query_points, neighbors)

        orientations = freud.common.convert_array(
            orientations, shape=(points.shape[0], 4))
        if query_orientations is None:
            query_orientations = orientations
        else:
            query_orientations = freud.common.convert_array(
                query_orientations, shape=(query_points.shape[0], 4))

        equiv_orientations = freud.common.convert_array(
            equiv_orientations, shape=(None, 4))

        cdef const float[:, ::1] l_orientations = orientations
        cdef const float[:, ::1] l_query_orientations = query_orientations
        cdef const float[:, ::1] l_equiv_orientations = equiv_orientations

        cdef unsigned int n_equiv_orientations = l_equiv_orientations.shape[0]

        self.thisptr.compute(
            nq.get_ptr(),
            <quat[float]*> &l_orientations[0, 0],
            <vec3[float]*> &l_query_points[0, 0],
            <quat[float]*> &l_query_orientations[0, 0],
            num_query_points,
            <quat[float]*> &l_equiv_orientations[0, 0],
            n_equiv_orientations,
            nlistptr.get_ptr(),
            dereference(qargs.thisptr))
        return self

    @Compute._computed_property()
    def angles(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getAngles(),
            freud.util.arr_type_t.FLOAT)

    def __repr__(self):
        return "freud.environment.{cls}()".format(
            cls=type(self).__name__)

    @Compute._computed_property()
    def nlist(self):
        return freud.locality.nlist_from_cnlist(self.thisptr.getNList())


cdef class AngularSeparationGlobal(Compute):
    R"""Calculates the minimum angles of separation between particles and
    references.

    Args:
        r_max (float):
            Cutoff radius for cell list and clustering algorithm. Values near
            the first minimum of the RDF are recommended.
        num_neighbors (int):
            The number of neighbors.

    Attributes:
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list.
        neighbor_angles (:math:`\left(N_{bonds}\right)` :class:`numpy.ndarray`):
            The neighbor angles in radians. **This field is only populated
            after** :meth:`~.computeNeighbor` **is called.** The angles
            are stored in the order of the neighborlist object.
        global_angles (:math:`\left(N_{particles}, N_{global} \right)` :class:`numpy.ndarray`):
            The global angles in radians. **This field is only populated
            after** :meth:`~.computeGlobal` **is called.** The angles
            are stored in the order of the neighborlist object.
    """  # noqa: E501
    cdef freud._environment.AngularSeparationGlobal * thisptr

    def __cinit__(self):
        self.thisptr = new freud._environment.AngularSeparationGlobal()

    def __dealloc__(self):
        del self.thisptr

    @Compute._compute()
    def compute(self, global_orientations,
                orientations, equiv_orientations):
        R"""Calculates the minimum angles of separation between
        :code:`global_orientations` and :code:`orientations`, checking for underlying symmetry as
        encoded in :code:`equiv_orientations`. The result is stored in the
        :code:`global_angles` class attribute.


        Args:
            global_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Reference orientations to calculate the order parameter.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Orientations to calculate the order parameter.
            equiv_orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                The set of all equivalent quaternions that takes the particle
                as it is defined to some global reference orientation.
                Important: :code:`equiv_orientations` must include both :math:`q` and
                :math:`-q`, for all included quaternions.
        """  # noqa
        global_orientations = freud.common.convert_array(
            global_orientations, shape=(None, 4))
        orientations = freud.common.convert_array(
            orientations, shape=(None, 4))
        equiv_orientations = freud.common.convert_array(
            equiv_orientations, shape=(None, 4))

        cdef const float[:, ::1] l_global_orientations = global_orientations
        cdef const float[:, ::1] l_orientations = orientations
        cdef const float[:, ::1] l_equiv_orientations = equiv_orientations

        cdef unsigned int n_global = l_global_orientations.shape[0]
        cdef unsigned int n_points = l_orientations.shape[0]
        cdef unsigned int n_equiv_orientations = l_equiv_orientations.shape[0]

        self.thisptr.compute(
            <quat[float]*> &l_global_orientations[0, 0],
            n_global,
            <quat[float]*> &l_orientations[0, 0],
            n_points,
            <quat[float]*> &l_equiv_orientations[0, 0],
            n_equiv_orientations)
        return self

    @Compute._computed_property()
    def angles(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getAngles(),
            freud.util.arr_type_t.FLOAT)

    def __repr__(self):
        return "freud.environment.{cls}()".format(
            cls=type(self).__name__)


cdef class LocalBondProjection(Compute):
    R"""Calculates the maximal projection of nearest neighbor bonds for each
    particle onto some set of reference vectors, defined in the particles'
    local reference frame.

    Args:
        r_max (float):
            Cutoff radius.
        num_neighbors (unsigned int):
            The number of neighbors.

    Attributes:
        projections ((:math:`\left(N_{reference}, N_{neighbors}, N_{projection\_vecs} \right)` :class:`numpy.ndarray`):
            The projection of each bond between reference particles and their
            neighbors onto each of the projection vectors.
        normed_projections ((:math:`\left(N_{reference}, N_{neighbors}, N_{projection\_vecs} \right)` :class:`numpy.ndarray`)
            The normalized projection of each bond between reference particles
            and their neighbors onto each of the projection vectors.
        num_reference_particles (int):
            The number of reference points used in the last calculation.
        num_particles (int):
            The number of points used in the last calculation.
        num_proj_vectors (int):
            The number of projection vectors used in the last calculation.
        box (:class:`freud.box.Box`):
            The box used in the last calculation.
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list generated in the last calculation.
    """  # noqa: E501
    cdef freud._environment.LocalBondProjection * thisptr
    cdef float r_max
    cdef unsigned int num_neighbors
    cdef freud.locality.NeighborList nlist_

    def __cinit__(self, r_max, num_neighbors):
        self.thisptr = new freud._environment.LocalBondProjection()
        self.r_max = r_max
        self.num_neighbors = int(num_neighbors)

    def __dealloc__(self):
        del self.thisptr

    @Compute._computed_property()
    def nlist(self):
        return self.nlist_

    @Compute._compute()
    def compute(self, box, proj_vecs, points,
                orientations, query_points=None,
                equiv_orientations=np.array([[1, 0, 0, 0]]), nlist=None):
        R"""Calculates the maximal projections of nearest neighbor bonds
        (between :code:`points` and :code:`query_points`) onto the set of
        reference vectors :code:`proj_vecs`, defined in the local reference
        frames of the :code:`points` as defined by the orientations
        :code:`orientations`. This computation accounts for the underlying
        symmetries of the reference frame as encoded in :code:`equiv_orientations`.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            proj_vecs ((:math:`N_{vectors}`, 3) :class:`numpy.ndarray`):
                The set of reference vectors, defined in the reference
                particles' reference frame, to calculate maximal local bond
                projections onto.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Reference points used in the calculation.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Reference orientations used in the calculation.
            query_points ((:math:`N_{query_points}`, 3) :class:`numpy.ndarray`, optional):
                Points (neighbors of :code:`points`) used in the
                calculation. Uses :code:`points` if not provided or
                :code:`None`.
                (Default value = :code:`None`).
            equiv_orientations ((:math:`N_{quats}`, 4) :class:`numpy.ndarray`, optional):
                The set of all equivalent quaternions that takes the particle
                as it is defined to some global reference orientation. Note
                that this does not need to include both :math:`q` and
                :math:`-q`, since :math:`q` and :math:`-q` effect the same
                rotation on vectors. (Default value = :code:`[1, 0, 0, 0]`)
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef freud.box.Box b = freud.common.convert_box(box)
        points = freud.common.convert_array(points, shape=(None, 3))
        orientations = freud.common.convert_array(
            orientations, shape=(None, 4))

        self.nlist_ = freud.locality.make_default_nlist(
            b, points, query_points,
            dict(num_neighbors=self.num_neighbors,
                 r_guess=self.r_max), nlist).copy()

        if query_points is None:
            query_points = points
        else:
            query_points = freud.common.convert_array(
                query_points, shape=(None, 3))
        equiv_orientations = freud.common.convert_array(
            equiv_orientations, shape=(None, 4))
        proj_vecs = freud.common.convert_array(proj_vecs, shape=(None, 3))

        cdef const float[:, ::1] l_points = points
        cdef const float[:, ::1] l_orientations = orientations
        cdef const float[:, ::1] l_query_points = query_points
        cdef const float[:, ::1] l_equiv_orientations = equiv_orientations
        cdef const float[:, ::1] l_proj_vecs = proj_vecs

        cdef unsigned int n_points = l_points.shape[0]
        cdef unsigned int n_query_points = l_query_points.shape[0]
        cdef unsigned int n_equiv = l_equiv_orientations.shape[0]
        cdef unsigned int n_proj = l_proj_vecs.shape[0]

        self.thisptr.compute(
            dereference(b.thisptr),
            <vec3[float]*> &l_proj_vecs[0, 0], n_proj,
            <vec3[float]*> &l_points[0, 0],
            <quat[float]*> &l_orientations[0, 0], n_points,
            <vec3[float]*> &l_query_points[0, 0], n_query_points,
            <quat[float]*> &l_equiv_orientations[0, 0], n_equiv,
            self.nlist_.get_ptr())
        return self

    @Compute._computed_property()
    def projections(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getProjections(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property()
    def normed_projections(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getNormedProjections(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property()
    def num_points(self):
        return self.thisptr.getNPoints()

    @Compute._computed_property()
    def num_query_points(self):
        return self.thisptr.getNQueryPoints()

    @Compute._computed_property()
    def num_proj_vectors(self):
        return self.thisptr.getNproj()

    @Compute._computed_property()
    def box(self):
        return freud.box.BoxFromCPP(<freud._box.Box> self.thisptr.getBox())

    def __repr__(self):
        return ("freud.environment.{cls}(r_max={r_max}, "
                "num_neighbors={num_neighbors})").format(
                    cls=type(self).__name__, r_max=self.r_max,
                    num_neighbors=self.num_neighbors)
