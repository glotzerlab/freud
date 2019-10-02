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

from freud.util cimport Compute
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
        bins (unsigned int or sequence of length 2):
            If an unsigned int, the number of bins in :math:`\theta` and
            :math:`\phi`. If a sequence of two integers, interpreted as
            :code:`(num_bins_theta, num_bins_phi)`.
        mode (str, optional):
            Mode to calculate bond order. Options are :code:`'bod'`,
            :code:`'lbod'`, :code:`'obcd'`, or :code:`'oocd'`
            (Default value = :code:`'bod'`).

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

    known_modes = {'bod': freud._environment.bod,
                   'lbod': freud._environment.lbod,
                   'obcd': freud._environment.obcd,
                   'oocd': freud._environment.oocd}

    def __cinit__(self, bins, str mode="bod"):
        try:
            n_bins_theta, n_bins_phi = bins
        except TypeError:
            n_bins_theta = n_bins_phi = bins

        cdef freud._environment.BondOrderMode l_mode
        try:
            l_mode = self.known_modes[mode]
        except KeyError:
            raise ValueError(
                'Unknown BondOrder mode: {}'.format(mode))

        self.thisptr = self.histptr = new freud._environment.BondOrder(
            n_bins_theta, n_bins_phi, l_mode)

    def __dealloc__(self):
        del self.thisptr

    @property
    def default_query_args(self):
        raise NotImplementedError('No default query arguments for BondOrder.')

    def accumulate(self, neighbor_query, orientations, query_points=None,
                   query_orientations=None, neighbors=None):
        R"""Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate bonds.
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Reference orientations used to calculate bonds.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Points used to calculate bonds. Uses :code:`points` if not
                provided or :code:`None`.
                (Default value = :code:`None`).
            query_orientations ((:math:`N_{query\_points}`, 4) :class:`numpy.ndarray`, optional):
                Orientations used to calculate bonds. Uses
                :code:`orientations` if not provided or :code:`None`.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, query_points, neighbors)
        if query_orientations is None:
            query_orientations = orientations

        orientations = freud.util._convert_array(
            orientations, shape=(nq.points.shape[0], 4))
        query_orientations = freud.util._convert_array(
            query_orientations, shape=(num_query_points, 4))

        cdef const float[:, ::1] l_orientations = orientations
        cdef const float[:, ::1] l_query_orientations = query_orientations

        self.thisptr.accumulate(
            nq.get_ptr(),
            <quat[float]*> &l_orientations[0, 0],
            <vec3[float]*> &l_query_points[0, 0],
            <quat[float]*> &l_query_orientations[0, 0],
            num_query_points,
            nlist.get_ptr(), dereference(qargs.thisptr))
        return self

    @Compute._computed_property
    def bond_order(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getBondOrder(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property
    def box(self):
        return freud.box.BoxFromCPP(self.thisptr.getBox())

    def reset(self):
        R"""Resets the values of the bond order in memory."""
        self.thisptr.reset()

    def compute(self, neighbor_query, orientations, query_points=None,
                query_orientations=None, neighbors=None):
        R"""Calculates the bond order histogram. Will overwrite the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Reference points used to calculate bonds.
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Reference orientations used to calculate bonds.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                query_points used to calculate bonds. Uses :code:`points` if not
                provided or :code:`None`.
                (Default value = :code:`None`).
            query_orientations ((:math:`N_{query\_points}`, 4) :class:`numpy.ndarray`, optional):
                Orientations used to calculate bonds. Uses
                :code:`orientations` if not provided or :code:`None`.
                (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
        """  # noqa: E501
        self.reset()
        self.accumulate(neighbor_query, orientations,
                        query_points, query_orientations, neighbors)
        return self

    def __repr__(self):
        return ("freud.environment.{cls}(bins=({bins}), mode='{mode}')".format(
            cls=type(self).__name__,
            bins=', '.join([str(b) for b in self.nbins]),
            mode=self.mode))

    @property
    def mode(self):
        mode = self.thisptr.getMode()
        for key, value in self.known_modes.items():
            if value == mode:
                return key


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
        l_max (unsigned int):
            Maximum spherical harmonic :math:`l` to consider.
        negative_m (bool, optional):
            True if we should also calculate :math:`Y_{lm}` for negative
            :math:`m`. (Default value = :code:`True`)
        mode (str, optional):
            Orientation mode to use for environments, either
            :code:`'neighborhood'` to use the orientation of the local
            neighborhood, :code:`'particle_local'` to use the given
            particle orientations, or :code:`'global'` to not rotate
            environments (Default value = :code:`'neighborhood'`).

    Attributes:
        sph (:math:`\left(N_{bonds}, \text{SphWidth} \right)` :class:`numpy.ndarray`):
            A reference to the last computed spherical harmonic array.
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
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list from the last compute, which is generated if not
            provided as the :code:`neighbors` argument.
    """  # noqa: E501
    cdef freud._environment.LocalDescriptors * thisptr

    known_modes = {'neighborhood': freud._environment.LocalNeighborhood,
                   'global': freud._environment.Global,
                   'particle_local': freud._environment.ParticleLocal}

    def __cinit__(self, l_max, negative_m=True, mode='neighborhood'):
        cdef freud._environment.LocalDescriptorOrientation l_mode
        try:
            l_mode = self.known_modes[mode]
        except KeyError:
            raise ValueError(
                'Unknown LocalDescriptors orientation mode: {}'.format(mode))

        self.thisptr = new freud._environment.LocalDescriptors(
            l_max, negative_m, l_mode)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, neighbor_query, query_points=None, orientations=None,
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
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Destination points to calculate the order parameter
                (Default value = :code:`None`).
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`, optional):
                Orientation of each point (Default value = :code:`None`).
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = :code:`None`).
        """  # noqa: E501
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, query_points, neighbors)

        # The l_orientations_ptr is only used for 'particle_local' mode.
        cdef const float[:, ::1] l_orientations
        cdef quat[float] *l_orientations_ptr = NULL
        if self.mode == 'particle_local':
            if orientations is None:
                raise RuntimeError(
                    ('Orientations must be given to orient LocalDescriptors '
                        'with particles\' orientations'))

            orientations = freud.util._convert_array(
                orientations, shape=(nq.points.shape[0], 4))

            l_orientations = orientations
            l_orientations_ptr = <quat[float]*> &l_orientations[0, 0]

        self.thisptr.compute(
            nq.get_ptr(),
            <vec3[float]*> &l_query_points[0, 0], num_query_points,
            l_orientations_ptr,
            nlist.get_ptr(), dereference(qargs.thisptr))
        return self

    @Compute._computed_property
    def nlist(self):
        return freud.locality._nlist_from_cnlist(self.thisptr.getNList())

    @Compute._computed_property
    def sph(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getSph(),
            freud.util.arr_type_t.COMPLEX_FLOAT)

    @Compute._computed_property
    def num_sphs(self):
        return self.thisptr.getNSphs()

    @property
    def l_max(self):
        return self.thisptr.getLMax()

    @property
    def negative_m(self):
        return self.thisptr.getNegativeM()

    @property
    def mode(self):
        mode = self.thisptr.getMode()
        for key, value in self.known_modes.items():
            if value == mode:
                return key

    def __repr__(self):
        return ("freud.environment.{cls}(l_max={l_max}, "
                "negative_m={negative_m}, mode='{mode}')").format(
                    cls=type(self).__name__, l_max=self.l_max,
                    negative_m=self.negative_m, mode=self.mode)


def _minimizeRMSD(box, ref_points, points, registration=False):
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
    cdef freud.box.Box b = freud.util._convert_box(box)

    ref_points = freud.util._convert_array(ref_points, shape=(None, 3))
    points = freud.util._convert_array(points, shape=(None, 3))

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
        freud._environment.minimizeRMSD(
            dereference(b.thisptr),
            <vec3[float]*> &l_ref_points[0, 0],
            <vec3[float]*> &l_points[0, 0],
            nRef1, min_rmsd, registration)
    return [min_rmsd, np.asarray(l_points), results_map]


def _isSimilar(box, ref_points, points, threshold, registration=False):
    R"""Test if the motif provided by ref_points is similar to the motif
    provided by points.

    Args:
        ref_points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
            Vectors that make up motif 1.
        points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
            Vectors that make up motif 2.
        threshold (float):
            Maximum magnitude of the vector difference between two vectors,
            below which they are "matching". Typically, a good choice is
            between 10% and 30% of the first well in the radial
            distribution function (this has distance units).
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
    cdef freud.box.Box b = freud.util._convert_box(box)

    ref_points = freud.util._convert_array(ref_points, shape=(None, 3))
    points = freud.util._convert_array(points, shape=(None, 3))

    cdef const float[:, ::1] l_ref_points = ref_points
    cdef const float[:, ::1] l_points = points
    cdef unsigned int nRef1 = l_ref_points.shape[0]
    cdef unsigned int nRef2 = l_points.shape[0]
    cdef float threshold_sq = threshold*threshold

    if nRef1 != nRef2:
        raise ValueError(
            ("The number of vectors in ref_points must MATCH"
                "the number of vectors in points"))

    cdef map[unsigned int, unsigned int] vec_map = \
        freud._environment.isSimilar(
            dereference(b.thisptr), <vec3[float]*> &l_ref_points[0, 0],
            <vec3[float]*> &l_points[0, 0], nRef1, threshold_sq,
            registration)
    return [np.asarray(l_points), vec_map]


cdef class _MatchEnv(PairCompute):
    R"""Parent for environment matching methods.

    Attributes:
        point_environments (:math:`\left(N_{points}, N_{neighbors}, 3\right)` :class:`numpy.ndarray`):
            All environments for all points.
    """  # noqa: E501
    cdef freud._environment.MatchEnv * matchptr

    def __cinit__(self, *args, **kwargs):
        # Abstract class
        pass

    @Compute._computed_property
    def point_environments(self):
        return freud.util.make_managed_numpy_array(
            &self.matchptr.getPointEnvironments(),
            freud.util.arr_type_t.FLOAT, 3)

    def __repr__(self):
        return ("freud.environment.{cls}()").format(
            cls=type(self).__name__)


cdef class EnvironmentCluster(_MatchEnv):
    R"""Clusters particles according to whether their local environments match
    or not, according to various shape matching metrics.

    Attributes:
        point_environments (:math:`\left(N_{points}, N_{neighbors}, 3\right)` :class:`numpy.ndarray`):
            All environments for all points.
        num_clusters (unsigned int):
            The number of clusters.
        clusters (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The per-particle index indicating cluster membership.
        cluster_environments (:math:`\left(N_{clusters}, N_{neighbors}, 3\right)` :class:`numpy.ndarray`):
            The environments for all clusters.
    """  # noqa: E501

    cdef freud._environment.EnvironmentCluster * thisptr

    def __cinit__(self):
        self.thisptr = self.matchptr = \
            new freud._environment.EnvironmentCluster()

    def __dealloc__(self):
        del self.thisptr

    def compute(self, neighbor_query, threshold, neighbors=None,
                env_neighbors=None, registration=False,
                global_search=False):
        R"""Determine clusters of particles with matching environments.

        In general, it is recommended to specify a number of neighbors rather
        than just a distance cutoff as part of your neighbor querying when
        performing this computation. Using a distance cutoff alone could easily
        lead to situations where a point doesn't match a cluster because a
        required neighbor is just outside the cutoff.

        Args:
            neighbor_query ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Destination points to calculate the order parameter.
            threshold (float):
                Maximum magnitude of the vector difference between two vectors,
                below which they are "matching". Typically, a good choice is
                between 10% and 30% of the first well in the radial
                distribution function (this has distance units).
            neighbors (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find neighbors of every particle, to
                compare environments (Default value = :code:`None`).
            env_neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                NeighborList to use to find the environment of every particle
                (Default value = :code:`None`).
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
        """  # noqa: E501
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist, env_nlist
            freud.locality._QueryArgs qargs, env_qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, neighbors=neighbors)

        if env_neighbors is None:
            env_neighbors = neighbors
        env_nlist, env_qargs = self._resolve_neighbors(env_neighbors)

        self.thisptr.compute(
            nq.get_ptr(), nlist.get_ptr(), dereference(qargs.thisptr),
            env_nlist.get_ptr(), dereference(env_qargs.thisptr), threshold,
            registration, global_search)
        return self

    @Compute._computed_property
    def clusters(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusters(),
            freud.util.arr_type_t.UNSIGNED_INT)

    @Compute._computed_property
    def num_clusters(self):
        return self.thisptr.getNumClusters()

    @Compute._computed_property
    def cluster_environments(self):
        envs = self.thisptr.getClusterEnvironments()
        return [np.asarray([[p.x, p.y, p.z] for p in env])
                for env in envs]

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


cdef class EnvironmentMotifMatch(_MatchEnv):
    R"""Find matches between local arrangements of a set of points and a provided motif.

    In general, it is recommended to specify a number of neighbors rather than
    just a distance cutoff as part of your neighbor querying when performing
    this computation since it can otherwise be very sensitive. Specifically, it
    is highly recommended that you choose a number of neighbors that you
    specify a number of neighbors query that requests at least as many
    neighbors as the size of the motif you intend to test against. Otherwise,
    you will struggle to match the motif. However, this is not currently
    enforced.

    Attributes:
        matches (:math:`(N_p, )` :class:`numpy.ndarray`):
            A boolean array indicating whether each point matches the motif.
        point_environments (:math:`\left(N_{points}, N_{neighbors}, 3\right)` :class:`numpy.ndarray`):
            All environments for all points.
    """  # noqa: E501

    cdef freud._environment.EnvironmentMotifMatch * thisptr

    def __cinit__(self):
        self.thisptr = self.matchptr = \
            new freud._environment.EnvironmentMotifMatch()

    def compute(self, neighbor_query, motif, threshold, neighbors=None,
                registration=False):
        R"""Determine clusters of particles that match the motif provided by
        motif.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                NeighborQuery.
            motif ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up the motif against which we are matching.
            threshold (float):
                Maximum magnitude of the vector difference between two vectors,
                below which they are "matching". Typically, a good choice is
                between 10% and 30% of the first well in the radial
                distribution function (this has distance units).
            neighbors (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets
                (Default value = False).
        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, neighbors=neighbors)

        motif = freud.util._convert_array(motif, shape=(None, 3))
        cdef const float[:, ::1] l_motif = motif
        cdef unsigned int nRef = l_motif.shape[0]

        self.thisptr.compute(
            nq.get_ptr(), nlist.get_ptr(), dereference(qargs.thisptr),
            <vec3[float]*>
            <vec3[float]*> &l_motif[0, 0], nRef,
            threshold, registration)

    @Compute._computed_property
    def matches(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getMatches(),
            freud.util.arr_type_t.BOOL)


cdef class _EnvironmentRMSDMinimizer(_MatchEnv):
    R"""Find linear transformations that map the environments of points onto a motif.

    In general, it is recommended to specify a number of neighbors rather than
    just a distance cutoff as part of your neighbor querying when performing
    this computation since it can otherwise be very sensitive. Specifically, it
    is highly recommended that you choose a number of neighbors that you
    specify a number of neighbors query that requests at least as many
    neighbors as the size of the motif you intend to test against. Otherwise,
    you will struggle to match the motif. However, this is not currently
    enforced (but we could add a warning to the compute...).

    Attributes:
        rmsds (:math:`(N_p, )` :class:`numpy.ndarray`):
            A boolean array of the RMSDs found for each point's environment.
        point_environments (:math:`\left(N_{points}, N_{neighbors}, 3\right)` :class:`numpy.ndarray`):
            All environments for all points.
    """  # noqa: E501

    cdef freud._environment.EnvironmentRMSDMinimizer * thisptr

    def __cinit__(self):
        self.thisptr = self.matchptr = \
            new freud._environment.EnvironmentRMSDMinimizer()

    @Compute._computed_property
    def rmsds(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getRMSDs(),
            freud.util.arr_type_t.FLOAT)

    def compute(self, neighbor_query, motif, neighbors=None,
                registration=False):
        R"""Rotate (if registration=True) and permute the environments of all
        particles to minimize their RMSD with respect to the motif provided by
        motif.

        Args:
            neighbor_query ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                NeighborQuery or (box, points).
            motif ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up the motif against which we are matching.
            neighbors (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value =
                :code:`None`).
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets
                (Default value = :code:`False`).
        Returns:
            :math:`\left(N_{particles}\right)` :class:`numpy.ndarray`:
                Vector of minimal RMSD values, one value per particle.

        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, neighbors=neighbors)

        motif = freud.util._convert_array(motif, shape=(None, 3))
        cdef const float[:, ::1] l_motif = motif
        cdef unsigned int nRef = l_motif.shape[0]

        self.thisptr.compute(
            nq.get_ptr(), nlist.get_ptr(), dereference(qargs.thisptr),
            <vec3[float]*>
            <vec3[float]*> &l_motif[0, 0], nRef,
            registration)

        return self


cdef class AngularSeparationNeighbor(PairCompute):
    R"""Calculates the minimum angles of separation between particles and
    references.

    Attributes:
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list from the last compute, which is generated if not
            provided as the :code:`neighbors` argument.
        angles (:math:`\left(N_{bonds}\right)` :class:`numpy.ndarray`):
            The neighbor angles in radians. The angles are stored in the order
            of the neighborlist object.
    """  # noqa: E501
    cdef freud._environment.AngularSeparationNeighbor * thisptr

    def __cinit__(self):
        self.thisptr = new freud._environment.AngularSeparationNeighbor()

    def __dealloc__(self):
        del self.thisptr

    def compute(self, neighbor_query, orientations, query_points=None,
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
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`):
                Points used to calculate the order parameter. Uses :code:`points`
                if not provided or :code:`None`.
            query_orientations ((:math:`N_{query\_points}`, 4) :class:`numpy.ndarray`):
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
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, query_points, neighbors)

        orientations = freud.util._convert_array(
            orientations, shape=(nq.points.shape[0], 4))
        if query_orientations is None:
            query_orientations = orientations
        else:
            query_orientations = freud.util._convert_array(
                query_orientations, shape=(query_points.shape[0], 4))

        equiv_orientations = freud.util._convert_array(
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
            nlist.get_ptr(),
            dereference(qargs.thisptr))
        return self

    @Compute._computed_property
    def angles(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getAngles(),
            freud.util.arr_type_t.FLOAT)

    def __repr__(self):
        return "freud.environment.{cls}()".format(
            cls=type(self).__name__)

    @Compute._computed_property
    def nlist(self):
        return freud.locality._nlist_from_cnlist(self.thisptr.getNList())


cdef class AngularSeparationGlobal(Compute):
    R"""Calculates the minimum angles of separation between particles and
    references.

    Attributes:
        angles (:math:`\left(N_{bonds}\right)` :class:`numpy.ndarray`):
            The global angles in radians.
    """  # noqa: E501
    cdef freud._environment.AngularSeparationGlobal * thisptr

    def __cinit__(self):
        self.thisptr = new freud._environment.AngularSeparationGlobal()

    def __dealloc__(self):
        del self.thisptr

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
        global_orientations = freud.util._convert_array(
            global_orientations, shape=(None, 4))
        orientations = freud.util._convert_array(
            orientations, shape=(None, 4))
        equiv_orientations = freud.util._convert_array(
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

    @Compute._computed_property
    def angles(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getAngles(),
            freud.util.arr_type_t.FLOAT)

    def __repr__(self):
        return "freud.environment.{cls}()".format(
            cls=type(self).__name__)


cdef class LocalBondProjection(PairCompute):
    R"""Calculates the maximal projection of nearest neighbor bonds for each
    particle onto some set of reference vectors, defined in the particles'
    local reference frame.

    Attributes:
        projections ((:math:`\left(N_{reference}, N_{neighbors}, N_{projection\_vecs} \right)` :class:`numpy.ndarray`):
            The projection of each bond between reference particles and their
            neighbors onto each of the projection vectors.
        normed_projections ((:math:`\left(N_{reference}, N_{neighbors}, N_{projection\_vecs} \right)` :class:`numpy.ndarray`)
            The projection of each bond between reference particles and their
            neighbors onto each of the projection vectors, normalized by the
            length of the bond.
        nlist (:class:`freud.locality.NeighborList`):
            The neighbor list from the last compute, which is generated if not
            provided as the :code:`neighbors` argument.
    """  # noqa: E501
    cdef freud._environment.LocalBondProjection * thisptr

    def __cinit__(self):
        self.thisptr = new freud._environment.LocalBondProjection()

    def __dealloc__(self):
        del self.thisptr

    @Compute._computed_property
    def nlist(self):
        return freud.locality._nlist_from_cnlist(self.thisptr.getNList())

    def compute(self, neighbor_query, orientations, proj_vecs,
                query_points=None, equiv_orientations=np.array([[1, 0, 0, 0]]),
                neighbors=None):
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
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
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
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, query_points, neighbors)

        orientations = freud.util._convert_array(
            orientations, shape=(None, 4))

        equiv_orientations = freud.util._convert_array(
            equiv_orientations, shape=(None, 4))
        proj_vecs = freud.util._convert_array(proj_vecs, shape=(None, 3))

        cdef const float[:, ::1] l_orientations = orientations
        cdef const float[:, ::1] l_equiv_orientations = equiv_orientations
        cdef const float[:, ::1] l_proj_vecs = proj_vecs

        cdef unsigned int n_equiv = l_equiv_orientations.shape[0]
        cdef unsigned int n_proj = l_proj_vecs.shape[0]

        self.thisptr.compute(
            nq.get_ptr(),
            <quat[float]*> &l_orientations[0, 0],
            <vec3[float]*> &l_query_points[0, 0], num_query_points,
            <vec3[float]*> &l_proj_vecs[0, 0], n_proj,
            <quat[float]*> &l_equiv_orientations[0, 0], n_equiv,
            nlist.get_ptr(), dereference(qargs.thisptr))
        return self

    @Compute._computed_property
    def projections(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getProjections(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property
    def normed_projections(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getNormedProjections(),
            freud.util.arr_type_t.FLOAT)

    def __repr__(self):
        return ("freud.environment.{cls}()").format(cls=type(self).__name__)
