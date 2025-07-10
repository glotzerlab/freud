# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

r"""
The :class:`freud.environment` module contains functions which characterize the
local environments of particles in the system. These methods use the positions
and orientations of particles in the local neighborhood of a given particle to
characterize the particle environment.
"""

import collections.abc
import warnings
from importlib.util import find_spec

import numpy as np

import freud._environment
import freud.box
import freud.locality
import freud.util
from freud._util import (  # noqa F401
    ManagedArray_double,
    ManagedArray_float,
    ManagedArray_unsignedint,
    ManagedArrayVec3_float,
    Vector_double,
    Vector_float,
    Vector_unsignedint,
    VectorVec3_float,
)
from freud.errors import NO_DEFAULT_QUERY_ARGS_MESSAGE
from freud.locality import _Compute, _PairCompute, _SpatialHistogram

_HAS_MPL = find_spec("matplotlib") is not None
if _HAS_MPL:
    import freud.plot
else:
    msg_mpl = "Plotting requires matplotlib."


class AngularSeparationNeighbor(_PairCompute):
    r"""Calculates the minimum angles of separation between orientations and
    query orientations."""

    def __init__(self):
        self._cpp_obj = freud._environment.AngularSeparationNeighbor()

    def compute(
        self,
        system,
        orientations,
        query_points=None,
        query_orientations=None,
        equiv_orientations=((1, 0, 0, 0),),
        neighbors=None,
    ):
        r"""Calculates the minimum angles of separation between :code:`orientations`
        and :code:`query_orientations`, checking for underlying symmetry as encoded
        in :code:`equiv_orientations`. The result is stored in the :code:`neighbor_angles`
        class attribute.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Orientations associated with system points that are used to
                calculate bonds.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`,
                          optional):
                Query points used to calculate the correlation function.  Uses
                the system's points if :code:`None` (Default
                value = :code:`None`).
            query_orientations ((:math:`N_{query\_points}`, 4) :class:`numpy.ndarray`,
                          optional):
                Query orientations used to calculate bonds. Uses
                :code:`orientations` if :code:`None`.  (Default
                value = :code:`None`).
            equiv_orientations ((:math:`N_{equiv}`, 4) :class:`numpy.ndarray`,
                          optional):
                The set of all equivalent quaternions that map the particle
                to itself (the elements of its rotational symmetry group).
                Important: :code:`equiv_orientations` must include both
                :math:`q` and :math:`-q`, for all included quaternions. Note
                that this calculation assumes that all points in the system
                share the same set of equivalent orientations.
                (Default value = :code:`((1, 0, 0, 0),)`)
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """  # noqa: E501
        equiv_orientations = np.asarray(equiv_orientations)
        nq, nlist, qargs, query_points, num_query_points = self._preprocess_arguments(
            system, query_points, neighbors
        )

        orientations = freud.util._convert_array(
            orientations, shape=(nq.points.shape[0], 4)
        )
        if query_orientations is None:
            query_orientations = orientations
        else:
            query_orientations = freud.util._convert_array(
                query_orientations, shape=(query_points.shape[0], 4)
            )

        equiv_orientations = freud.util._convert_array(
            equiv_orientations, shape=(None, 4)
        )

        self._cpp_obj.compute(
            nq._cpp_obj,
            orientations,
            query_points,
            query_orientations,
            equiv_orientations,
            nlist._cpp_obj,
            qargs._cpp_obj,
        )
        return self

    @_Compute._computed_property
    def angles(self):
        """:math:`\\left(N_{bonds}\\right)` :class:`numpy.ndarray`: The
        neighbor angles in radians. The angles are stored in the order of the
        neighborlist object."""
        return self._cpp_obj.getAngles().toNumpyArray()

    def __repr__(self):
        return f"freud.environment.{type(self).__name__}()"

    @_Compute._computed_property
    def nlist(self):
        """:class:`freud.locality.NeighborList`: The neighbor list from the
        last compute."""
        nlist = freud.locality._nlist_from_cnlist(self._cpp_obj.getNList())
        nlist._compute = self
        return nlist


class AngularSeparationGlobal(_Compute):
    r"""Calculates the minimum angles of separation between orientations and
    global orientations."""

    def __init__(self):
        self._cpp_obj = freud._environment.AngularSeparationGlobal()

    def compute(
        self, global_orientations, orientations, equiv_orientations=((1, 0, 0, 0),)
    ):
        r"""Calculates the minimum angles of separation between
        :code:`global_orientations` and :code:`orientations`, checking for
        underlying symmetry as encoded in :code:`equiv_orientations`. The
        result is stored in the :code:`global_angles` class attribute.

        Args:
            global_orientations ((:math:`N_{global}`, 4) :class:`numpy.ndarray`):
                Set of global orientations to calculate the order
                parameter.
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Orientations to calculate the order parameter.
            equiv_orientations ((:math:`N_{equiv}`, 4) :class:`numpy.ndarray`, optional):
                The set of all equivalent quaternions that map the particle
                to itself (the elements of its rotational symmetry group).
                Important: :code:`equiv_orientations` must include both
                :math:`q` and :math:`-q`, for all included quaternions. Note
                that this calculation assumes that all points in the system
                share the same set of equivalent orientations.
                (Default value = :code:`((1, 0, 0, 0),)`)
        """  # noqa: E501
        equiv_orientations = np.asarray(equiv_orientations)
        global_orientations = freud.util._convert_array(
            global_orientations, shape=(None, 4)
        )
        orientations = freud.util._convert_array(orientations, shape=(None, 4))
        equiv_orientations = freud.util._convert_array(
            equiv_orientations, shape=(None, 4)
        )

        self._cpp_obj.compute(global_orientations, orientations, equiv_orientations)
        return self

    @_Compute._computed_property
    def angles(self):
        """:math:`\\left(N_{orientations}, N_{global\\_orientations}\\right)` :class:`numpy.ndarray`:
        The global angles in radians."""  # noqa: E501
        return self._cpp_obj.getAngles().toNumpyArray()

    def __repr__(self):
        return f"freud.environment.{type(self).__name__}()"


class BondOrder(_SpatialHistogram):
    r"""Compute the bond orientational order diagram for the system of
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
    variants. These variants can be accessed via the :code:`mode` arguments to
    the :meth:`~BondOrder.compute` method. Available modes of calculation are:

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
    """

    known_modes = {
        "bod": freud._environment.bod,
        "lbod": freud._environment.lbod,
        "obcd": freud._environment.obcd,
        "oocd": freud._environment.oocd,
    }

    def __init__(self, bins, mode="bod"):
        if isinstance(bins, collections.abc.Sequence):
            n_bins_theta, n_bins_phi = bins
        else:
            n_bins_theta = n_bins_phi = bins

        try:
            l_mode = self.known_modes[mode]
        except KeyError as err:
            msg = f"Unknown BondOrder mode: {mode}"
            raise ValueError(msg) from err

        self._cpp_obj = freud._environment.BondOrder(n_bins_theta, n_bins_phi, l_mode)

    @property
    def default_query_args(self):
        """No default query arguments."""
        # Must override the generic histogram's defaults.
        raise NotImplementedError(
            NO_DEFAULT_QUERY_ARGS_MESSAGE.format(type(self).__name__)
        )

    def compute(
        self,
        system,
        orientations=None,
        query_points=None,
        query_orientations=None,
        neighbors=None,
        reset=True,
    ):
        r"""Calculates the correlation function and adds to the current
        histogram.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Orientations associated with system points that are used to
                calculate bonds. Uses identity quaternions if :code:`None`
                (Default value = :code:`None`).
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`,
                          optional):
                Query points used to calculate the correlation function.  Uses
                the system's points if :code:`None` (Default
                value = :code:`None`).
            query_orientations ((:math:`N_{query\_points}`, 4) :class:`numpy.ndarray`,
                          optional):
                Query orientations used to calculate bonds. Uses
                :code:`orientations` if :code:`None`.  (Default
                value = :code:`None`).
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
            reset (bool):
                Whether to erase the previously computed values before adding
                the new computation; if False, will accumulate data (Default
                value: True).
        """
        if reset:
            self._reset()

        nq, nlist, qargs, l_query_points, num_query_points = self._preprocess_arguments(
            system, query_points, neighbors
        )
        if orientations is None:
            orientations = np.array([[1, 0, 0, 0]] * nq.points.shape[0])
        if query_orientations is None:
            query_orientations = orientations
        if query_points is None:
            query_points = nq.points

        orientations = freud.util._convert_array(
            orientations, shape=(nq.points.shape[0], 4)
        )
        query_orientations = freud.util._convert_array(
            query_orientations, shape=(num_query_points, 4)
        )
        query_points = freud.util._convert_array(
            query_points, shape=(num_query_points, 3)
        )

        self._cpp_obj.accumulate(
            nq._cpp_obj,
            orientations,
            query_points,
            query_orientations,
            nlist._cpp_obj,
            qargs._cpp_obj,
        )
        return self

    @_Compute._computed_property
    def bond_order(self):
        """:math:`\\left(N_{\\phi}, N_{\\theta} \\right)` :class:`numpy.ndarray`: Bond order."""  # noqa: E501
        return self._cpp_obj.getBondOrder().toNumpyArray()

    @_Compute._computed_property
    def box(self):
        """:class:`freud.box.Box`: Box used in the calculation."""
        return freud.box.BoxFromCPP(self._cpp_obj.getBox())

    def __repr__(self):
        return "freud.environment.{cls}(bins=({bins}), mode='{mode}')".format(
            cls=type(self).__name__,
            bins=", ".join([str(b) for b in self.nbins]),
            mode=self.mode,
        )

    @property
    def mode(self):
        """str: Bond order mode."""
        return self._cpp_obj.getMode().name


class LocalDescriptors(_PairCompute):
    r"""Compute a set of descriptors (a numerical "fingerprint") of a particle's
    local environment.

    The resulting spherical harmonic array will be a complex-valued
    array of shape :code:`(num_bonds, num_sphs)`. Spherical harmonic
    calculation can be restricted to some number of nearest neighbors
    through the :code:`max_num_neighbors` argument; if a particle has more
    bonds than this number, the last one or more rows of bond spherical
    harmonics for each particle will not be set. This feature is useful for
    computing descriptors on the same system but with different subsets of
    neighbors; a :class:`freud.locality.NeighborList` with the correct
    ordering can then be reused in multiple calls to :meth:`~.compute`
    with different values of :code:`max_num_neighbors` to compute descriptors
    for different local neighborhoods with maximum efficiency.

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
    """

    known_modes = {
        "neighborhood": freud._environment.LocalNeighborhood,
        "global": freud._environment.Global,
        "particle_local": freud._environment.ParticleLocal,
    }

    def __init__(self, l_max, negative_m=True, mode="neighborhood"):
        try:
            l_mode = self.known_modes[mode]
        except KeyError:
            msg = f"Unknown LocalDescriptors orientation mode: {mode}"
            raise ValueError(msg) from None

        self._cpp_obj = freud._environment.LocalDescriptors(l_max, negative_m, l_mode)

    def compute(
        self,
        system,
        query_points=None,
        orientations=None,
        neighbors=None,
        max_num_neighbors=0,
    ):
        r"""Calculates the local descriptors of bonds from a set of source
        points to a set of destination points.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the correlation function.  Uses
                the system's points if :code:`None` (Default
                value = :code:`None`).
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Orientations associated with system points that are used to
                calculate bonds.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
            max_num_neighbors (unsigned int, optional):
                Hard limit on the maximum number of neighbors to use for each
                particle for the given neighbor-finding algorithm. Uses
                all neighbors if set to 0 (Default value = 0).
        """  # noqa: E501
        nq, nlist, qargs, query_points, num_query_points = self._preprocess_arguments(
            system, query_points, neighbors
        )

        # The l_orientations_ptr is only used for 'particle_local' mode.
        if self.mode == "particle_local":
            if orientations is None:
                msg = (
                    "Orientations must be given to orient LocalDescriptors "
                    "with particles' orientations"
                )
                raise RuntimeError(msg)

            orientations = freud.util._convert_array(
                orientations, shape=(nq.points.shape[0], 4)
            )

        self._cpp_obj.compute(
            nq._cpp_obj,
            query_points,
            num_query_points,
            orientations,
            nlist._cpp_obj,
            qargs._cpp_obj,
            max_num_neighbors,
        )
        return self

    @_Compute._computed_property
    def nlist(self):
        """:class:`freud.locality.NeighborList`: The neighbor list from the
        last compute."""
        nlist = freud.locality._nlist_from_cnlist(self._cpp_obj.getNList())
        nlist._compute = self
        return nlist

    @_Compute._computed_property
    def sph(self):
        """:math:`\\left(N_{bonds}, \\text{SphWidth} \\right)`
        :class:`numpy.ndarray`: The last computed spherical harmonic array."""
        return self._cpp_obj.getSph().toNumpyArray()

    @_Compute._computed_property
    def num_sphs(self):
        """unsigned int: The last number of spherical harmonics computed. This
        is equal to the number of bonds in the last computation, which is at
        most the number of :code:`points` multiplied by the lower of the
        :code:`num_neighbors` arguments passed to the last compute call or the
        constructor (it may be less if there are not enough neighbors for every
        particle)."""
        return self._cpp_obj.getNSphs()

    @property
    def l_max(self):
        """unsigned int: The maximum spherical harmonic :math:`l` calculated
        for."""
        return self._cpp_obj.getLMax()

    @property
    def negative_m(self):
        """bool: True if we also calculated :math:`Y_{lm}` for negative
        :math:`m`."""
        return self._cpp_obj.getNegativeM()

    @property
    def mode(self):
        """str: Orientation mode to use for environments, either
        :code:`'neighborhood'` to use the orientation of the local
        neighborhood, :code:`'particle_local'` to use the given particle
        orientations, or :code:`'global'` to not rotate environments."""
        mode = self._cpp_obj.getMode()
        for key, value in self.known_modes.items():
            if value == mode:
                return key
        return None

    def __repr__(self):
        return (
            f"freud.environment.{type(self).__name__}(l_max={self.l_max}, "
            f"negative_m={self.negative_m}, mode='{self.mode}')"
        )


def _minimize_RMSD(box, ref_points, points, registration=False):
    r"""Get the somewhat-optimal RMSD between the set of vectors ref_points
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
    b = freud.util._convert_box(box)

    ref_points = freud.util._convert_array(ref_points, shape=(None, 3))
    points = freud.util._convert_array(points, shape=(None, 3))

    nRef1 = ref_points.shape[0]
    nRef2 = points.shape[0]

    if nRef1 != nRef2:
        msg = (
            "The number of vectors in ref_points must MATCH"
            "the number of vectors in points"
        )
        raise ValueError(msg)

    min_rmsd = -1
    min_rmsd, results_map = freud._environment.minimizeRMSD(
        b._cpp_obj, ref_points, points, nRef1, min_rmsd, registration
    )
    return [min_rmsd, np.asarray(points), results_map]


def _is_similar_motif(box, ref_points, points, threshold, registration=False):
    r"""Test if the motif provided by ref_points is similar to the motif
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
    b = freud.util._convert_box(box)

    ref_points = freud.util._convert_array(ref_points, shape=(None, 3))
    points = freud.util._convert_array(points, shape=(None, 3))

    nRef1 = ref_points.shape[0]
    nRef2 = points.shape[0]
    threshold_sq = threshold * threshold

    if nRef1 != nRef2:
        msg = (
            "The number of vectors in ref_points must match"
            "the number of vectors in points"
        )
        raise ValueError(msg)

    vec_map = freud._environment.isSimilar(
        b._cpp_obj, ref_points, points, nRef1, threshold_sq, registration
    )
    return [np.asarray(points), vec_map]


class _MatchEnv(_PairCompute):
    r"""Parent for environment matching methods."""

    def __init__(self, *args, **kwargs):
        # Abstract class
        self._cpp_obj = freud._environment.MatchEnv()

    @_Compute._computed_property
    def point_environments(self):
        """:math:`\\left(N_{points}, N_{neighbors}, 3\\right)`
        list[:class:`numpy.ndarray`]: All environments for all points."""
        envs = self._cpp_obj.getPointEnvironments()
        return [np.asarray(env) for env in envs]

    def __repr__(self):
        return f"freud.environment.{type(self).__name__}()"


class EnvironmentCluster(_MatchEnv):
    r"""Clusters particles according to whether their local environments match
    or not, using various shape matching metrics defined in :cite:`Teich2019`.
    """

    def __init__(self):
        self._cpp_obj = freud._environment.EnvironmentCluster()

    def compute(
        self,
        system,
        threshold,
        cluster_neighbors=None,
        env_neighbors=None,
        registration=False,
    ):
        r"""Determine clusters of particles with matching environments.

        An environment is defined by the bond vectors between a particle and its
        neighbors as defined by :code:`env_neighbors`.
        For example, :code:`env_neighbors= {'num_neighbors': 8}` means that every
        particle's local environment is defined by its 8 nearest neighbors.
        Then, each particle's environment is compared to the environments of
        particles that satisfy a different cutoff parameter :code:`cluster_neighbors`.
        For example, :code:`cluster_neighbors={'r_max': 3.0}`
        means that the environment of each particle will be compared to the
        environment of every particle within a distance of 3.0.

        Two environments are compared using `point-set registration
        <https://en.wikipedia.org/wiki/Point-set_registration)>`_.

        The thresholding criterion we apply in order to determine if the two point sets
        match is quite conservative: the two point sets match if and only if,
        for every pair of matched points between the sets, the distance between
        the matched pair is less than :code:`threshold`.

        When :code:`registration=False`, environments are not rotated prior to comparison
        between them. Which pairs of vectors are compared depends on the order in
        which the vectors of the environment are traversed.

        When :code:`registration=True`, we use the `Kabsch algorithm
        <https://en.wikipedia.org/wiki/Kabsch_algorithm>`_ to find the optimal
        rotation matrix mapping one environment onto another. The Kabsch
        algorithm assumes a one-to-one correspondence between points in the two
        environments. However, we typically do not know which point in one environment
        corresponds to a particular point in the other environment. The brute force
        solution is to try all possible correspondences, but the resulting
        combinatorial explosion makes this problem infeasible, so we use the thresholding
        criterion above to terminate the search when we have found a permutation
        of points that results in a sufficiently low RMSD.

        .. note::

            Using a distance cutoff for :code:`env_neighbors` could
            lead to situations where the :code:`cluster_environments`
            contain different numbers of neighbors. In this case, the
            environments which have a number of neighbors less than
            the environment with the maximum number of neighbors
            :math:`k_{max}` will have their entry in :code:`cluster_environments`
            padded with zero vectors. For example, a cluster environment
            with :math:`m < k` neighbors, will have :math:`k - m` zero
            vectors at the end of its entry in :code:`cluster_environments`.


        .. warning::

            All vectors of :code:`cluster_environments` and :code:`point_environments`
            are defined with respect to the query particle.
            Zero vectors are only used to pad the cluster vectors so that they
            have the same shape.
            In a future version of freud, zero-padding will be removed.

        .. warning::

            Comparisons between two environments are only made when both
            environments contain the same number of neighbors.
            However, no warning will be given at runtime if mismatched
            environments are provided for comparison.

        Example::

            >>> import freud
            >>> # Compute clusters of particles with matching environments
            >>> box, points = freud.data.make_random_system(10, 100, seed=0)
            >>> env_cluster = freud.environment.EnvironmentCluster()
            >>> env_cluster.compute(
            ...     system = (box, points),
            ...     threshold=0.2,
            ...     cluster_neighbors={'num_neighbors': 6},
            ...     registration=False)
            freud.environment.EnvironmentCluster()

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            threshold (float):
                Maximum magnitude of the vector difference between two vectors,
                below which they are "matching". Typically, a good choice is
                between 10% and 30% of the first well in the radial
                distribution function (this has distance units).
            cluster_neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_.
                Defines the neighbors used for comparing different particle
                environments (Default value: None).
            env_neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_.
                Defines the neighbors used as the environment of each particle.
                If ``None``, the value provided for ``neighbors`` will be used
                (Default value: None).
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets. Enabling this
                option incurs a significant performance penalty.
                (Default value = :code:`False`)
        """  # noqa: E501
        nq, nlist, qargs, l_query_points, num_query_points = self._preprocess_arguments(
            system, neighbors=cluster_neighbors
        )

        if env_neighbors is None:
            env_neighbors = cluster_neighbors
        env_nlist, env_qargs = self._resolve_neighbors(env_neighbors)

        self._cpp_obj.compute(
            nq._cpp_obj,
            nlist._cpp_obj,
            qargs._cpp_obj,
            env_nlist._cpp_obj,
            env_qargs._cpp_obj,
            threshold,
            registration,
        )
        return self

    @_Compute._computed_property
    def cluster_idx(self):
        """:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`: The
        per-particle index indicating cluster membership."""
        return self._cpp_obj.getClusters().toNumpyArray()

    @_Compute._computed_property
    def num_clusters(self):
        """unsigned int: The number of clusters."""
        return self._cpp_obj.getNumClusters()

    @_Compute._computed_property
    def cluster_environments(self):
        """:math:`\\left(N_{clusters}, N_{neighbors}, 3\\right)`
        list[:class:`numpy.ndarray`]: The environments for all clusters."""
        envs = self._cpp_obj.getClusterEnvironments()
        return [np.asarray(env) for env in envs]

    def plot(self, ax=None):
        """Plot cluster distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        if not _HAS_MPL:
            raise ImportError(msg_mpl)
        try:
            values, counts = np.unique(self.cluster_idx, return_counts=True)
        except ValueError:
            return None
        else:
            return freud.plot.clusters_plot(
                values, counts, num_clusters_to_plot=10, ax=ax
            )

    def _repr_png_(self):
        try:
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


class EnvironmentMotifMatch(_MatchEnv):
    r"""Find matches between local arrangements of a set of points and
    a provided motif, as done in :cite:`Teich2019`.

    A particle's environment can only match the motif if it contains the
    same number of neighbors as the motif. Any environment with a
    different number of neighbors than the motif will always fail to match
    the motif. See :class:`freud.environment.EnvironmentCluster.compute` for
    the matching criterion.
    """

    def __init__(self):
        self._cpp_obj = freud._environment.EnvironmentMotifMatch()

    def compute(self, system, motif, threshold, env_neighbors=None, registration=False):
        r"""Determine which particles have local environments
            matching the given environment motif.

        .. warning::

            Comparisons between two environments are only made when both
            environments contain the same number of neighbors.
            However, no warning will be given at runtime if mismatched
            environments are provided for comparison.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            motif ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):
                Vectors that make up the motif against which we are matching.
            threshold (float):
                Maximum magnitude of the vector difference between two vectors,
                below which they are "matching". Typically, a good choice is
                between 10% and 30% of the first well in the radial
                distribution function (this has distance units).
            env_neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_.
                Defines the environment of the query particles
                (Default value: None).
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets
                (Default value = False).
        """
        nq, nlist, qargs, l_query_points, num_query_points = self._preprocess_arguments(
            system, neighbors=env_neighbors
        )

        motif = freud.util._convert_array(motif, shape=(None, 3))
        if (motif == 0).all(axis=1).any():
            warnings.warn(
                "Attempting to match a motif containing the zero "
                "vector is likely to result in zero matches.",
                RuntimeWarning,
                stacklevel=2,
            )
        nRef = motif.shape[0]

        self._cpp_obj.compute(
            nq._cpp_obj,
            nlist._cpp_obj,
            qargs._cpp_obj,
            motif,
            nRef,
            threshold,
            registration,
        )
        return self

    @_Compute._computed_property
    def matches(self):
        """(:math:`N_{points}`) :class:`numpy.ndarray`: A boolean array indicating
        whether each point matches the motif."""
        # NOTE: Numpy stores bools as a byte each, so this cast should be free
        return self._cpp_obj.getMatches().toNumpyArray().astype(bool)


class _EnvironmentRMSDMinimizer(_MatchEnv):
    r"""Find linear transformations that map the environments of points onto a
    motif.

    In general, it is recommended to specify a number of neighbors rather than
    just a distance cutoff as part of your neighbor querying when performing
    this computation since it can otherwise be very sensitive. Specifically, it
    is highly recommended that you choose a number of neighbors that you
    specify a number of neighbors query that requests at least as many
    neighbors as the size of the motif you intend to test against. Otherwise,
    you will struggle to match the motif. However, this is not currently
    enforced (but we could add a warning to the compute...).
    """

    def __init__(self):
        self._cpp_obj = freud._environment.EnvironmentRMSDMinimizer()

    def compute(self, system, motif, neighbors=None, registration=False):
        r"""Rotate (if registration=True) and permute the environments of all
        particles to minimize their RMSD with respect to the motif provided by
        motif.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            motif ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Vectors that make up the motif against which we are matching.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
            registration (bool, optional):
                If True, first use brute force registration to orient one set
                of environment vectors with respect to the other set such that
                it minimizes the RMSD between the two sets
                (Default value = :code:`False`).
        Returns:
            :math:`\left(N_{particles}\right)` :class:`numpy.ndarray`:
                Vector of minimal RMSD values, one value per particle.

        """
        nq, nlist, qargs, l_query_points, num_query_points = self._preprocess_arguments(
            system, neighbors=neighbors
        )

        motif = freud.util._convert_array(motif, shape=(None, 3))
        nRef = motif.shape[0]

        self._cpp_obj.compute(
            nq._cpp_obj, nlist._cpp_obj, qargs._cpp_obj, motif, nRef, registration
        )
        return self

    @_Compute._computed_property
    def rmsds(self):
        """:math:`(N_p, )` :class:`numpy.ndarray`: A boolean array of the RMSDs
        found for each point's environment."""
        return self._cpp_obj.getRMSDs().toNumpyArray()


class LocalBondProjection(_PairCompute):
    r"""Calculates the maximal projection of nearest neighbor bonds for each
    particle onto some set of reference vectors, defined in the particles'
    local reference frame.
    """

    def __init__(self):
        self._cpp_obj = freud._environment.LocalBondProjection()

    def compute(
        self,
        system,
        orientations,
        proj_vecs,
        query_points=None,
        equiv_orientations=((1, 0, 0, 0),),
        neighbors=None,
    ):
        r"""Calculates the maximal projections of nearest neighbor bonds
        (between :code:`points` and :code:`query_points`) onto the set of
        reference vectors :code:`proj_vecs`, defined in the local reference
        frames of the :code:`points` as defined by the orientations
        :code:`orientations`. This computation accounts for the underlying
        symmetries of the reference frame as encoded in :code:`equiv_orientations`.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            orientations ((:math:`N_{points}`, 4) :class:`numpy.ndarray`):
                Orientations associated with system points that are used to
                calculate bonds.
            proj_vecs ((:math:`N_{vectors}`, 3) :class:`numpy.ndarray`):
                The set of projection vectors, defined in the query
                particles' reference frame, to calculate maximal local bond
                projections onto.
            query_points ((:math:`N_{query\_points}`, 3) :class:`numpy.ndarray`, optional):
                Query points used to calculate the correlation function.  Uses
                the system's points if :code:`None` (Default
                value = :code:`None`).
                (Default value = :code:`None`).
            equiv_orientations ((:math:`N_{equiv}`, 4) :class:`numpy.ndarray`, optional):
                The set of all equivalent quaternions that map the particle
                to itself (the elements of its rotational symmetry group).
                Important: :code:`equiv_orientations` must include both
                :math:`q` and :math:`-q`, for all included quaternions. Note
                that this calculation assumes that all points in the system
                share the same set of equivalent orientations.
                (Default value = :code:`((1, 0, 0, 0),)`)
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """  # noqa: E501
        equiv_orientations = np.asarray(equiv_orientations)
        nq, nlist, qargs, query_points, num_query_points = self._preprocess_arguments(
            system, query_points, neighbors
        )

        orientations = freud.util._convert_array(orientations, shape=(None, 4))

        equiv_orientations = freud.util._convert_array(
            equiv_orientations, shape=(None, 4)
        )
        proj_vecs = freud.util._convert_array(proj_vecs, shape=(None, 3))

        self._cpp_obj.compute(
            nq._cpp_obj,
            orientations,
            query_points,
            proj_vecs,
            equiv_orientations,
            nlist._cpp_obj,
            qargs._cpp_obj,
        )
        return self

    @_Compute._computed_property
    def nlist(self):
        """:class:`freud.locality.NeighborList`: The neighbor list from the
        last compute."""
        nlist = freud.locality._nlist_from_cnlist(self._cpp_obj.getNList())
        nlist._compute = self
        return nlist

    @_Compute._computed_property
    def projections(self):
        """:math:`\\left(N_{bonds}, N_{projection\\_vecs} \\right)` :class:`numpy.ndarray`:
        The projection of each bond between query particles and their neighbors
        onto each of the projection vectors."""  # noqa: E501
        return self._cpp_obj.getProjections().toNumpyArray()

    @_Compute._computed_property
    def normed_projections(self):
        """:math:`\\left(N_{bonds}, N_{projection\\_vecs} \\right)` :class:`numpy.ndarray`:
        The projection of each bond between query particles and their neighbors
        onto each of the projection vectors, normalized by the length of the
        bond."""  # noqa: E501
        return self._cpp_obj.getNormedProjections().toNumpyArray()

    def __repr__(self):
        return f"freud.environment.{type(self).__name__}()"
