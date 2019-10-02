# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.order` module contains functions which compute order
parameters for the whole system or individual particles. Order parameters take
bond order data and interpret it in some way to quantify the degree of order in
a system using a scalar value. This is often done through computing spherical
harmonics of the bond order diagram, which are the spherical analogue of
Fourier Transforms.
"""

import warnings
import numpy as np
import time
import freud.locality
import logging

from freud.util cimport Compute
from freud.locality cimport PairCompute
from freud.util cimport vec3, quat
from cython.operator cimport dereference

cimport freud._order
cimport freud.locality
cimport freud.box
cimport freud.util

cimport numpy as np

logger = logging.getLogger(__name__)

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Cubatic(Compute):
    R"""Compute the cubatic order parameter [HajiAkbari2015]_ for a system of
    particles using simulated annealing instead of Newton-Raphson root finding.

    Args:
        t_initial (float):
            Starting temperature.
        t_final (float):
            Final temperature.
        scale (float):
            Scaling factor to reduce temperature.
        n_replicates (unsigned int, optional):
            Number of replicate simulated annealing runs.
            (Default value = :code:`1`).
        seed (unsigned int, optional):
            Random seed to use in calculations. If :code:`None`, system time is used.
            (Default value = :code:`None`).

    Attributes:
        t_initial (float):
            The value of the initial temperature.
        t_final (float):
            The value of the final temperature.
        scale (float):
            The scale
        order (float):
            The cubatic order parameter.
        orientation (:math:`\left(4 \right)` :class:`numpy.ndarray`):
            The quaternion of global orientation.
        particle_order (:class:`numpy.ndarray`):
             Cubatic order parameter.
        particle_tensor (:math:`\left(N_{particles}, 3, 3, 3, 3 \right)` :class:`numpy.ndarray`):
            Rank 5 tensor corresponding to each individual particle
            orientation.
        global_tensor (:math:`\left(3, 3, 3, 3 \right)` :class:`numpy.ndarray`):
            Rank 4 tensor corresponding to global orientation.
        cubatic_tensor (:math:`\left(3, 3, 3, 3 \right)` :class:`numpy.ndarray`):
            Rank 4 cubatic tensor.
        gen_r4_tensor (:math:`\left(3, 3, 3, 3 \right)` :class:`numpy.ndarray`):
            Rank 4 tensor corresponding to each individual particle
            orientation.
    """  # noqa: E501
    cdef freud._order.Cubatic * thisptr
    cdef n_replicates
    cdef seed

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
            except (OverflowError, TypeError, ValueError):
                logger.warning("The supplied seed could not be used. "
                               "Using current time as seed.")
                seed = int(time.time())

        self.thisptr = new freud._order.Cubatic(
            t_initial, t_final, scale, n_replicates, seed)
        self.n_replicates = n_replicates

    def __dealloc__(self):
        del self.thisptr

    def compute(self, orientations):
        R"""Calculates the per-particle and global order parameter.

        Args:
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Orientations as angles to use in computation.
        """
        orientations = freud.util._convert_array(
            orientations, shape=(None, 4))

        cdef const float[:, ::1] l_orientations = orientations
        cdef unsigned int num_particles = l_orientations.shape[0]

        self.thisptr.compute(
            <quat[float]*> &l_orientations[0, 0], num_particles)
        return self

    @property
    def t_initial(self):
        return self.thisptr.getTInitial()

    @property
    def t_final(self):
        return self.thisptr.getTFinal()

    @property
    def scale(self):
        return self.thisptr.getScale()

    @property
    def seed(self):
        return self.thisptr.getSeed()

    @Compute._computed_property
    def order(self):
        return self.thisptr.getCubaticOrderParameter()

    @Compute._computed_property
    def orientation(self):
        cdef quat[float] q = self.thisptr.getCubaticOrientation()
        return np.asarray([q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)

    @Compute._computed_property
    def particle_order(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getParticleOrderParameter(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property
    def global_tensor(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getGlobalTensor(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property
    def cubatic_tensor(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getCubaticTensor(),
            freud.util.arr_type_t.FLOAT)

    def __repr__(self):
        return ("freud.order.{cls}(t_initial={t_initial}, t_final={t_final}, "
                "scale={scale}, n_replicates={n_replicates}, "
                "seed={seed})").format(cls=type(self).__name__,
                                       t_initial=self.t_initial,
                                       t_final=self.t_final,
                                       scale=self.scale,
                                       n_replicates=self.n_replicates,
                                       seed=self.seed)


cdef class Nematic(Compute):
    R"""Compute the nematic order parameter for a system of particles.

    Args:
        u (:math:`\left(3 \right)` :class:`numpy.ndarray`):
            The nematic director of a single particle in the reference state
            (without any rotation applied).

    Attributes:
        order (float):
            Nematic order parameter.
        director (:math:`\left(3 \right)` :class:`numpy.ndarray`):
            The average nematic director.
        particle_tensor (:math:`\left(N_{particles}, 3, 3 \right)` :class:`numpy.ndarray`):
            One 3x3 matrix per-particle corresponding to each individual
            particle orientation.
        nematic_tensor (:math:`\left(3, 3 \right)` :class:`numpy.ndarray`):
            3x3 matrix corresponding to the average particle orientation.
        u (:math:`\left(3 \right)` :class:`numpy.ndarray`):
            The normalized reference director (the normalized vector provided
            on construction).
    """  # noqa: E501
    cdef freud._order.Nematic *thisptr

    def __cinit__(self, u):
        # run checks
        if len(u) != 3:
            raise ValueError('u needs to be a three-dimensional vector')

        cdef vec3[float] l_u = vec3[float](u[0], u[1], u[2])
        self.thisptr = new freud._order.Nematic(l_u)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, orientations):
        R"""Calculates the per-particle and global order parameter.

        Args:
            orientations (:math:`\left(N_{particles}, 4 \right)` :class:`numpy.ndarray`):
                Orientations to calculate the order parameter.
        """  # noqa: E501
        orientations = freud.util._convert_array(
            orientations, shape=(None, 4))

        cdef const float[:, ::1] l_orientations = orientations
        cdef unsigned int num_particles = l_orientations.shape[0]

        self.thisptr.compute(<quat[float]*> &l_orientations[0, 0],
                             num_particles)
        return self

    @Compute._computed_property
    def order(self):
        return self.thisptr.getNematicOrderParameter()

    @Compute._computed_property
    def director(self):
        cdef vec3[float] n = self.thisptr.getNematicDirector()
        return np.asarray([n.x, n.y, n.z], dtype=np.float32)

    @Compute._computed_property
    def particle_tensor(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getParticleTensor(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property
    def nematic_tensor(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getNematicTensor(),
            freud.util.arr_type_t.FLOAT)

    @property
    def u(self):
        cdef vec3[float] u = self.thisptr.getU()
        return np.asarray([u.x, u.y, u.z], dtype=np.float32)

    def __repr__(self):
        return "freud.order.{cls}(u={u})".format(cls=type(self).__name__,
                                                 u=self.u.tolist())


cdef class Hexatic(PairCompute):
    R"""Calculates the :math:`k`-atic order parameter for 2D systems.

    The :math:`k`-atic order parameter (called the hexatic order parameter for
    :math:`k = 6`) is analogous to Steinhardt order parameters, and is used to
    measure order in the bonds of 2D systems.

    The :math:`k`-atic order parameter for a particle :math:`i` and its
    :math:`n` neighbors :math:`j` is given by:

    :math:`\psi_k \left( i \right) = \frac{1}{n}
    \sum_j^n e^{k i \phi_{ij}}`

    The parameter :math:`k` governs the symmetry of the order parameter and
    typically matches the number of neighbors to be found for each particle.
    The quantity :math:`\phi_{ij}` is the angle between the
    vector :math:`r_{ij}` and :math:`\left( 1,0 \right)`.

    .. note::
        **2D:** :class:`freud.order.Hexatic` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.

    Args:
        k (unsigned int, optional):
            Symmetry of order parameter. (Default value = :code:`6`).

    Attributes:
        k (unsigned int):
            Symmetry of the order parameter.
        particle_order (:math:`\left(N_{particles} \right)` :class:`numpy.ndarray`):
            Order parameter.
    """  # noqa: E501
    cdef freud._order.Hexatic * thisptr

    def __cinit__(self, k=6):
        self.thisptr = new freud._order.Hexatic(k)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, neighbor_query, neighbors=None):
        R"""Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds.
                (Default value = :code:`None`).
            query_args (dict): A dictionary of query arguments (Default value =
                :code:`None`).
        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, neighbors=neighbors)
        self.thisptr.compute(nlist.get_ptr(),
                             nq.get_ptr(), dereference(qargs.thisptr))
        return self

    @property
    def default_query_args(self):
        return dict(mode="nearest", num_neighbors=self.k)

    @Compute._computed_property
    def particle_order(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getOrder(),
            freud.util.arr_type_t.COMPLEX_FLOAT)

    @property
    def k(self):
        return self.thisptr.getK()

    def __repr__(self):
        return "freud.order.{cls}(k={k})".format(
            cls=type(self).__name__, k=self.k)


cdef class Translational(PairCompute):
    R"""Compute the translational order parameter for each particle.

    .. note::
        **2D:** :class:`freud.order.Translational` is only defined for 2D
        systems. The points must be passed in as :code:`[x, y, 0]`.

    Args:
        k (float, optional):
            Symmetry of order parameter. (Default value = :code:`6.0`).

    Attributes:
        k (float):
            Normalization value (order is divided by k).
        particle_order (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            Reference to the last computed translational order array.
    """  # noqa E501
    cdef freud._order.Translational * thisptr

    def __cinit__(self, k=6.0):
        self.thisptr = new freud._order.Translational(k)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, neighbor_query, neighbors=None):
        R"""Calculates the local descriptors.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds.
                (Default value = :code:`None`).
            query_args (dict): A dictionary of query arguments (Default value =
                :code:`None`).
        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, neighbors=neighbors)

        self.thisptr.compute(nlist.get_ptr(),
                             nq.get_ptr(), dereference(qargs.thisptr))
        return self

    @property
    def default_query_args(self):
        return dict(mode="nearest", num_neighbors=int(self.k))

    @Compute._computed_property
    def particle_order(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getOrder(),
            freud.util.arr_type_t.COMPLEX_FLOAT)

    @property
    def k(self):
        return self.thisptr.getK()

    def __repr__(self):
        return "freud.order.{cls}(k={k})".format(
            cls=type(self).__name__, k=self.k)


cdef class Steinhardt(PairCompute):
    R"""Compute the local Steinhardt [Steinhardt1983]_ rotationally invariant
    :math:`Q_l` :math:`W_l` order parameter for a set of points.

    Implements the local rotationally invariant :math:`Q_l` or :math:`W_l` order
    parameter described by Steinhardt. For a particle i, we calculate the
    average order parameter by summing the spherical harmonics between particle
    :math:`i` and its neighbors :math:`j` in a local region:
    :math:`\overline{Q}_{lm}(i) = \frac{1}{N_b} \displaystyle\sum_{j=1}^{N_b}
    Y_{lm}(\theta(\vec{r}_{ij}), \phi(\vec{r}_{ij}))`. The particles included in
    the sum are determined by the r_max argument to the constructor.

    For :math:`Q_l`, this is then combined in a rotationally invariant fashion to
    remove local orientational order as follows:
    :math:`Q_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l}
    |\overline{Q}_{lm}|^2 }`.

    For :math:`W_l`, it is then defined as a weighted average over the
    :math:`\overline{Q}_{lm}(i)` values using Wigner 3j symbols
    (Clebsch-Gordan coefficients). The resulting combination is rotationally
    (i.e. frame) invariant.

    The average argument in the constructor provides access to a variant
    of this parameter that performs a average over the first and second shell
    combined [Lechner2008]_. To compute this parameter, we perform a second
    averaging over the first neighbor shell of the particle to implicitly
    include information about the second neighbor shell. This averaging is
    performed by replacing the value :math:`\overline{Q}_{lm}(i)` in the
    original definition by the average value of :math:`\overline{Q}_{lm}(k)`
    over all the :math:`k` neighbors of particle :math:`i` as well as itself.

    The :code:`norm` attribute argument provides normalized versions of the
    order parameter, where the normalization is performed by averaging the
    :math:`Q_{lm}` values over all particles before computing the order
    parameter of choice.

    Args:
        l (unsigned int):
            Spherical harmonic quantum number l.
        average (bool, optional):
            Determines whether to calculate the averaged Steinhardt order
            parameter. (Default value = :code:`False`)
        Wl (bool, optional):
            Determines whether to use the :math:`W_l` version of the Steinhardt
            order parameter. (Default value = :code:`False`)
        weighted (bool, optional):
            Determines whether to use neighbor weights in the computation of
            spherical harmonics over neighbors. If enabled and used with a
            Voronoi neighbor list, this results in the Minkowski Structure
            Metrics :math:`Q'_l`. (Default value = :code:`False`)

    Attributes:
        particle_order (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed selected variant of the Steinhardt order
            parameter for each particle (filled with NaN for particles with no
            neighbors).
        order (float):
            The system wide normalization of the :math:`Q_l` or
            :math:`W_l` order parameter.
    """  # noqa: E501
    cdef freud._order.Steinhardt * thisptr

    def __cinit__(self, l, average=False, Wl=False, weighted=False):
        self.thisptr = new freud._order.Steinhardt(l, average, Wl, weighted)

    def __dealloc__(self):
        del self.thisptr

    @property
    def average(self):
        return self.thisptr.isAverage()

    @property
    def Wl(self):
        return self.thisptr.isWl()

    @property
    def weighted(self):
        return self.thisptr.isWeighted()

    @property
    def l(self):  # noqa: E743
        return self.thisptr.getL()

    @Compute._computed_property
    def order(self):
        return self.thisptr.getOrder()

    @Compute._computed_property
    def particle_order(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getParticleOrder(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property
    def Ql(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getQl(),
            freud.util.arr_type_t.FLOAT)

    def compute(self, neighbor_query, neighbors=None):
        R"""Compute the order parameter.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds.
                (Default value = :code:`None`).
        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, neighbors=neighbors)

        self.thisptr.compute(nlist.get_ptr(),
                             nq.get_ptr(),
                             dereference(qargs.thisptr))
        return self

    def __repr__(self):
        return ("freud.order.{cls}(l={l}, average={average}, Wl={Wl}, "
                "weighted={weighted})").format(
                    cls=type(self).__name__,
                    l=self.l, # noqa: 743
                    average=self.average,
                    Wl=self.Wl,
                    weighted=self.weighted)

    def plot(self, ax=None):
        """Plot order parameter distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        xlabel = r"${mode_letter}{prime}_{{{sph_l}{average}}}$".format(
            mode_letter='w' if self.Wl else 'q',
            prime='\'' if self.weighted else '',
            sph_l=self.l,
            average=',ave' if self.average else '')

        return freud.plot.histogram_plot(
            self.order,
            title="Steinhardt Order Parameter " + xlabel,
            xlabel=xlabel,
            ylabel=r"Number of particles",
            ax=ax)

    def _repr_png_(self):
        import freud.plot
        try:
            return freud.plot.ax_to_bytes(self.plot())
        except AttributeError:
            return None


cdef class SolidLiquid(PairCompute):
    R"""Identifies solid-like clusters using dot products of :math:`Q_{lm}`.

    The solid-liquid order parameter [tenWolde1995]_ uses a Steinhardt-like
    approach to identify solid-like particles. First, a bond parameter
    :math:`Q_l(i, j)` is computed for each neighbor bond.

    If :code:`normalize_Q` is true (default), the bond parameter is given by
    :math:`Q_l(i, j) = \frac{\sum_{m=-l}^{l} \text{Re}~Q_{lm}(i) Q_{lm}^*(j)}
    {\sqrt{\sum_{m=-l}^{l} \lvert Q_{lm}(i) \rvert^2}
    \sqrt{\sum_{m=-l}^{l} \lvert Q_{lm}(j) \rvert^2}}`

    If :code:`normalize_Q` is false, then the denominator of the above
    expression is left out.

    Next, the bonds are filtered to keep only "solid-like" bonds with
    :math:`Q_l(i, j)` above a cutoff value :math:`Q_{threshold}`.

    If a particle has more than :math:`S_{threshold}` solid-like bonds, then
    the particle is considered solid-like. Finally, solid-like particles are
    clustered.

    .. [tenWolde1995] ten Wolde, P. R., Ruiz-Montero, M. J., & Frenkel, D.
       (1995).  Numerical Evidence for bcc Ordering at the Surface of a
       Critical fcc Nucleus. Phys. Rev. Lett., 75 (2714).
       https://doi.org/10.1103/PhysRevLett.75.2714

    .. [Filion2010] Filion, L., Hermes, M., Ni, R., & Dijkstra, M. (2010).
       Crystal nucleation of hard spheres using molecular dynamics, umbrella
       sampling, and forward flux sampling: A comparison of simulation
       techniques. J. Chem. Phys. 133 (244115).
       https://doi.org/10.1063/1.3506838

    Args:
        l (unsigned int):
            Spherical harmonic quantum number l.
        Q_threshold (float):
            Value of dot product threshold when evaluating
            :math:`Q_l(i, j)` to determine if a bond is solid-like. For
            :math:`l=6`, 0.7 is generally good for FCC or BCC structures
            [Filion2010]_.
        S_threshold (unsigned int):
            Minimum required number of adjacent solid-like bonds for a particle
            to be considered solid-like for clustering. For :math:`l=6`, 6-8
            is generally good for FCC or BCC structures.
        normalize_Q (bool):
            Whether to normalize the dot product (Default value =
            :code:`True`).

    Attributes:
        clusters (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed set of solid-like cluster indices for each
            particle.
        cluster_sizes (unsigned int):
            The sizes of all clusters.
        largest_cluster_size (unsigned int):
            The largest cluster size.
        num_connections (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The number of solid-like bonds for each particle.
    """  # noqa: E501
    cdef freud._order.SolidLiquid * thisptr

    def __cinit__(self, l, Q_threshold, S_threshold, normalize_Q=True):
        self.thisptr = new freud._order.SolidLiquid(
            l, Q_threshold, S_threshold, normalize_Q)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, neighbor_query, neighbors=None):
        R"""Compute the order parameter.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds.
                (Default value = :code:`None`).
        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(neighbor_query, neighbors=neighbors)
        self.thisptr.compute(nlist.get_ptr(),
                             nq.get_ptr(),
                             dereference(qargs.thisptr))

    @property
    def l(self):  # noqa: E743
        return self.thisptr.getL()

    @property
    def Q_threshold(self):
        return self.thisptr.getQThreshold()

    @property
    def S_threshold(self):
        return self.thisptr.getSThreshold()

    @property
    def normalize_Q(self):
        return self.thisptr.getNormalizeQ()

    @Compute._computed_property
    def cluster_idx(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterIdx(),
            freud.util.arr_type_t.UNSIGNED_INT)

    @Compute._computed_property
    def Ql_ij(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getQlij(),
            freud.util.arr_type_t.FLOAT)

    @Compute._computed_property
    def cluster_sizes(self):
        return np.asarray(self.thisptr.getClusterSizes())

    @Compute._computed_property
    def largest_cluster_size(self):
        return self.thisptr.getLargestClusterSize()

    @Compute._computed_property
    def nlist(self):
        return freud.locality._nlist_from_cnlist(self.thisptr.getNList())

    @Compute._computed_property
    def num_connections(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getNumberOfConnections(),
            freud.util.arr_type_t.UNSIGNED_INT)

    def __repr__(self):
        return ("freud.order.{cls}(l={sph_l}, Q_threshold={Q_threshold}, "
                "S_threshold={S_threshold}, "
                "normalize_Q={normalize_Q})").format(
                    cls=type(self).__name__,
                    sph_l=self.l,
                    Q_threshold=self.Q_threshold,
                    S_threshold=self.S_threshold,
                    normalize_Q=self.normalize_Q)

    def plot(self, ax=None):
        """Plot solid-like cluster distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis.
                (Default value = :code:`None`)

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        try:
            values, counts = np.unique(self.cluster_idx, return_counts=True)
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


cdef class RotationalAutocorrelation(Compute):
    """Calculates a measure of total rotational autocorrelation based on
    hyperspherical harmonics as laid out in "Design rules for engineering
    colloidal plastic crystals of hard polyhedra - phase behavior and
    directional entropic forces" by Karas et al. (currently in preparation).
    The output is not a correlation function, but rather a scalar value that
    measures total system orientational correlation with an initial state. As
    such, the output can be treated as an order parameter measuring degrees of
    rotational (de)correlation. For analysis of a trajectory, the compute call
    needs to be done at each trajectory frame.

    Args:
        l (int):
            Order of the hyperspherical harmonic. Must be a positive, even
            integer.

    Attributes:
        azimuthal (int):
            The azimuthal quantum number, which defines the order of the
            hyperspherical harmonic. Must be a positive, even integer.
        order (float):
            The autocorrelation computed in the last call to compute.
        particle_order ((:math:`N_{orientations}`) :class:`numpy.ndarray`):
            The per-orientation array of rotational autocorrelation values
            calculated by the last call to compute.
    """
    cdef freud._order.RotationalAutocorrelation * thisptr

    def __cinit__(self, l):
        if l % 2 or l < 0:
            raise ValueError(
                "The quantum number must be a positive, even integer.")
        self.thisptr = new freud._order.RotationalAutocorrelation(l)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, ref_orientations, orientations):
        """Calculates the rotational autocorrelation function for a single frame.

        Args:
            ref_orientations ((:math:`N_{orientations}`, 4) :class:`numpy.ndarray`):
                Reference orientations for the initial frame.
            orientations ((:math:`N_{orientations}`, 4) :class:`numpy.ndarray`):
                Orientations for the frame of interest.
        """  # noqa
        ref_orientations = freud.util._convert_array(
            ref_orientations, shape=(None, 4))
        orientations = freud.util._convert_array(
            orientations, shape=ref_orientations.shape)

        cdef const float[:, ::1] l_ref_orientations = ref_orientations
        cdef const float[:, ::1] l_orientations = orientations
        cdef unsigned int nP = orientations.shape[0]

        self.thisptr.compute(
            <quat[float]*> &l_ref_orientations[0, 0],
            <quat[float]*> &l_orientations[0, 0],
            nP)
        return self

    @Compute._computed_property
    def order(self):
        return self.thisptr.getRotationalAutocorrelation()

    @Compute._computed_property
    def particle_order(self):
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getRAArray(),
            freud.util.arr_type_t.COMPLEX_FLOAT)

    @property
    def azimuthal(self):
        return self.thisptr.getL()

    def __repr__(self):
        return "freud.order.{cls}(l={sph_l})".format(cls=type(self).__name__,
                                                     sph_l=self.azimuthal)
