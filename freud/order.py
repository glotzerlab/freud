# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.
# ruff: noqa: E731
r"""
The :class:`freud.order` module contains functions which compute order
parameters for the whole system or individual particles. Order parameters take
bond order data and interpret it in some way to quantify the degree of order in
a system using a scalar value. This is often done through computing spherical
harmonics of the bond order diagram, which are the spherical analogue of
Fourier Transforms.
"""

from freud.util import _Compute  # , quat, vec3

from freud.errors import FreudDeprecationWarning

# from cython.operator cimport dereference

from freud.locality import _PairCompute

import collections.abc
import logging
import time
import warnings

import numpy as np

import freud.locality

import numpy as np

import freud._order
# cimport freud.locality
# cimport freud.util

logger = logging.getLogger(__name__)

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
# np.import_array()

print("ADFHAFGJAKFHAJ")

# cdef class Cubatic(_Compute):
#     r"""Compute the cubatic order parameter :cite:`Haji_Akbari_2015` for a system of
#     particles using simulated annealing instead of Newton-Raphson root finding.

#     Args:
#         t_initial (float):
#             Starting temperature.
#         t_final (float):
#             Final temperature.
#         scale (float):
#             Scaling factor to reduce temperature.
#         n_replicates (unsigned int, optional):
#             Number of replicate simulated annealing runs.
#             (Default value = :code:`1`).
#         seed (unsigned int, optional):
#             Random seed to use in calculations. If :code:`None`, system time is used.
#             (Default value = :code:`None`).
#     """  # noqa: E501
#     cdef freud._order.Cubatic * thisptr

#     def __cinit__(self, t_initial, t_final, scale, n_replicates=1, seed=None):
#         if seed is None:
#             seed = int(time.time())

#         self.thisptr = new freud._order.Cubatic(
#             t_initial, t_final, scale, n_replicates, seed)

#     def __dealloc__(self):
#         del self.thisptr

#     def compute(self, orientations):
#         r"""Calculates the per-particle and global order parameter.

#         Args:
#             orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
#                 Orientations as angles to use in computation.
#         """
#         orientations = freud.util._convert_array(
#             orientations, shape=(None, 4))

#         cdef const float[:, ::1] l_orientations = orientations
#         cdef unsigned int num_particles = l_orientations.shape[0]

#         self.thisptr.compute(
#             <quat[float]*> &l_orientations[0, 0], num_particles)
#         return self

#     @property
#     def t_initial(self):
#         """float: The value of the initial temperature."""
#         return self.thisptr.getTInitial()

#     @property
#     def t_final(self):
#         """float: The value of the final temperature."""
#         return self.thisptr.getTFinal()

#     @property
#     def scale(self):
#         """float: The scale."""
#         return self.thisptr.getScale()

#     @property
#     def n_replicates(self):
#         """unsigned int: Number of replicate simulated annealing runs."""
#         return self.thisptr.getNReplicates()

#     @property
#     def seed(self):
#         """unsigned int: Random seed to use in calculations."""
#         return self.thisptr.getSeed()

#     @_Compute._computed_property
#     def order(self):
#         """float: Cubatic order parameter of the system."""
#         return self.thisptr.getCubaticOrderParameter()

#     @_Compute._computed_property
#     def orientation(self):
#         """:math:`\\left(4 \\right)` :class:`numpy.ndarray`: The quaternion of
#         global orientation."""
#         cdef quat[float] q = self.thisptr.getCubaticOrientation()
#         return np.asarray([q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)

#     @_Compute._computed_property
#     def particle_order(self):
#         """:math:`\\left(N_{particles} \\right)` :class:`numpy.ndarray`: Order
#         parameter."""
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getParticleOrderParameter(),
#             freud.util.arr_type_t.FLOAT)

#     @_Compute._computed_property
#     def global_tensor(self):
#         """:math:`\\left(3, 3, 3, 3 \\right)` :class:`numpy.ndarray`: Rank 4
#         tensor corresponding to the global orientation. Computed from all
#         orientations."""
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getGlobalTensor(),
#             freud.util.arr_type_t.FLOAT)

#     @_Compute._computed_property
#     def cubatic_tensor(self):
#         """:math:`\\left(3, 3, 3, 3 \\right)` :class:`numpy.ndarray`: Rank 4
#         homogeneous tensor representing the optimal system-wide coordinates."""
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getCubaticTensor(),
#             freud.util.arr_type_t.FLOAT)

#     def __repr__(self):
#         return ("freud.order.{cls}(t_initial={t_initial}, t_final={t_final}, "
#                 "scale={scale}, n_replicates={n_replicates}, "
#                 "seed={seed})").format(cls=type(self).__name__,
#                                        t_initial=self.t_initial,
#                                        t_final=self.t_final,
#                                        scale=self.scale,
#                                        n_replicates=self.n_replicates,
#                                        seed=self.seed)


class Nematic(_Compute):
    r"""Compute the nematic order parameter for a system of particles.

    Note:
        In some cases, such as HOOMD-blue simulations, orientations are represented as
        quaternions which rotate a particle from an initial reference orientation. To
        convert from the quaternion representation to the orientation vectors taken by
        *freud*, one should rotate the reference orientation by the particle's
        orientation quaternion. An example of this process using the *rowan* library is
        shown below.

    Example::

        >>> import rowan
        >>> import numpy as np
        >>> import freud
        >>> quats = np.array([[1, 0, 0, 0]] * 100)
        >>> starting_orientation = np.array([1, 1, 0])
        >>> orientations = rowan.rotate(quats,starting_orientation)
        >>> nematic = freud.order.Nematic()
        >>> nematic.compute(orientations)
        freud.order.Nematic()

    """

    def __init__(self):
        self._cpp_obj = freud._order.Nematic()

    def compute(self, orientations):
        r"""Calculates the per-particle and global order parameter.

        Example::

            >>> orientations = np.array([[1, 0, 0]] * 100)
            >>> nematic = freud.order.Nematic()
            >>> nematic.compute(orientations)
            freud.order.Nematic()
            >>> print(nematic.order)
            1.0

        Args:
            orientations (:math:`\left(N_{particles}, 3 \right)` :class:`numpy.ndarray`):
                Orientation vectors for which to calculate the order parameter.
        """  # noqa: E501
        if orientations.shape[1] == 4:
            raise ValueError(
                "In freud versions >=3.0.0, Nematic.compute() takes "
                "3d orientation vectors instead of 4d quaternions."
            )
        orientations = freud.util._convert_array(orientations, shape=(None, 3))

        if len(np.where(~orientations.any(axis=1))[0]) != 0:
            warnings.warn(
                "Including zero vector in the orientations array "
                "may lead to undefined behavior.",
                UserWarning,
            )

        self._cpp_obj.compute(orientations)
        return self

    @_Compute._computed_property
    def order(self):
        """float: Nematic order parameter of the system."""
        return self._cpp_obj.getNematicOrderParameter()

    @_Compute._computed_property
    def director(self):
        """:math:`\\left(3 \\right)` :class:`numpy.ndarray`: The average
        nematic director."""
        return np.asarray([*self._cpp_obj.getNematicDirector()], dtype=np.float32)

    @_Compute._computed_property
    def particle_tensor(self):
        """:math:`\\left(N_{particles}, 3, 3 \\right)` :class:`numpy.ndarray`:
        One 3x3 matrix per-particle corresponding to each individual
        particle orientation."""
        return self._cpp_obj.getParticleTensor().toNumpyArray()

    @_Compute._computed_property
    def nematic_tensor(self):
        """:math:`\\left(3, 3 \\right)` :class:`numpy.ndarray`: 3x3 matrix
        corresponding to the average particle orientation."""
        return self._cpp_obj.getNematicTensor().toNumpyArray()

    def __repr__(self):
        return f"freud.order.{type(self).__name__}()"


# cdef class Hexatic(_PairCompute):
#     r"""Calculates the :math:`k`-atic order parameter for 2D systems.

#     The :math:`k`-atic order parameter (called the hexatic order parameter for
#     :math:`k = 6`) is analogous to Steinhardt order parameters, and is used to
#     measure order in the bonds of 2D systems.

#     The :math:`k`-atic order parameter for a particle :math:`i` and its
#     :math:`N_b` neighbors :math:`j` is given by:

#     :math:`\psi_k \left( i \right) = \frac{1}{N_b}
#     \sum \limits_{j=1}^{N_b} e^{i k \phi_{ij}}`

#     The parameter :math:`k` governs the symmetry of the order parameter and
#     typically matches the number of neighbors to be found for each particle.
#     The quantity :math:`\phi_{ij}` is the angle between the
#     vector :math:`r_{ij}` and :math:`\left(1, 0\right)`.

#     If the weighted mode is enabled, contributions of each neighbor are
#     weighted. Neighbor weights :math:`w_{ij}` default to 1 but are defined for a
#     :class:`freud.locality.NeighborList` from :class:`freud.locality.Voronoi`
#     or one with user-provided weights. The formula is modified as follows:

#     :math:`\psi'_k \left( i \right) = \frac{1}{\sum_{j=1}^{N_b} w_{ij}}
#     \sum \limits_{j=1}^{N_b} w_{ij} e^{i k \phi_{ij}}`

#     The hexatic order parameter as written above is **complex-valued**. The
#     **magnitude** of the complex value,
#     :code:`np.abs(hex_order.particle_order)`, is frequently what is desired
#     when determining the :math:`k`-atic order for each particle. The complex
#     phase angle :code:`np.angle(hex_order.particle_order)` indicates the
#     orientation of the bonds as an angle measured counterclockwise from the
#     vector :math:`\left(1, 0\right)`. The complex valued order parameter is
#     not rotationally invariant because of this phase angle, but the magnitude
#     *is* rotationally invariant.

#     .. note::
#         **2D:** :class:`freud.order.Hexatic` is only defined for 2D systems.
#         The points must be passed in as :code:`[x, y, 0]`.

#     Args:
#         k (unsigned int, optional):
#             Symmetry of order parameter (Default value = :code:`6`).
#         weighted (bool, optional):
#             Determines whether to use neighbor weights in the computation of
#             spherical harmonics over neighbors. If enabled and used with a
#             Voronoi neighbor list, this results in the 2D Minkowski Structure
#             Metrics :math:`\psi'_k` :cite:`Mickel2013` (Default value =
#             :code:`False`).
#     """  # noqa: E501
#     cdef freud._order.Hexatic * thisptr

#     def __cinit__(self, k=6, weighted=False):
#         self.thisptr = new freud._order.Hexatic(k, weighted)

#     def __dealloc__(self):
#         del self.thisptr

#     def compute(self, system, neighbors=None):
#         r"""Calculates the hexatic order parameter.

#         Example::

#             >>> box, points = freud.data.make_random_system(
#             ...     box_size=10, num_points=100, is2D=True, seed=0)
#             >>> # Compute the hexatic (6-fold) order for the 2D system
#             >>> hex_order = freud.order.Hexatic(k=6)
#             >>> hex_order.compute(system=(box, points))
#             freud.order.Hexatic(...)
#             >>> print(hex_order.particle_order)
#             [...]

#         Args:
#             system:
#                 Any object that is a valid argument to
#                 :class:`freud.locality.NeighborQuery.from_system`.
#             neighbors (:class:`freud.locality.NeighborList` or dict, optional):
#                 Either a :class:`NeighborList <freud.locality.NeighborList>` of
#                 neighbor pairs to use in the calculation, or a dictionary of
#                 `query arguments
#                 <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
#                 (Default value: None).
#         """   # noqa: E501
#         cdef:
#             freud.locality.NeighborQuery nq
#             freud.locality.NeighborList nlist
#             freud.locality._QueryArgs qargs
#             const float[:, ::1] l_query_points
#             unsigned int num_query_points

#         nq, nlist, qargs, l_query_points, num_query_points = \
#             self._preprocess_arguments(system, neighbors=neighbors)
#         self.thisptr.compute(nlist.get_ptr(),
#                              nq.get_ptr(), dereference(qargs.thisptr))
#         return self

#     @property
#     def default_query_args(self):
#         """The default query arguments are
#         :code:`{'mode': 'nearest', 'num_neighbors': self.k}`."""
#         return dict(mode="nearest", num_neighbors=self.k)

#     @_Compute._computed_property
#     def particle_order(self):
#         """:math:`\\left(N_{particles} \\right)` :class:`numpy.ndarray`: Order
#         parameter."""
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getOrder(),
#             freud.util.arr_type_t.COMPLEX_FLOAT)

#     @property
#     def k(self):
#         """unsigned int: Symmetry of the order parameter."""
#         return self.thisptr.getK()

#     @property
#     def weighted(self):
#         """bool: Whether neighbor weights were used in the computation."""
#         return self.thisptr.isWeighted()

#     def __repr__(self):
#         return "freud.order.{cls}(k={k}, weighted={weighted})".format(
#             cls=type(self).__name__, k=self.k, weighted=self.weighted)

#     def plot(self, ax=None):
#         """Plot order parameter distribution.

#         Args:
#             ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
#                 :code:`None`, make a new figure and axis
#                 (Default value = :code:`None`).

#         Returns:
#             (:class:`matplotlib.axes.Axes`): Axis with the plot.
#         """
#         import freud.plot
#         xlabel = r"$\left|\psi{prime}_{k}\right|$".format(
#             prime='\'' if self.weighted else '',
#             k=self.k)

#         return freud.plot.histogram_plot(
#             np.absolute(self.particle_order),
#             title="Hexatic Order Parameter " + xlabel,
#             xlabel=xlabel,
#             ylabel=r"Number of particles",
#             ax=ax)

#     def _repr_png_(self):
#         try:
#             import freud.plot
#             return freud.plot._ax_to_bytes(self.plot())
#         except (AttributeError, ImportError):
#             return None


class Steinhardt(_PairCompute):
    r"""Compute one or more of the rotationally invariant Steinhardt order
    parameter :math:`q_l` or :math:`w_l` for a set of points
    :cite:`Steinhardt:1983aa`.

    Implements the local rotationally invariant :math:`q_l` or :math:`w_l`
    order parameter described by Steinhardt.

    First, we describe the computation of :math:`q_l(i)`.  For a particle :math:`i`,
    we calculate the quantity :math:`q_{lm}` by summing the spherical harmonics
    between particle :math:`i` and its neighbors :math:`j` in a local region:

    .. math::

        q_{lm}(i) = \frac{1}{N_b} \sum \limits_{j=1}^{N_b}
        Y_{lm}(\theta(\vec{r}_{ij}), \phi(\vec{r}_{ij}))

    Then the :math:`q_l` order parameter is computed by combining the :math:`q_{lm}`
    in a rotationally invariant fashion to remove local orientational order:

    .. math::

        q_l(i) = \sqrt{\frac{4\pi}{2l+1} \sum \limits_{m=-l}^{l}
        |q_{lm}(i)|^2 }

    If the ``wl`` parameter is ``True``, this class computes the quantity
    :math:`w_l`, defined as a weighted average over the
    :math:`q_{lm}(i)` values using `Wigner 3-j symbols
    <https://en.wikipedia.org/wiki/3-j_symbol>`__ (related to `Clebsch-Gordan
    coefficients
    <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`__).
    The resulting combination is rotationally invariant:

    .. math::

        w_l(i) = \sum \limits_{m_1 + m_2 + m_3 = 0} \begin{pmatrix}
            l & l & l \\
            m_1 & m_2 & m_3
        \end{pmatrix}
        q_{lm_1}(i) q_{lm_2}(i) q_{lm_3}(i)

    If ``wl`` is ``True``, then setting the ``wl_normalize`` parameter to ``True`` will
    normalize the :math:`w_l` order parameter as follows (if ``wl=False``,
    ``wl_normalize`` has no effect):

    .. math::

        w_l(i) = \frac{
            \sum \limits_{m_1 + m_2 + m_3 = 0} \begin{pmatrix}
                l & l & l \\
                m_1 & m_2 & m_3
            \end{pmatrix}
            q_{lm_1}(i) q_{lm_2}(i) q_{lm_3}(i)}
            {\left(\sum \limits_{m=-l}^{l} |q_{lm}(i)|^2 \right)^{3/2}}

    If ``average`` is ``True``, the class computes a variant of this order
    parameter that performs an average over the first and second shell combined
    :cite:`Lechner_2008`. To compute this parameter, we perform a second
    averaging over the first neighbor shell of the particle to implicitly
    include information about the second neighbor shell. This averaging is
    performed by replacing the value :math:`q_{lm}(i)` in the original
    definition by :math:`\overline{q}_{lm}(i)`, the average value of
    :math:`q_{lm}(k)` over all the :math:`N_b` neighbors :math:`k`
    of particle :math:`i`, including particle :math:`i` itself:

    .. math::
        \overline{q}_{lm}(i) = \frac{1}{N_b} \sum \limits_{k=0}^{N_b}
        q_{lm}(k)

    If ``weighted`` is True, the contributions of each neighbor are weighted.
    Neighbor weights :math:`w_{ij}` are defined for a
    :class:`freud.locality.NeighborList` obtained from
    :class:`freud.locality.Voronoi` or one with user-provided weights, and
    default to 1 if not otherwise provided. The formulas are modified as
    follows, replacing :math:`q_{lm}(i)` with the weighted value
    :math:`q'_{lm}(i)`:

    .. math::

        q'_{lm}(i) = \frac{1}{\sum_{j=1}^{N_b} w_{ij}}
        \sum \limits_{j=1}^{N_b} w_{ij} Y_{lm}(\theta(\vec{r}_{ij}),
        \phi(\vec{r}_{ij}))

    .. note::
        The value of per-particle order parameter will be set to NaN for
        particles with no neighbors. We choose this value rather than setting
        the order parameter to 0 because in more complex order parameter
        calculations (such as when computing the :math:`w_l`), it is possible
        to observe a value of 0 for the per-particle order parameter even with
        a finite number of neighbors. If you would like to ignore this
        distinction, you can mask the output order parameter values using
        NumPy: :code:`numpy.nan_to_num(particle_order)`.

    Args:
        l (unsigned int or sequence of unsigned int):
            One or more spherical harmonic quantum number l's used to compute
            the Steinhardt order parameter.
        average (bool, optional):
            Determines whether to calculate the averaged Steinhardt order
            parameter (Default value = :code:`False`).
        wl (bool, optional):
            Determines whether to use the :math:`w_l` version of the Steinhardt
            order parameter (Default value = :code:`False`).
        weighted (bool, optional):
            Determines whether to use neighbor weights in the computation of
            spherical harmonics over neighbors. If enabled and used with a
            Voronoi neighbor list, this results in the 3D Minkowski Structure
            Metrics :math:`q'_l` :cite:`Mickel2013` (Default value =
            :code:`False`).
        wl_normalize (bool, optional):
            Determines whether to normalize the :math:`w_l` version
            of the Steinhardt order parameter (Default value = :code:`False`).
    """

    def __init__(self, l, average=False, wl=False, weighted=False, wl_normalize=False):
        if not isinstance(l, collections.abc.Sequence):
            l = [l]
        if len(l) == 0:
            raise ValueError("At least one l must be specified.")
        self._cpp_obj = freud._order.Steinhardt(l, average, wl, weighted, wl_normalize)


    @property
    def average(self):
        """bool: Whether the averaged Steinhardt order parameter was
        calculated."""
        return self._cpp_obj.isAverage()

    @property
    def wl(self):
        """bool: Whether the :math:`w_l` version of the Steinhardt order
        parameter was used."""
        return self._cpp_obj.isWl()

    @property
    def weighted(self):
        """bool: Whether neighbor weights were used in the computation."""
        return self._cpp_obj.isWeighted()

    @property
    def wl_normalize(self):
        return self._cpp_obj.isWlNormalized()

    @property
    def l(self):  # noqa: E743
        """unsigned int: Spherical harmonic quantum number l."""
        ls = self._cpp_obj.getL()
        return ls[0] if len(ls) == 1 else ls

    @_Compute._computed_property
    def order(self):
        r"""float: The system wide normalization of the order parameter,
        computed by averaging the :math:`q_{lm}` values (or
        :math:`\overline{q}_{lm}` values if ``average`` is enabled) over all
        particles before computing the rotationally-invariant order
        parameter."""
        # list conversion is necessary as otherwise CI Cython complains about
        # compiling the below expression with two different types.
        order = self._cpp_obj.getOrder()
        return order[0] if len(order) == 1 else order

    @_Compute._computed_property
    def particle_order(self):
        """:math:`\\left(N_{particles}, N_l \\right)` :class:`numpy.ndarray`:
        Variant of the Steinhardt order parameter for each particle (filled with
        :code:`nan` for particles with no neighbors)."""
        particle_orders = self._cpp_obj.getParticleOrder().toNumpyArray()
        if particle_orders.shape[1] == 1:
            return np.ravel(particle_orders)
        return particle_orders

    @_Compute._computed_property
    def ql(self):
        """:math:`\\left(N_{particles}, N_l\\right)` :class:`numpy.ndarray`:
        :math:`q_l` Steinhardt order parameter for each particle (filled with
        :code:`nan` for particles with no neighbors). This is always available,
        no matter which other options are selected. It obeys the ``weighted``
        argument but otherwise returns the "plain" :math:`q_l` regardless of
        ``average``, ``wl``, ``wl_normalize``."""
        ql = self._cpp_obj.getQl().toNumpyArray()
        return np.ravel(ql) if ql.shape[1] == 1 else ql
        # array = freud.util.make_managed_numpy_array(
        #     &self.thisptr.getQl(), freud.util.arr_type_t.FLOAT)
        # if array.shape[1] == 1:
        #     return np.ravel(array)
        # return ql

    @_Compute._computed_property
    def particle_harmonics(self):
        """:math:`\\left(N_{particles}, 2l+1\\right)` :class:`numpy.ndarray`:
        The raw array of :math:`q_{lm}(i)`. The array is provided in the
        order given by fsph: :math:`m = 0, 1, ..., l, -1, ..., -l`."""
        qlm_list = [qlm.toNumpyArray() for qlm in self._cpp_obj.getQlm()]
        return qlm_list if len(qlm_list) > 1 else qlm_list[0]

    def compute(self, system, neighbors=None):
        r"""Compute the order parameter.

        Example::

            >>> box, points = freud.data.make_random_system(10, 100, seed=0)
            >>> ql = freud.order.Steinhardt(l=6)
            >>> ql.compute((box, points), {'r_max':3})
            freud.order.Steinhardt(l=6, ...)

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """

        # Call the pair compute setup function
        nq, nlist, qargs, l_query_points, num_query_points = self._preprocess_arguments(system, neighbors=neighbors)

        self._cpp_obj.compute(nlist._cpp_obj, nq._cpp_obj, qargs._cpp_obj)
        return self

    def __repr__(self):
        return (f"freud.order.{type(self).__name__}(l={self.l}, "
                f"average={self.average}, wl={self.wl}, weighted={self.weighted}, "
                f"wl_normalize={self.wl_normalize})")

#     def plot(self, ax=None):
#         """Plot order parameter distribution.

#         Args:
#             ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
#                 :code:`None`, make a new figure and axis
#                 (Default value = :code:`None`).

#         Returns:
#             (:class:`matplotlib.axes.Axes`): Axis with the plot.
#         """
#         import freud.plot

#         ls = self.l
#         if not isinstance(ls, list):
#             ls = [ls]

#         legend_labels = [
#             r"${mode_letter}{prime}_{{{sph_l}{average}}}$".format(
#                 mode_letter='w' if self.wl else 'q',
#                 prime='\'' if self.weighted else '',
#                 sph_l=sph_l,
#                 average=',ave' if self.average else '')
#             for sph_l in ls
#         ]
#         xlabel = ', '.join(legend_labels)

#         # Don't print legend if only one l requested.
#         if len(legend_labels) == 1:
#             legend_labels = None

#         return freud.plot.histogram_plot(
#             self.particle_order,
#             title="Steinhardt Order Parameter " + xlabel,
#             xlabel=xlabel,
#             ylabel=r"Number of particles",
#             ax=ax,
#             legend_labels=legend_labels)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


# cdef class SolidLiquid(_PairCompute):
#     r"""Identifies solid-like clusters using dot products of :math:`q_{lm}`.

#     The solid-liquid order parameter :cite:`Wolde:1995aa,Filion_2010` uses a
#     Steinhardt-like approach to identify solid-like particles. First, a bond
#     parameter :math:`q_l(i, j)` is computed for each neighbor bond.

#     If :code:`normalize_q` is true (default), the bond parameter is given by
#     :math:`q_l(i, j) = \frac{\sum \limits_{m=-l}^{l} \text{Re}~q_{lm}(i) q_{lm}^*(j)}
#     {\sqrt{\sum \limits_{m=-l}^{l} \lvert q_{lm}(i) \rvert^2}
#     \sqrt{\sum \limits_{m=-l}^{l} \lvert q_{lm}(j) \rvert^2}}`

#     If :code:`normalize_q` is false, then the denominator of the above
#     expression is left out.

#     Next, the bonds are filtered to keep only "solid-like" bonds with
#     :math:`q_l(i, j)` above a cutoff value :math:`q_{threshold}`.

#     If a particle has more than :math:`S_{threshold}` solid-like bonds, then
#     the particle is considered solid-like. Finally, solid-like particles are
#     clustered.

#     Args:
#         l (unsigned int):
#             Spherical harmonic quantum number l.
#         q_threshold (float):
#             Value of dot product threshold when evaluating
#             :math:`q_l(i, j)` to determine if a bond is solid-like. For
#             :math:`l=6`, 0.7 is generally good for FCC or BCC structures
#             :cite:`Filion_2010`.
#         solid_threshold (unsigned int):
#             Minimum required number of adjacent solid-like bonds for a particle
#             to be considered solid-like for clustering. For :math:`l=6`, 6-8
#             is generally good for FCC or BCC structures.
#         normalize_q (bool):
#             Whether to normalize the dot product (Default value =
#             :code:`True`).
#     """  # noqa: E501
#     cdef freud._order.SolidLiquid * thisptr

#     def __cinit__(self, l, q_threshold, solid_threshold, normalize_q=True):
#         self.thisptr = new freud._order.SolidLiquid(
#             l, q_threshold, solid_threshold, normalize_q)

#     def __dealloc__(self):
#         del self.thisptr

#     def compute(self, system, neighbors=None):
#         r"""Compute the order parameter.

#         Args:
#             system:
#                 Any object that is a valid argument to
#                 :class:`freud.locality.NeighborQuery.from_system`.
#             neighbors (:class:`freud.locality.NeighborList` or dict, optional):
#                 Either a :class:`NeighborList <freud.locality.NeighborList>` of
#                 neighbor pairs to use in the calculation, or a dictionary of
#                 `query arguments
#                 <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
#                 (Default value: None).
#         """
#         cdef:
#             freud.locality.NeighborQuery nq
#             freud.locality.NeighborList nlist
#             freud.locality._QueryArgs qargs
#             const float[:, ::1] l_query_points
#             unsigned int num_query_points

#         nq, nlist, qargs, l_query_points, num_query_points = \
#             self._preprocess_arguments(system, neighbors=neighbors)
#         self.thisptr.compute(nlist.get_ptr(),
#                              nq.get_ptr(),
#                              dereference(qargs.thisptr))
#         return self

#     @property
#     def l(self):  # noqa: E743
#         """unsigned int: Spherical harmonic quantum number l."""
#         return self.thisptr.getL()

#     @property
#     def q_threshold(self):
#         """float: Value of dot product threshold."""
#         return self.thisptr.getQThreshold()

#     @property
#     def solid_threshold(self):
#         """float: Value of number-of-bonds threshold."""
#         return self.thisptr.getSolidThreshold()

#     @property
#     def normalize_q(self):
#         """bool: Whether the dot product is normalized."""
#         return self.thisptr.getNormalizeQ()

#     @_Compute._computed_property
#     def cluster_idx(self):
#         """:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
#         Solid-like cluster indices for each particle."""
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getClusterIdx(),
#             freud.util.arr_type_t.UNSIGNED_INT)

#     @_Compute._computed_property
#     def ql_ij(self):
#         """:math:`\\left(N_{bonds}\\right)` :class:`numpy.ndarray`: Bond dot
#         products :math:`q_l(i, j)`. Indexed by the elements of
#         :code:`self.nlist`."""
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getQlij(),
#             freud.util.arr_type_t.FLOAT)

#     @_Compute._computed_property
#     def particle_harmonics(self):
#         """:math:`\\left(N_{particles}, 2*l+1\\right)` :class:`numpy.ndarray`:
#         The raw array of \\overline{q}_{lm}(i). The array is provided in the
#         order given by fsph: :math:`m = 0, 1, ..., l, -1, ..., -l`."""
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getQlm(),
#             freud.util.arr_type_t.COMPLEX_FLOAT)

#     @_Compute._computed_property
#     def cluster_sizes(self):
#         """:math:`(N_{clusters}, )` :class:`np.ndarray`: The sizes of all
#         clusters."""
#         return np.asarray(self.thisptr.getClusterSizes())

#     @_Compute._computed_property
#     def largest_cluster_size(self):
#         """unsigned int: The largest cluster size."""
#         return self.thisptr.getLargestClusterSize()

#     @_Compute._computed_property
#     def nlist(self):
#         """:class:`freud.locality.NeighborList`: Neighbor list of solid-like
#         bonds."""
#         nlist = freud.locality._nlist_from_cnlist(self.thisptr.getNList())
#         nlist._compute = self
#         return nlist

#     @_Compute._computed_property
#     def num_connections(self):
#         """:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`: The
#         number of solid-like bonds for each particle."""
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getNumberOfConnections(),
#             freud.util.arr_type_t.UNSIGNED_INT)

#     def __repr__(self):
#         return ("freud.order.{cls}(l={sph_l}, q_threshold={q_threshold}, "
#                 "solid_threshold={solid_threshold}, "
#                 "normalize_q={normalize_q})").format(
#                     cls=type(self).__name__,
#                     sph_l=self.l,
#                     q_threshold=self.q_threshold,
#                     solid_threshold=self.solid_threshold,
#                     normalize_q=self.normalize_q)

#     def plot(self, ax=None):
#         """Plot solid-like cluster distribution.

#         Args:
#             ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
#                 :code:`None`, make a new figure and axis
#                 (Default value = :code:`None`).

#         Returns:
#             (:class:`matplotlib.axes.Axes`): Axis with the plot.
#         """
#         import freud.plot
#         try:
#             values, counts = np.unique(self.cluster_idx, return_counts=True)
#         except ValueError:
#             return None
#         else:
#             return freud.plot.clusters_plot(
#                 values, counts, num_clusters_to_plot=10, ax=ax)

#     def _repr_png_(self):
#         try:
#             import freud.plot
#             return freud.plot._ax_to_bytes(self.plot())
#         except (AttributeError, ImportError):
#             return None


class RotationalAutocorrelation(_Compute):
    """Calculates a measure of total rotational autocorrelation.

    For any calculation of rotational correlations of extended (i.e. non-point)
    particles, encoding the symmetries of these particles is crucial to
    appropriately determining correlations. For systems of anisotropic
    particles in three dimensions, representing such equivalence can be quite
    mathematically challenging. This calculation is based on the hyperspherical
    harmonics as laid out in :cite:`Karas2019`. Generalizations of spherical
    harmonics to four dimensions, hyperspherical harmonics provide a natural
    basis for periodic functions in 4 dimensions just as harmonic functions
    (sines and cosines) or spherical harmonics do in lower dimensions. The idea
    behind this calculation is to embed orientation quaternions into a
    4-dimensional space and then use hyperspherical harmonics to find
    correlations in a symmetry-aware fashion.

    The choice of the hyperspherical harmonic parameter :math:`l` determines
    the symmetry of the functions. The output is not a correlation function,
    but rather a scalar value that measures total system orientational
    correlation with an initial state. As such, the output can be treated as an
    order parameter measuring degrees of rotational (de)correlation. For
    analysis of a trajectory, the compute call needs to be
    done at each trajectory frame.

    Args:
        l (int):
            Order of the hyperspherical harmonic. Must be a positive, even
            integer.
    """

    def __init__(self, l):  # noqa: E741
        if l % 2 or l < 0:
            raise ValueError("The quantum number must be a positive, even integer.")  # noqa: EM101
        self._cpp_obj = freud._order.RotationalAutocorrelation(l)

    def compute(self, ref_orientations, orientations):
        """Calculates the rotational autocorrelation function for a single frame.

        Args:
            ref_orientations ((:math:`N_{orientations}`, 4) :class:`numpy.ndarray`):
                Orientations for the initial frame.
            orientations ((:math:`N_{orientations}`, 4) :class:`numpy.ndarray`):
                Orientations for the frame of interest.
        """
        assert len(ref_orientations) == len(
            orientations
        ), "orientations and ref_orientations must have the same shape."
        self._cpp_obj.compute(ref_orientations, orientations)
        return self

    @_Compute._computed_property
    def order(self):
        """float: Autocorrelation of the system."""
        return self._cpp_obj.getRotationalAutocorrelation()

    @_Compute._computed_property
    def particle_order(self):
        """(:math:`N_{orientations}`) :class:`numpy.ndarray`: Rotational
        autocorrelation values calculated for each orientation."""
        return self._cpp_obj.getRAArray().toNumpyArray()

    @property
    def l(self):  # noqa: E743
        """int: The azimuthal quantum number, which defines the order of the
        hyperspherical harmonic."""
        return self._cpp_obj.getL()

    def __repr__(self):
        return f"freud.order.{type(self).__name__}(l={self.l})"


# cdef class ContinuousCoordination(_PairCompute):
#     r"""Computes the continuous local coordination number.

#     The :class:`ContinuousCoordination` class implements extensions of the Voronoi
#     discrete coordination number to the real numbers. The formulas for the
#     various implementations are:

#     Power:

#     .. math::

#         CN_p = N^{2.0 - m} \sum_{i=1}^{k}
#         {\left[\left(\frac{V_i}{V}\right)^{m}\right]}^{-1}

#     Log:

#     .. math::

#         CN_{log} = \frac{-1}{\log{N}} \sum_{i=1}^{k}\log{\left(\frac{V_i}{V}\right)}

#     Exponential:

#     .. math::

#         CN_{exp} = \sum_{i=1}^{k}\exp{\left(\frac{V_i}{V} - \frac{1}{N} \right)}

#     where :math:`k` is the number of neighbors a particle has, :math:`V_i` is
#     the volume of the pyramid (or area of the triangle in 2D) whose base is the
#     Voronoi polytope facet between the central particle and neighbor :math:`i`
#     and whose height is half the distance vector, and :math:`V` is the
#     volume/area of the Voronoi polytope.

#     Note:
#         When using multiple powers, space them out to avoid high correlations. A
#         minimum spacing of 2.0 is recommended with even larger values leading to
#         less correlation.

#     Args:
#         powers (list[float], optional): The powers to compute the continuous
#             coordination number for. The default value indicates only compute
#             for power 2.0.
#             (Default value: None)
#         compute_log (`bool`, optional): Whether to compute the log continuous
#             coordination number.
#             (Default value: :code:`True`)
#         compute_exp (`bool`, optional): Whether to compute the exp continuous
#             coordination number.
#             (Default value: :code:`True`)
#     """
#     cdef freud._order.ContinuousCoordination* thisptr

#     def __cinit__(self, powers=None, compute_log=True, compute_exp=True):
#         if powers is None:
#             powers = [2.0]
#         self.thisptr = new freud._order.ContinuousCoordination(
#             powers, compute_log, compute_exp)

#     def __dealloc__(self):
#         del self.thisptr

#     def compute(self, system=None, voronoi=None):
#         r"""Calculates the local coordination number for the specified points.

#         Example::

#             >>> import freud
#             >>> box, points = freud.data.make_random_system(10, 100, seed=0)
#             >>> # Compute ContinuousCoordination
#             >>> coord = freud.order.ContinuousCoordination([2, 4], True)
#             >>> coord.compute(system=(box, points))
#             freud.order.ContinuousCoordination(...)

#         Args:
#             system (optional):
#                 Any object that is a valid argument to
#                 :class:`freud.locality.NeighborQuery.from_system`.
#                 (Default value: None).
#             voronoi (:class:`freud.locality.Voronoi`, optional):
#                 A precomputed Voronoi compute object. If provided, the object is
#                 assumed to have been computed already, and system is ignored.
#                 (Default value: None).
#         """
#         cdef freud.locality.Voronoi cpp_voronoi
#         if system is None and voronoi is None:
#             raise ValueError("Must specify system or voronoi.")
#         if voronoi is None:
#             voronoi = freud.locality.Voronoi()
#             voronoi.compute(system)
#         elif not hasattr(voronoi, "nlist"):
#             raise RuntimeError(
#                 "Must call compute on Voronoi object prior to computing coordination.")
#         cpp_voronoi = voronoi
#         self.thisptr.compute(cpp_voronoi.thisptr)
#         return self

#     @_Compute._computed_property
#     def particle_order(self):
#         """(:math:`(N_{points}, N_{coord}`) :class:`numpy.ndarray`: \
#                 coordination of points per query point.

#         Coordination numbers are in order of selected powers, log, and exp.
#         """
#         return self.coordination

#     @_Compute._computed_property
#     def coordination(self):
#         """(:math:`(N_{points}, N_{coord}`) :class:`numpy.ndarray`): \
#                 coordination of points per query point.

#         Coordination numbers are in order of selected powers, log, and exp.
#         """
#         return freud.util.make_managed_numpy_array(
#             &self.thisptr.getCoordination(),
#             freud.util.arr_type_t.FLOAT)

#     @property
#     def powers(self):
#         """list[float]: The powers to compute the continuous coordination number.

#         Changes to this property are not reflected when computing coordination
#         numbers.
#         """
#         return self.thisptr.getPowers()

#     @property
#     def compute_log(self):
#         """bool: Whether to compute the log continuous coordination number."""
#         return self.thisptr.getComputeLog()

#     @property
#     def compute_exp(self):
#         """bool: Whether to compute the exponential coordination number."""
#         return self.thisptr.getComputeExp()

#     @property
#     def number_of_coordinations(self):
#         """int: The number of coordination numbers computed."""
#         return self.thisptr.getNumberOfCoordinations()

#     def __repr__(self):
#         return (
#             "freud.order.{cls}(powers={powers}, "
#             "compute_log={compute_log}, compute_exp={compute_exp})"
#         ).format(
#             cls=type(self).__name__,
#             powers=self.powers,
#             compute_log=self.compute_log,
#             compute_exp=self.compute_exp,
#         )
