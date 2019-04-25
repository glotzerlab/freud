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

import freud.common
import warnings
import numpy as np
import time
import freud.locality
import logging

from freud.util._VectorMath cimport vec3, quat
from cython.operator cimport dereference

# The below are maintained for backwards compatibility
# but have been moved to the environment module
from freud.environment cimport BondOrder as _EBO
from freud.environment cimport LocalDescriptors as _ELD
from freud.environment cimport MatchEnv as _EME
from freud.environment cimport AngularSeparation as _EAS

cimport freud._order
cimport freud.locality
cimport freud.box

cimport numpy as np

logger = logging.getLogger(__name__)

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class CubaticOrderParameter:
    R"""Compute the cubatic order parameter [HajiAkbari2015]_ for a system of
    particles using simulated annealing instead of Newton-Raphson root finding.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        t_initial (float):
            Starting temperature.
        t_final (float):
            Final temperature.
        scale (float):
            Scaling factor to reduce temperature.
        n_replicates (unsigned int):
            Number of replicate simulated annealing runs.
        seed (unsigned int):
            Random seed to use in calculations. If None, system time is used.

    Attributes:
        t_initial (float):
            The value of the initial temperature.
        t_final (float):
            The value of the final temperature.
        scale (float):
            The scale
        cubatic_order_parameter (float):
            The cubatic order parameter.
        orientation (:math:`\left(4 \right)` :class:`numpy.ndarray`):
            The quaternion of global orientation.
        particle_order_parameter (:class:`numpy.ndarray`):
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
    cdef freud._order.CubaticOrderParameter * thisptr
    cdef t_initial
    cdef t_final
    cdef scale
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

        # for c++ code
        # create generalized rank four tensor, pass into c++
        cdef float[:, ::1] kd = np.eye(3, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] dijkl = np.einsum(
            "ij,kl->ijkl", kd, kd, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] dikjl = np.einsum(
            "ik,jl->ijkl", kd, kd, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] diljk = np.einsum(
            "il,jk->ijkl", kd, kd, dtype=np.float32)
        cdef float[:, :, :, ::1] r4 = (dijkl + dikjl + diljk) * (2.0/5.0)
        self.thisptr = new freud._order.CubaticOrderParameter(
            t_initial, t_final, scale, <float*> &r4[0, 0, 0, 0], n_replicates,
            seed)
        self.t_initial = t_initial
        self.t_final = t_final
        self.scale = scale
        self.n_replicates = n_replicates
        self.seed = seed

    def compute(self, orientations):
        R"""Calculates the per-particle and global order parameter.

        Args:
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Orientations as angles to use in computation.
        """
        orientations = freud.common.convert_array(
            orientations, 2, dtype=np.float32, contiguous=True,
            array_name="orientations")
        if orientations.shape[1] != 4:
            raise TypeError('orientations should be an Nx4 array')

        cdef float[:, ::1] l_orientations = orientations
        cdef unsigned int num_particles = l_orientations.shape[0]

        with nogil:
            self.thisptr.compute(
                <quat[float]*> &l_orientations[0, 0], num_particles, 1)
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
    def cubatic_order_parameter(self):
        return self.thisptr.getCubaticOrderParameter()

    @property
    def orientation(self):
        cdef quat[float] q = self.thisptr.getCubaticOrientation()
        return np.asarray([q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)

    @property
    def particle_order_parameter(self):
        cdef unsigned int n_particles = self.thisptr.getNumParticles()
        cdef float[::1] particle_order_parameter = \
            <float[:n_particles]> \
            self.thisptr.getParticleCubaticOrderParameter().get()
        return np.asarray(particle_order_parameter)

    @property
    def particle_tensor(self):
        cdef unsigned int n_particles = self.thisptr.getNumParticles()
        cdef float[:, :, :, :, ::1] particle_tensor = \
            <float[:n_particles, :3, :3, :3, :3]> \
            self.thisptr.getParticleTensor().get()
        return np.asarray(particle_tensor)

    @property
    def global_tensor(self):
        cdef float[:, :, :, ::1] global_tensor = \
            <float[:3, :3, :3, :3]> \
            self.thisptr.getGlobalTensor().get()
        return np.asarray(global_tensor)

    @property
    def cubatic_tensor(self):
        cdef float[:, :, :, ::1] cubatic_tensor = \
            <float[:3, :3, :3, :3]> \
            self.thisptr.getCubaticTensor().get()
        return np.asarray(cubatic_tensor)

    @property
    def gen_r4_tensor(self):
        cdef float[:, :, :, ::1] gen_r4_tensor = \
            <float[:3, :3, :3, :3]> \
            self.thisptr.getGenR4Tensor().get()
        return np.asarray(gen_r4_tensor)

    def __repr__(self):
        return ("freud.order.{cls}(t_initial={t_initial}, t_final={t_final}, "
                "scale={scale}, n_replicates={n_replicates}, "
                "seed={seed})").format(cls=type(self).__name__,
                                       t_initial=self.t_initial,
                                       t_final=self.t_final,
                                       scale=self.scale,
                                       n_replicates=self.n_replicates,
                                       seed=self.seed)

    def __str__(self):
        return repr(self)

cdef class NematicOrderParameter:
    R"""Compute the nematic order parameter for a system of particles.

    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    .. versionadded:: 0.7.0

    Args:
        u (:math:`\left(3 \right)` :class:`numpy.ndarray`):
            The nematic director of a single particle in the reference state
            (without any rotation applied).

    Attributes:
        nematic_order_parameter (float):
            Nematic order parameter.
        director (:math:`\left(3 \right)` :class:`numpy.ndarray`):
            The average nematic director.
        particle_tensor (:math:`\left(N_{particles}, 3, 3 \right)` :class:`numpy.ndarray`):
            One 3x3 matrix per-particle corresponding to each individual
            particle orientation.
        nematic_tensor (:math:`\left(3, 3 \right)` :class:`numpy.ndarray`):
            3x3 matrix corresponding to the average particle orientation.
    """  # noqa: E501
    cdef freud._order.NematicOrderParameter *thisptr
    cdef u

    def __cinit__(self, u):
        # run checks
        if len(u) != 3:
            raise ValueError('u needs to be a three-dimensional vector')

        cdef vec3[float] l_u = vec3[float](u[0], u[1], u[2])
        self.thisptr = new freud._order.NematicOrderParameter(l_u)
        self.u = freud.common.convert_array(
            u, 1, dtype=np.float32, contiguous=True, array_name="u")

    def compute(self, orientations):
        R"""Calculates the per-particle and global order parameter.

        Args:
            orientations (:math:`\left(N_{particles}, 4 \right)` :class:`numpy.ndarray`):
                Orientations to calculate the order parameter.
        """  # noqa: E501
        orientations = freud.common.convert_array(
            orientations, 2, dtype=np.float32, contiguous=True,
            array_name="orientations")
        if orientations.shape[1] != 4:
            raise TypeError('orientations should be an Nx4 array')

        cdef float[:, ::1] l_orientations = orientations
        cdef unsigned int num_particles = l_orientations.shape[0]

        with nogil:
            self.thisptr.compute(<quat[float]*> &l_orientations[0, 0],
                                 num_particles)

    @property
    def nematic_order_parameter(self):
        return self.thisptr.getNematicOrderParameter()

    @property
    def director(self):
        cdef vec3[float] n = self.thisptr.getNematicDirector()
        return np.asarray([n.x, n.y, n.z], dtype=np.float32)

    @property
    def particle_tensor(self):
        cdef unsigned int n_particles = self.thisptr.getNumParticles()
        cdef float[:, :, ::1] particle_tensor = \
            <float[:n_particles, :3, :3]> \
            self.thisptr.getParticleTensor().get()
        return np.asarray(particle_tensor)

    @property
    def nematic_tensor(self):
        cdef float[:, ::1] nematic_tensor = \
            <float[:3, :3]> self.thisptr.getNematicTensor().get()
        return np.asarray(nematic_tensor)

    def __repr__(self):
        return "freud.order.{cls}(u={u})".format(cls=type(self).__name__,
                                                 u=self.u.tolist())

    def __str__(self):
        return repr(self)


cdef class HexOrderParameter:
    R"""Calculates the :math:`k`-atic order parameter for each particle in the
    system.

    The :math:`k`-atic order parameter for a particle :math:`i` and its
    :math:`n` neighbors :math:`j` is given by:

    :math:`\psi_k \left( i \right) = \frac{1}{n}
    \sum_j^n e^{k i \phi_{ij}}`

    The parameter :math:`k` governs the symmetry of the order parameter while
    the parameter :math:`n` governs the number of neighbors of particle
    :math:`i` to average over. :math:`\phi_{ij}` is the angle between the
    vector :math:`r_{ij}` and :math:`\left( 1,0 \right)`.

    .. note::
        **2D:** :class:`freud.order.HexOrderParameter` properly handles 2D
        boxes. The points must be passed in as :code:`[x, y, 0]`. Failing to
        set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        rmax (float):
            +/- r distance to search for neighbors.
        k (unsigned int):
            Symmetry of order parameter (:math:`k=6` is hexatic).
        n (unsigned int):
            Number of neighbors (:math:`n=k` if :math:`n` not specified).

    Attributes:
        psi (:math:`\left(N_{particles} \right)` :class:`numpy.ndarray`):
            Order parameter.
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        K (unsigned int):
            Symmetry of the order parameter.
    """
    cdef freud._order.HexOrderParameter * thisptr
    cdef int num_neigh
    cdef float rmax

    def __cinit__(self, rmax, k=int(6), n=int(0)):
        self.thisptr = new freud._order.HexOrderParameter(rmax, k, n)
        self.rmax = rmax
        self.num_neigh = (n if n else int(k))

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, nlist=None):
        R"""Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist_nn(
            b, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        with nogil:
            self.thisptr.compute(dereference(b.thisptr), nlist_.get_ptr(),
                                 <vec3[float]*> &l_points[0, 0], nP)
        return self

    @property
    def psi(self):
        cdef unsigned int n_particles = self.thisptr.getNP()
        cdef np.complex64_t[::1] psi = \
            <np.complex64_t[:n_particles]> self.thisptr.getPsi().get()
        return np.asarray(psi, dtype=np.complex64)

    @property
    def box(self):
        return freud.box.BoxFromCPP(<freud._box.Box> self.thisptr.getBox())

    @property
    def num_particles(self):
        cdef unsigned int np = self.thisptr.getNP()
        return np

    @property
    def K(self):
        cdef unsigned int k = self.thisptr.getK()
        return k

    def __repr__(self):
        return "freud.order.{cls}(rmax={rmax}, k={k}, n={n})".format(
            cls=type(self).__name__, rmax=self.rmax, k=self.K,
            n=self.num_neigh)

    def __str__(self):
        return repr(self)


cdef class TransOrderParameter:
    R"""Compute the translational order parameter for each particle.

    .. moduleauthor:: Wenbo Shen <shenwb@umich.edu>

    Args:
        rmax (float):
            +/- r distance to search for neighbors.
        k (float):
            Symmetry of order parameter (:math:`k=6` is hexatic).
        n (unsigned int):
            Number of neighbors (:math:`n=k` if :math:`n` not specified).

    Attributes:
        d_r (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            Reference to the last computed translational order array.
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        K (float):
            Normalization value (d_r is divided by K).
    """
    cdef freud._order.TransOrderParameter * thisptr
    cdef num_neigh
    cdef rmax

    def __cinit__(self, rmax, k=6.0, n=0):
        self.thisptr = new freud._order.TransOrderParameter(rmax, k)
        self.rmax = rmax
        self.num_neigh = (n if n else int(k))

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, nlist=None):
        R"""Calculates the local descriptors.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist_nn(
            b, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        with nogil:
            self.thisptr.compute(dereference(b.thisptr), nlist_.get_ptr(),
                                 <vec3[float]*> &l_points[0, 0], nP)
        return self

    @property
    def d_r(self):
        cdef unsigned int n_particles = self.thisptr.getNP()
        cdef np.complex64_t[::1] d_r = \
            <np.complex64_t[:n_particles]> self.thisptr.getDr().get()
        return np.asarray(d_r, dtype=np.complex64)

    @property
    def box(self):
        return freud.box.BoxFromCPP(<freud._box.Box> self.thisptr.getBox())

    @property
    def num_particles(self):
        cdef unsigned int np = self.thisptr.getNP()
        return np

    @property
    def K(self):
        cdef float k = self.thisptr.getK()
        return k

    def __repr__(self):
        return "freud.order.{cls}(rmax={rmax}, k={k}, n={n})".format(
            cls=type(self).__name__, rmax=self.rmax, k=self.K,
            n=self.num_neigh)

    def __str__(self):
        return repr(self)


cdef class LocalQl:
    R"""Compute the local Steinhardt [Steinhardt1983]_ rotationally invariant
    :math:`Q_l` order parameter for a set of points.

    Implements the local rotationally invariant :math:`Q_l` order parameter
    described by Steinhardt. For a particle i, we calculate the average
    :math:`Q_l` by summing the spherical harmonics between particle :math:`i`
    and its neighbors :math:`j` in a local region:
    :math:`\overline{Q}_{lm}(i) = \frac{1}{N_b}
    \displaystyle\sum_{j=1}^{N_b} Y_{lm}(\theta(\vec{r}_{ij}),
    \phi(\vec{r}_{ij}))`. The particles included in the sum are determined
    by the rmax argument to the constructor.

    This is then combined in a rotationally invariant fashion to remove local
    orientational order as follows: :math:`Q_l(i)=\sqrt{\frac{4\pi}{2l+1}
    \displaystyle\sum_{m=-l}^{l} |\overline{Q}_{lm}|^2 }`.

    The :meth:`~computeAve` method provides access to a variant of this
    parameter that performs a average over the first and second shell combined
    [Lechner2008]_. To compute this parameter, we perform a second averaging
    over the first neighbor shell of the particle to implicitly include
    information about the second neighbor shell. This averaging is performed by
    replacing the value :math:`\overline{Q}_{lm}(i)` in the original
    definition by the average value of :math:`\overline{Q}_{lm}(k)` over all
    the :math:`k` neighbors of particle :math:`i` as well as itself.

    The :meth:`~computeNorm` and :meth:`~computeAveNorm` methods provide
    normalized versions of :meth:`~compute` and :meth:`~computeAve`,
    where the normalization is performed by dividing by the average
    :math:`Q_{lm}` values over all particles.

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near the first
            minimum of the RDF are recommended.
        l (unsigned int):
            Spherical harmonic quantum number l. Must be a positive number.
        rmin (float):
            Can look at only the second shell or some arbitrary RDF region.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        Ql (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Ql (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`\bar{Q_l}` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Ql (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Ql (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`\bar{Q_l}` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).

    .. todo:: move box to compute, this is old API
    """  # noqa: E501
    cdef freud._order.LocalQl * qlptr
    cdef freud.box.Box m_box
    cdef rmax
    cdef sph_l
    cdef rmin

    def __cinit__(self, box, rmax, l, rmin=0, *args, **kwargs):
        cdef freud.box.Box b = freud.common.convert_box(box)
        if type(self) is LocalQl:
            self.m_box = b
            self.rmax = rmax
            self.sph_l = l
            self.rmin = rmin
            self.qlptr = new freud._order.LocalQl(
                dereference(b.thisptr), rmax, l, rmin)

    def __dealloc__(self):
        if type(self) is LocalQl:
            del self.qlptr
            self.qlptr = NULL

    @property
    def box(self):
        return freud.box.BoxFromCPP(<freud._box.Box> self.qlptr.getBox())

    @box.setter
    def box(self, value):
        cdef freud.box.Box b = freud.common.convert_box(value)
        self.qlptr.setBox(dereference(b.thisptr))

    def setBox(self, box):
        R"""Reset the simulation box.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
        """
        self.box = box

    @property
    def num_particles(self):
        cdef unsigned int np = self.qlptr.getNP()
        return np

    @property
    def Ql(self):
        cdef unsigned int n_particles = self.qlptr.getNP()
        cdef float[::1] Ql = \
            <float[:n_particles]> self.qlptr.getQl().get()
        return np.asarray(Ql)

    @property
    def ave_Ql(self):
        cdef unsigned int n_particles = self.qlptr.getNP()
        cdef float[::1] ave_Ql = \
            <float[:n_particles]> self.qlptr.getAveQl().get()
        return np.asarray(ave_Ql)

    @property
    def norm_Ql(self):
        cdef unsigned int n_particles = self.qlptr.getNP()
        cdef float[::1] norm_Ql = \
            <float[:n_particles]> self.qlptr.getQlNorm().get()
        return np.asarray(norm_Ql)

    @property
    def ave_norm_Ql(self):
        cdef unsigned int n_particles = self.qlptr.getNP()
        cdef float[::1] ave_norm_Ql = \
            <float[:n_particles]> self.qlptr.getQlAveNorm().get()
        return np.asarray(ave_norm_Ql)

    def compute(self, points, nlist=None):
        R"""Compute the order parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        self.qlptr.compute(nlist_.get_ptr(), <vec3[float]*> &l_points[0, 0],
                           nP)
        return self

    def computeAve(self, points, nlist=None):
        R"""Compute the order parameter over two nearest neighbor shells.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        self.qlptr.compute(nlist_.get_ptr(),
                           <vec3[float]*> &l_points[0, 0], nP)
        self.qlptr.computeAve(nlist_.get_ptr(),
                              <vec3[float]*> &l_points[0, 0], nP)
        return self

    def computeNorm(self, points, nlist=None):
        R"""Compute the order parameter normalized by the average spherical
        harmonic value over all the particles.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        self.qlptr.compute(nlist_.get_ptr(),
                           <vec3[float]*> &l_points[0, 0], nP)
        self.qlptr.computeNorm(<vec3[float]*> &l_points[0, 0], nP)
        return self

    def computeAveNorm(self, points, nlist=None):
        R"""Compute the order parameter over two nearest neighbor shells
        normalized by the average spherical harmonic value over all the
        particles.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        self.qlptr.compute(nlist_.get_ptr(),
                           <vec3[float]*> &l_points[0, 0], nP)
        self.qlptr.computeAve(nlist_.get_ptr(),
                              <vec3[float]*> &l_points[0, 0], nP)
        self.qlptr.computeAveNorm(<vec3[float]*> &l_points[0, 0], nP)
        return self

    def __repr__(self):
        return ("freud.order.{cls}(box={box}, rmax={rmax}, l={sph_l}, "
                "rmin={rmin})").format(cls=type(self).__name__,
                                       box=self.m_box,
                                       rmax=self.rmax,
                                       sph_l=self.sph_l,
                                       rmin=self.rmin)

    def __str__(self):
        return repr(self)


cdef class LocalQlNear(LocalQl):
    R"""A variant of the :class:`~LocalQl` class that performs its average
    over nearest neighbor particles as determined by an instance of
    :class:`freud.locality.NeighborList`. The number of included neighbors
    is determined by the kn parameter to the constructor.

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near the first
            minimum of the RDF are recommended.
        l (unsigned int):
            Spherical harmonic quantum number l. Must be a positive number.
        kn (unsigned int):
            Number of nearest neighbors. must be a positive integer.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        Ql (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Ql (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`\bar{Q_l}` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Ql (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Ql (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`\bar{Q_l}` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).

    .. todo:: move box to compute, this is old API
    """  # noqa: E501
    cdef num_neigh

    def __cinit__(self, box, rmax, l, kn=12):
        # Note that we cannot leverage super here because the
        # type conditional in the parent will prevent it.
        # Unfortunately, this is necessary for proper memory
        # management in this inheritance structure.
        cdef freud.box.Box b = freud.common.convert_box(box)
        if type(self) == LocalQlNear:
            self.qlptr = new freud._order.LocalQl(
                dereference(b.thisptr), rmax, l, 0)
            self.m_box = b
            self.rmax = rmax
            self.sph_l = l
            self.num_neigh = kn

    def __dealloc__(self):
        if type(self) == LocalQlNear:
            del self.qlptr
            self.qlptr = NULL

    def compute(self, points, nlist=None):
        R"""Compute the order parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalQlNear, self).compute(points, nlist_)

    def computeAve(self, points, nlist=None):
        R"""Compute the order parameter over two nearest neighbor shells.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalQlNear, self).computeAve(points, nlist_)

    def computeNorm(self, points, nlist=None):
        R"""Compute the order parameter normalized by the average spherical
        harmonic value over all the particles.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalQlNear, self).computeNorm(points, nlist_)

    def computeAveNorm(self, points, nlist=None):
        R"""Compute the order parameter over two nearest neighbor shells
        normalized by the average spherical harmonic value over all the
        particles.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalQlNear, self).computeAveNorm(points, nlist_)

    def __repr__(self):
        return ("freud.order.{cls}(box={box}, rmax={rmax}, l={sph_l}, "
                "kn={kn})").format(cls=type(self).__name__,
                                   box=self.m_box,
                                   rmax=self.rmax,
                                   sph_l=self.sph_l,
                                   kn=self.num_neigh)

    def __str__(self):
        return repr(self)


cdef class LocalWl(LocalQl):
    R"""Compute the local Steinhardt [Steinhardt1983]_ rotationally invariant
    :math:`W_l` order parameter for a set of points.

    Implements the local rotationally invariant :math:`W_l` order parameter
    described by Steinhardt. For a particle i, we calculate the average
    :math:`W_l` by summing the spherical harmonics between particle :math:`i`
    and its neighbors :math:`j` in a local region:
    :math:`\overline{Q}_{lm}(i) = \frac{1}{N_b}
    \displaystyle\sum_{j=1}^{N_b} Y_{lm}(\theta(\vec{r}_{ij}),
    \phi(\vec{r}_{ij}))`. The particles included in the sum are determined
    by the rmax argument to the constructor.

    The :math:`W_l` is then defined as a weighted average over the
    :math:`\overline{Q}_{lm}(i)` values using Wigner 3j symbols
    (Clebsch-Gordan coefficients). The resulting combination is rotationally
    (i.e. frame) invariant.

    The :meth:`~computeAve` method provides access to a variant of this
    parameter that performs a average over the first and second shell combined
    [Lechner2008]_. To compute this parameter, we perform a second averaging
    over the first neighbor shell of the particle to implicitly include
    information about the second neighbor shell. This averaging is performed by
    replacing the value :math:`\overline{Q}_{lm}(i)` in the original
    definition by the average value of :math:`\overline{Q}_{lm}(k)` over all
    the :math:`k` neighbors of particle :math:`i` as well as itself.

    The :meth:`~computeNorm` and :meth:`~computeAveNorm` methods provide
    normalized versions of :meth:`~compute` and :meth:`~computeAve`,
    where the normalization is performed by dividing by the average
    :math:`Q_{lm}` values over all particles.

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near the first
            minimum of the RDF are recommended.
        l (unsigned int):
            Spherical harmonic quantum number l. Must be a positive number
        rmin (float):
            Lower bound for computing the local order parameter. Allows looking
            at, for instance, only the second shell, or some other arbitrary
            RDF region.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        Wl (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`W_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Wl (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`\bar{W}_l` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Wl (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`W_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Wl (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`\bar{W}_l` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).

    .. todo:: move box to compute, this is old API
    """  # noqa: E501
    cdef freud._order.LocalWl * thisptr

    # List of Ql attributes to remove
    delattrs = ['Ql', 'getQl',
                'ave_Ql', 'getAveQl',
                'norm_Ql', 'getQlNorm',
                'ave_norm_Ql', 'getQlAveNorm']

    def __cinit__(self, box, rmax, l, rmin=0, *args, **kwargs):
        cdef freud.box.Box b = freud.common.convert_box(box)
        if type(self) is LocalWl:
            self.thisptr = self.qlptr = new freud._order.LocalWl(
                dereference(b.thisptr), rmax, l, rmin)
            self.m_box = b
            self.rmax = rmax
            self.sph_l = l
            self.rmin = rmin

    def __dealloc__(self):
        if type(self) is LocalWl:
            del self.thisptr
            self.thisptr = NULL

    def __getattribute__(self, name):
        # Remove access to Ql methods from this class, their values may be
        # uninitialized and are not dependable.
        if name in LocalWl.delattrs:
            raise AttributeError(name)
        else:
            return super(LocalWl, self).__getattribute__(name)

    def __dir__(self):
        # Prevent unwanted Ql methods from appearing in dir output
        return sorted(set(dir(self.__class__)) -
                      set(self.__class__.delattrs))

    @property
    def Wl(self):
        cdef unsigned int n_particles = self.qlptr.getNP()
        cdef np.complex64_t[::1] Wl = \
            <np.complex64_t[:n_particles]> self.thisptr.getWl().get()
        return np.asarray(Wl, dtype=np.complex64)

    @property
    def ave_Wl(self):
        cdef unsigned int n_particles = self.qlptr.getNP()
        cdef np.complex64_t[::1] ave_Wl = \
            <np.complex64_t[:n_particles]> self.thisptr.getAveWl().get()
        return np.asarray(ave_Wl, dtype=np.complex64)

    @property
    def norm_Wl(self):
        cdef unsigned int n_particles = self.qlptr.getNP()
        cdef np.complex64_t[::1] norm_Wl = \
            <np.complex64_t[:n_particles]> self.thisptr.getWlNorm().get()
        return np.asarray(norm_Wl, dtype=np.complex64)

    @property
    def ave_norm_Wl(self):
        cdef unsigned int n_particles = self.qlptr.getNP()
        cdef np.complex64_t[::1] ave_norm_Wl = \
            <np.complex64_t[:n_particles]> self.thisptr.getAveNormWl().get()
        return np.asarray(ave_norm_Wl, dtype=np.complex64)

    def __repr__(self):
        return ("freud.order.{cls}(box={box}, rmax={rmax}, l={sph_l}, "
                "rmin={rmin})").format(cls=type(self).__name__,
                                       box=self.m_box,
                                       rmax=self.rmax,
                                       sph_l=self.sph_l,
                                       rmin=self.rmin)

    def __str__(self):
        return repr(self)


cdef class LocalWlNear(LocalWl):
    R"""A variant of the :class:`~LocalWl` class that performs its average
    over nearest neighbor particles as determined by an instance of
    :class:`freud.locality.NeighborList`. The number of included neighbors
    is determined by the kn parameter to the constructor.

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near the first
            minimum of the RDF are recommended.
        l (unsigned int):
            Spherical harmonic quantum number l. Must be a positive number
        kn (unsigned int):
            Number of nearest neighbors. Must be a positive number.


    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        Wl (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`W_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Wl (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`\bar{W}_l` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Wl (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`W_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Wl (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`\bar{W}_l` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).

    .. todo:: move box to compute, this is old API
    """  # noqa: E501
    cdef num_neigh

    def __cinit__(self, box, rmax, l, kn=12):
        cdef freud.box.Box b = freud.common.convert_box(box)
        if type(self) is LocalWlNear:
            self.thisptr = self.qlptr = new freud._order.LocalWl(
                dereference(b.thisptr), rmax, l, 0)
            self.m_box = b
            self.rmax = rmax
            self.sph_l = l
            self.num_neigh = kn

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def compute(self, points, nlist=None):
        R"""Compute the order parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalWlNear, self).compute(points, nlist_)

    def computeAve(self, points, nlist=None):
        R"""Compute the order parameter over two nearest neighbor shells.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalWlNear, self).computeAve(points, nlist_)

    def computeNorm(self, points, nlist=None):
        R"""Compute the order parameter normalized by the average spherical
        harmonic value over all the particles.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalWlNear, self).computeNorm(points, nlist_)

    def computeAveNorm(self, points, nlist=None):
        R"""Compute the order parameter over two nearest neighbor shells
        normalized by the average spherical harmonic value over all the
        particles.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalWlNear, self).computeAveNorm(points, nlist_)

    def __repr__(self):
        return ("freud.order.{cls}(box={box}, rmax={rmax}, l={sph_l}, "
                "kn={kn})").format(cls=type(self).__name__,
                                   box=self.m_box,
                                   rmax=self.rmax,
                                   sph_l=self.sph_l,
                                   kn=self.num_neigh)

    def __str__(self):
        return repr(self)


cdef class SolLiq:
    R"""Uses dot products of :math:`Q_{lm}` between particles for clustering.

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near first
            minimum of the RDF are recommended.
        Qthreshold (float):
            Value of dot product threshold when evaluating
            :math:`Q_{lm}^*(i) Q_{lm}(j)` to determine if a neighbor pair is a
            solid-like bond. (For :math:`l=6`, 0.7 generally good for FCC or
            BCC structures).
        Sthreshold (unsigned int):
            Minimum required number of adjacent solid-link bonds for a particle
            to be considered solid-like for clustering. (For :math:`l=6`, 6-8
            is generally good for FCC or BCC structures).
        l (unsigned int):
            Choose spherical harmonic :math:`Q_l`. Must be positive and even.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        largest_cluster_size (unsigned int):
            The largest cluster size. Must call a compute method first.
        cluster_sizes (unsigned int):
            The sizes of all clusters.
        largest_cluster_size (unsigned int):
            The largest cluster size. Must call a compute method first.
        Ql_mi (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_{lmi}` for each particle.
        clusters (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed set of solid-like cluster indices for each
            particle.
        num_connections (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The number of connections per particle.
        Ql_dot_ij (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            Reference to the qldot_ij values.
        num_particles (unsigned int):
            Number of particles.

    .. todo:: move box to compute, this is old API
    """  # noqa: E501
    cdef freud._order.SolLiq * thisptr
    cdef freud.box.Box m_box
    cdef rmax

    def __cinit__(self, box, rmax, Qthreshold, Sthreshold, l, *args, **kwargs):
        cdef freud.box.Box b = freud.common.convert_box(box)
        if type(self) is SolLiq:
            self.thisptr = new freud._order.SolLiq(
                dereference(b.thisptr), rmax, Qthreshold, Sthreshold, l)
            self.m_box = b
            self.rmax = rmax

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def compute(self, points, nlist=None):
        R"""Compute the solid-liquid order parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        self.thisptr.compute(nlist_.get_ptr(),
                             <vec3[float]*> &l_points[0, 0], nP)
        return self

    def computeSolLiqVariant(self, points, nlist=None):
        R"""Compute a variant of the solid-liquid order parameter.

        This variant method places a minimum threshold on the number
        of solid-like bonds a particle must have to be considered solid-like
        for clustering purposes.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        self.thisptr.computeSolLiqVariant(
            nlist_.get_ptr(), <vec3[float]*> &l_points[0, 0], nP)
        return self

    def computeSolLiqNoNorm(self, points, nlist=None):
        R"""Compute the solid-liquid order parameter without normalizing the dot
        product.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef float[:, ::1] l_points = points
        cdef unsigned int nP = l_points.shape[0]

        defaulted_nlist = freud.locality.make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]

        self.thisptr.computeSolLiqNoNorm(
            nlist_.get_ptr(), <vec3[float]*> &l_points[0, 0], nP)
        return self

    @property
    def box(self):
        return freud.box.BoxFromCPP(<freud._box.Box> self.thisptr.getBox())

    @box.setter
    def box(self, value):
        cdef freud.box.Box b = freud.common.convert_box(value)
        self.thisptr.setBox(dereference(b.thisptr))

    @property
    def largest_cluster_size(self):
        cdef unsigned int clusterSize = self.thisptr.getLargestClusterSize()
        return clusterSize

    @property
    def cluster_sizes(self):
        cdef unsigned int n_clusters = self.thisptr.getNumClusters()
        cdef unsigned int[::1] cluster_sizes = \
            <unsigned int[:n_clusters]> self.thisptr.getClusterSizes().data()
        return np.asarray(cluster_sizes, dtype=np.uint32)

    @property
    def Ql_mi(self):
        cdef unsigned int n_particles = self.thisptr.getNP()
        cdef np.complex64_t[::1] Ql_mi = \
            <np.complex64_t[:n_particles]> self.thisptr.getQlmi().get()
        return np.asarray(Ql_mi, dtype=np.complex64)

    @property
    def clusters(self):
        cdef unsigned int n_particles = self.thisptr.getNP()
        cdef unsigned int[::1] clusters = \
            <unsigned int[:n_particles]> self.thisptr.getClusters().get()
        return np.asarray(clusters, dtype=np.uint32)

    @property
    def num_connections(self):
        cdef unsigned int n_particles = self.thisptr.getNP()
        cdef unsigned int[::1] num_connections = \
            <unsigned int[:n_particles]> \
            self.thisptr.getNumberOfConnections().get()
        return np.asarray(num_connections, dtype=np.uint32)

    @property
    def Ql_dot_ij(self):
        cdef unsigned int n_clusters = self.thisptr.getNumClusters()
        cdef np.complex64_t[::1] Ql_dot_ij = \
            <np.complex64_t[:n_clusters]> self.thisptr.getQldot_ij().data()
        return np.asarray(Ql_dot_ij, dtype=np.complex64)

    @property
    def num_particles(self):
        cdef unsigned int np = self.thisptr.getNP()
        return np


cdef class SolLiqNear(SolLiq):
    R"""A variant of the :class:`~SolLiq` class that performs its average over nearest neighbor particles as determined by an instance of :class:`freud.locality.NeighborList`. The number of included neighbors is determined by the kn parameter to the constructor.

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near the first
            minimum of the RDF are recommended.
        Qthreshold (float):
            Value of dot product threshold when evaluating
            :math:`Q_{lm}^*(i) Q_{lm}(j)` to determine if a neighbor pair is a
            solid-like bond. (For :math:`l=6`, 0.7 generally good for FCC or
            BCC structures).
        Sthreshold (unsigned int):
            Minimum required number of adjacent solid-link bonds for a particle
            to be considered solid-like for clustering. (For :math:`l=6`, 6-8
            is generally good for FCC or BCC structures).
        l (unsigned int):
            Choose spherical harmonic :math:`Q_l`. Must be positive and even.
        kn (unsigned int):
            Number of nearest neighbors. Must be a positive number.

    Attributes:
        box (:class:`freud.box.Box`):
            Box used in the calculation.
        largest_cluster_size (unsigned int):
            The largest cluster size. Must call a compute method first.
        cluster_sizes (unsigned int):
            The sizes of all clusters.
        largest_cluster_size (unsigned int):
            The largest cluster size. Must call a compute method first.
        Ql_mi (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_{lmi}` for each particle.
        clusters (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The last computed set of solid-like cluster indices for each
            particle.
        num_connections (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            The number of connections per particle.
        Ql_dot_ij (:math:`\left(N_{particles}\right)` :class:`numpy.ndarray`):
            Reference to the qldot_ij values.
        num_particles (unsigned int):
            Number of particles.

    .. todo:: move box to compute, this is old API
    """  # noqa: E501
    cdef num_neigh

    def __cinit__(self, box, rmax, Qthreshold, Sthreshold, l, kn=12):
        cdef freud.box.Box b = freud.common.convert_box(box)
        if type(self) is SolLiqNear:
            self.thisptr = new freud._order.SolLiq(
                dereference(b.thisptr), rmax, Qthreshold, Sthreshold, l)
            self.m_box = b
            self.rmax = rmax
            self.num_neigh = kn

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def compute(self, points, nlist=None):
        R"""Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return SolLiq.compute(self, points, nlist_)

    def computeSolLiqVariant(self, points, nlist=None):
        R"""Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return SolLiq.computeSolLiqVariant(self, points, nlist_)

    def computeSolLiqNoNorm(self, points, nlist=None):
        R"""Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        defaulted_nlist = freud.locality.make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef freud.locality.NeighborList nlist_ = defaulted_nlist[0]
        return SolLiq.computeSolLiqNoNorm(self, points, nlist_)


cdef class RotationalAutocorrelation:
    """Calculates a measure of total rotational autocorrelation based on
    hyperspherical harmonics as laid out in "Design rules for engineering
    colloidal plastic crystals of hard polyhedra - phase behavior and
    directional entropic forces" by Karas et al. (currently in preparation).
    The output is not a correlation function, but rather a scalar value that
    measures total system orientational correlation with an initial state. As
    such, the output can be treated as an order parameter measuring degrees of
    rotational (de)correlation. For analysis of a trajectory, the compute call
    needs to be done at each trajectory frame.

    .. moduleauthor:: Andrew Karas <askaras@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    .. versionadded:: 1.0

    Args:
        l (int):
            Order of the hyperspherical harmonic. Must be a positive, even
            integer.

    Attributes:
        num_orientations (unsigned int):
            The number of orientations used in computing the last set.
        azimuthal (int):
            The azimuthal quantum number, which defines the order of the
            hyperspherical harmonic. Must be a positive, even integer.
        ra_array ((:math:`N_{orientations}`) :class:`numpy.ndarray`):
            The per-orientation array of rotational autocorrelation values
            calculated by the last call to compute.
        autocorrelation (float):
            The autocorrelation computed in the last call to compute.
    """
    cdef freud._order.RotationalAutocorrelation * thisptr
    cdef int l

    def __cinit__(self, l):
        if l % 2 or l < 0:
            raise ValueError(
                "The quantum number must be a positive, even integer.")
        self.l = l  # noqa
        self.thisptr = new freud._order.RotationalAutocorrelation(
            self.l)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, ref_ors, ors):
        """Calculates the rotational autocorrelation function for a single frame.

        Args:
            ref_ors ((:math:`N_{orientations}`, 4) :class:`numpy.ndarray`):
                Reference orientations for the initial frame.
            ors ((:math:`N_{orientations}`, 4) :class:`numpy.ndarray`):
                Orientations for the frame of interest.
        """
        ref_ors = freud.common.convert_array(
            ref_ors, 2, dtype=np.float32, contiguous=True,
            array_name="ref_ors")
        if ref_ors.shape[1] != 4:
            raise TypeError('ref_ors should be an Nx4 array')

        ors = freud.common.convert_array(
            ors, 2, dtype=np.float32, contiguous=True, array_name="ors")
        if ors.shape[1] != 4:
            raise TypeError('ors should be an Nx4 array')

        cdef const float[:, ::1] l_ref_ors = ref_ors
        cdef const float[:, ::1] l_ors = ors
        cdef unsigned int nP = ors.shape[0]

        with nogil:
            self.thisptr.compute(
                <quat[float]*> &l_ref_ors[0, 0],
                <quat[float]*> &l_ors[0, 0],
                nP)
        return self

    @property
    def autocorrelation(self):
        cdef float Ft = self.thisptr.getRotationalAutocorrelation()
        return Ft

    @property
    def ra_array(self):
        cdef unsigned int num_orientations = self.thisptr.getN()
        cdef np.complex64_t[::1] result = \
            <np.complex64_t[:num_orientations]> self.thisptr.getRAArray().get()
        return np.asarray(result, dtype=np.complex64)

    @property
    def num_orientations(self):
        cdef unsigned int num = self.thisptr.getN()
        return num

    @property
    def azimuthal(self):
        cdef unsigned int azimuthal = self.thisptr.getL()
        return azimuthal

    def __repr__(self):
        return "freud.order.{cls}(l={sph_l})".format(cls=type(self).__name__,
                                                     sph_l=self.l)

    def __str__(self):
        return repr(self)
