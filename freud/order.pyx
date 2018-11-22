# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The order module contains functions which compute order parameters for the
whole system or individual particles. Order parameters take bond order data and
interpret it in some way to quantify the degree of order in a system using a
scalar value. This is often done through computing spherical harmonics of the
bond order diagram, which are the spherical analogue of Fourier Transforms.
"""

import freud.common
import warnings
from freud.errors import FreudDeprecationWarning
import numpy as np
import time
import freud.locality

from freud.util._VectorMath cimport vec3, quat
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
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

    def get_t_initial(self):
        warnings.warn("The get_t_initial function is deprecated in favor "
                      "of the t_initial class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.t_initial

    @property
    def t_final(self):
        return self.thisptr.getTFinal()

    def get_t_final(self):
        warnings.warn("The get_t_final function is deprecated in favor "
                      "of the t_final class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.t_final

    @property
    def scale(self):
        return self.thisptr.getScale()

    def get_scale(self):
        warnings.warn("The get_scale function is deprecated in favor "
                      "of the scale class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.scale

    @property
    def cubatic_order_parameter(self):
        return self.thisptr.getCubaticOrderParameter()

    def get_cubatic_order_parameter(self):
        warnings.warn("The get_cubatic_order_parameter function is deprecated "
                      "in favor of the cubatic_order_parameter class "
                      "attribute and will be removed in a future version of "
                      "freud.",
                      FreudDeprecationWarning)
        return self.cubatic_order_parameter

    @property
    def orientation(self):
        cdef quat[float] q = self.thisptr.getCubaticOrientation()
        cdef np.ndarray[float, ndim=1] result = np.array(
            [q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)
        return result

    def get_orientation(self):
        warnings.warn("The get_orientation function is deprecated in favor "
                      "of the orientation class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.orientation

    @property
    def particle_order_parameter(self):
        cdef float * particle_op = \
            self.thisptr.getParticleCubaticOrderParameter().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32,
                                         <void*> particle_op)
        return result

    def get_particle_op(self):
        warnings.warn("The get_particle_op function is deprecated in favor "
                      "of the particle_order_parameter class attribute and "
                      "will be removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.particle_order_parameter

    @property
    def particle_tensor(self):
        cdef float * particle_tensor = self.thisptr.getParticleTensor().get()
        cdef np.npy_intp nbins[5]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp> 3
        nbins[2] = <np.npy_intp> 3
        nbins[3] = <np.npy_intp> 3
        nbins[4] = <np.npy_intp> 3
        cdef np.ndarray[np.float32_t, ndim=5] result = \
            np.PyArray_SimpleNewFromData(5, nbins, np.NPY_FLOAT32,
                                         <void*> particle_tensor)
        return result

    def get_particle_tensor(self):
        warnings.warn("The get_particle_tensor function is deprecated in "
                      "favor of the particle_tensor class attribute and will "
                      "be removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.particle_tensor

    @property
    def global_tensor(self):
        cdef float * global_tensor = self.thisptr.getGlobalTensor().get()
        cdef np.npy_intp nbins[4]
        nbins[0] = <np.npy_intp> 3
        nbins[1] = <np.npy_intp> 3
        nbins[2] = <np.npy_intp> 3
        nbins[3] = <np.npy_intp> 3
        cdef np.ndarray[np.float32_t, ndim=4] result = \
            np.PyArray_SimpleNewFromData(4, nbins, np.NPY_FLOAT32,
                                         <void*> global_tensor)
        return result

    def get_global_tensor(self):
        warnings.warn("The get_global_tensor function is deprecated in favor "
                      "of the global_tensor class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.global_tensor

    @property
    def cubatic_tensor(self):
        cdef float * cubatic_tensor = self.thisptr.getCubaticTensor().get()
        cdef np.npy_intp nbins[4]
        nbins[0] = <np.npy_intp> 3
        nbins[1] = <np.npy_intp> 3
        nbins[2] = <np.npy_intp> 3
        nbins[3] = <np.npy_intp> 3
        cdef np.ndarray[np.float32_t, ndim=4] result = \
            np.PyArray_SimpleNewFromData(4, nbins, np.NPY_FLOAT32,
                                         <void*> cubatic_tensor)
        return result

    def get_cubatic_tensor(self):
        warnings.warn("The get_cubatic_tensor function is deprecated in favor "
                      "of the cubatic_tensor class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.cubatic_tensor

    @property
    def gen_r4_tensor(self):
        cdef float * gen_r4_tensor = self.thisptr.getGenR4Tensor().get()
        cdef np.npy_intp nbins[4]
        nbins[0] = <np.npy_intp> 3
        nbins[1] = <np.npy_intp> 3
        nbins[2] = <np.npy_intp> 3
        nbins[3] = <np.npy_intp> 3
        cdef np.ndarray[np.float32_t, ndim=4] result = \
            np.PyArray_SimpleNewFromData(4, nbins, np.NPY_FLOAT32,
                                         <void*> gen_r4_tensor)
        return result

    def get_gen_r4_tensor(self):
        warnings.warn("The get_gen_r4_tensor function is deprecated in favor "
                      "of the gen_r4_tensor class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.gen_r4_tensor


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

    def __cinit__(self, u):
        # run checks
        if len(u) != 3:
            raise ValueError('u needs to be a three-dimensional vector')

        cdef vec3[float] l_u = vec3[float](u[0], u[1], u[2])
        self.thisptr = new freud._order.NematicOrderParameter(l_u)

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

    def get_nematic_order_parameter(self):
        warnings.warn("The get_nematic_order_parameter function is deprecated "
                      "in favor of the nematic_order_parameter class "
                      "attribute and will be removed in a future version of "
                      "freud.",
                      FreudDeprecationWarning)
        return self.nematic_order_parameter

    @property
    def director(self):
        cdef vec3[float] n = self.thisptr.getNematicDirector()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.array(
            [n.x, n.y, n.z], dtype=np.float32)
        return result

    def get_director(self):
        warnings.warn("The get_director function is deprecated in favor "
                      "of the director class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.director

    @property
    def particle_tensor(self):
        cdef float *particle_tensor = self.thisptr.getParticleTensor().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp> 3
        nbins[2] = <np.npy_intp> 3
        cdef np.ndarray[np.float32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32,
                                         <void*> particle_tensor)
        return result

    def get_particle_tensor(self):
        warnings.warn("The get_particle_tensor function is deprecated in "
                      "favor of the particle_tensor class attribute and will "
                      "be removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.particle_tensor

    @property
    def nematic_tensor(self):
        cdef float *nematic_tensor = self.thisptr.getNematicTensor().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> 3
        nbins[1] = <np.npy_intp> 3
        cdef np.ndarray[np.float32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32,
                                         <void*> nematic_tensor)
        return result

    def get_nematic_tensor(self):
        warnings.warn("The get_nematic_tensor function is deprecated in favor "
                      "of the nematic_tensor class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.nematic_tensor


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
        cdef float complex * psi = self.thisptr.getPsi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> psi)
        return result

    def getPsi(self):
        warnings.warn("The getPsi function is deprecated in favor "
                      "of the psi class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.psi

    @property
    def box(self):
        return freud.box.BoxFromCPP(< freud._box.Box > self.thisptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    @property
    def num_particles(self):
        cdef unsigned int np = self.thisptr.getNP()
        return np

    def getNP(self):
        warnings.warn("The getNP function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_particles

    @property
    def K(self):
        cdef unsigned int k = self.thisptr.getK()
        return k

    def getK(self):
        warnings.warn("The getK function is deprecated in favor "
                      "of the K class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.K


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
        cdef float complex * dr = self.thisptr.getDr().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> dr)
        return result

    def getDr(self):
        warnings.warn("The getDr function is deprecated in favor "
                      "of the d_r class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.d_r

    @property
    def box(self):
        return freud.box.BoxFromCPP(< freud._box.Box > self.thisptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    @property
    def num_particles(self):
        cdef unsigned int np = self.thisptr.getNP()
        return np

    def getNP(self):
        warnings.warn("The getNP function is deprecated in favor "
                      "of the num_particles class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_particles


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

    def __cinit__(self, box, rmax, l, rmin=0):
        cdef freud.box.Box b = freud.common.convert_box(box)
        if type(self) is LocalQl:
            self.m_box = b
            self.rmax = rmax
            self.qlptr = new freud._order.LocalQl(
                dereference(b.thisptr), rmax, l, rmin)

    def __dealloc__(self):
        if type(self) is LocalQl:
            del self.qlptr
            self.qlptr = NULL

    @property
    def box(self):
        return freud.box.BoxFromCPP(< freud._box.Box > self.qlptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

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

    def getNP(self):
        warnings.warn("The getNP function is deprecated in favor "
                      "of the num_particles class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_particles

    @property
    def Ql(self):
        cdef float * Ql = self.qlptr.getQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[float, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> Ql)
        return result

    def getQl(self):
        warnings.warn("The getQl function is deprecated in favor "
                      "of the Ql class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Ql

    @property
    def ave_Ql(self):
        cdef float * Ql = self.qlptr.getAveQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[float, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> Ql)
        return result

    def getAveQl(self):
        warnings.warn("The getAveQl function is deprecated in favor "
                      "of the ave_Ql class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.ave_Ql

    @property
    def norm_Ql(self):
        cdef float * Ql = self.qlptr.getQlNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[float, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> Ql)
        return result

    def getQlNorm(self):
        warnings.warn("The getQlNorm function is deprecated in favor "
                      "of the norm_Ql class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.norm_Ql

    @property
    def ave_norm_Ql(self):
        cdef float * Ql = self.qlptr.getQlAveNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[float, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> Ql)
        return result

    def getQlAveNorm(self):
        warnings.warn("The getQlAveNorm function is deprecated in favor "
                      "of the ave_norm_Ql class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.ave_norm_Ql

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
            self.num_neigh = kn

    def __dealloc__(self):
        if type(self) == LocalQlNear:
            del self.qlptr
            self.qlptr = NULL

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
        return self.getWl()

    def getWl(self):
        warnings.warn("The getWl function is deprecated in favor "
                      "of the Wl class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        cdef float complex * Wl = self.thisptr.getWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Wl)
        return result

    @property
    def ave_Wl(self):
        cdef float complex * Wl = self.thisptr.getAveWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Wl)
        return result

    def getAveWl(self):
        warnings.warn("The getAveWl function is deprecated in favor "
                      "of the ave_Wl class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.ave_Wl

    @property
    def norm_Wl(self):
        cdef float complex * Wl = self.thisptr.getWlNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Wl)
        return result

    def getWlNorm(self):
        warnings.warn("The getWlNorm function is deprecated in favor "
                      "of the norm_Wl class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.norm_Wl

    @property
    def ave_norm_Wl(self):
        cdef float complex * Wl = self.thisptr.getAveNormWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Wl)
        return result

    def getWlAveNorm(self):
        warnings.warn("The getWlAveNorm function is deprecated in favor "
                      "of the ave_norm_Wl class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.ave_norm_Wl


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
            self.num_neigh = kn

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

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
        return freud.box.BoxFromCPP(< freud._box.Box > self.thisptr.getBox())

    def getBox(self):
        warnings.warn("The getBox function is deprecated in favor "
                      "of the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.box

    def setClusteringRadius(self, rcutCluster):
        warnings.warn("Use constructor arguments instead of this setter. "
                      "This setter will be removed in the future.",
                      FreudDeprecationWarning)
        self.thisptr.setClusteringRadius(rcutCluster)

    @box.setter
    def box(self, value):
        cdef freud.box.Box b = freud.common.convert_box(value)
        self.thisptr.setBox(dereference(b.thisptr))

    def setBox(self, box):
        warnings.warn("The setBox function is deprecated in favor "
                      "of setting the box class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        self.box = box

    @property
    def largest_cluster_size(self):
        cdef unsigned int clusterSize = self.thisptr.getLargestClusterSize()
        return clusterSize

    def getLargestClusterSize(self):
        warnings.warn("The getLargestClusterSize function is deprecated in "
                      "favor of the largest_cluster_size class attribute and "
                      "will be removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.largest_cluster_size

    @property
    def cluster_sizes(self):
        cdef vector[unsigned int] clusterSizes = self.thisptr.getClusterSizes()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNumClusters()
        cdef np.ndarray[np.uint32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32,
                                         <void*> &clusterSizes)
        return result

    def getClusterSizes(self):
        warnings.warn("The getClusterSizes function is deprecated in favor "
                      "of the cluster_sizes class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.cluster_sizes

    @property
    def Ql_mi(self):
        cdef float complex * Qlmi = self.thisptr.getQlmi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Qlmi)
        return result

    def getQlmi(self):
        warnings.warn("The getQlmi function is deprecated in favor "
                      "of the Ql_mi class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Ql_mi

    @property
    def clusters(self):
        cdef unsigned int * clusters = self.thisptr.getClusters().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32,
                                         <void*> clusters)
        return result

    def getClusters(self):
        warnings.warn("The getClusters function is deprecated in favor "
                      "of the clusters class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.clusters

    @property
    def num_connections(self):
        cdef unsigned int * connections = \
            self.thisptr.getNumberOfConnections().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32,
                                         <void*> connections)
        return result

    def getNumberOfConnections(self):
        warnings.warn("The getNumberOfConnections function is deprecated in "
                      "favor of the num_connections class attribute and will "
                      "be removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_connections

    @property
    def Ql_dot_ij(self):
        cdef vector[float complex] Qldot = self.thisptr.getQldot_ij()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNumClusters()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> &Qldot)
        return result

    def getQldot_ij(self):
        warnings.warn("The getQldot_ij function is deprecated in favor "
                      "of the Ql_dot_ij class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.Ql_dot_ij

    @property
    def num_particles(self):
        cdef unsigned int np = self.thisptr.getNP()
        return np

    def getNP(self):
        warnings.warn("The getNumParticles function is deprecated in favor "
                      "of the num_particles class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_particles


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


class BondOrder(_EBO):
    def __init__(self, rmax, k, n, n_bins_t, n_bins_p):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.BondOrder instead!",
                      FreudDeprecationWarning)


class LocalDescriptors(_ELD):
    def __init__(self, num_neighbors, lmax, rmax, negative_m=True):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.LocalDescriptors instead!",
                      FreudDeprecationWarning)


class MatchEnv(_EME):
    def __init__(self, box, rmax, k):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.MatchEnv instead!",
                      FreudDeprecationWarning)


class AngularSeparation(_EAS):
    def __init__(self, rmax, n):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.AngularSeparation instead!",
                      FreudDeprecationWarning)
