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
from freud.locality import make_default_nlist, make_default_nlist_nn

from freud.util._VectorMath cimport vec3, quat
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from freud.box cimport BoxFromCPP
from freud.locality cimport NeighborList
cimport freud._box, freud._order, freud._locality

# The below are maintained for backwards compatibility
# but have been moved to the environment module
from freud.environment cimport BondOrder as _EBO
from freud.environment cimport LocalDescriptors as _ELD
from freud.environment cimport MatchEnv as _EME
from freud.environment cimport Pairing2D as _EP
from freud.environment cimport AngularSeparation as _EAS

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class CubaticOrderParameter:
    """Compute the cubatic order parameter [HajiAkbari2015]_ for a system of
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
    """
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
        cdef np.ndarray[float, ndim=2] kd = np.eye(3, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] dijkl = np.einsum(
            "ij,kl->ijkl", kd, kd, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] dikjl = np.einsum(
            "ik,jl->ijkl", kd, kd, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] diljk = np.einsum(
            "il,jk->ijkl", kd, kd, dtype=np.float32)
        cdef np.ndarray[float, ndim=4] r4 = dijkl+dikjl+diljk
        r4 *= (2.0/5.0)
        self.thisptr = new freud._order.CubaticOrderParameter(
            t_initial, t_final, scale, <float*> r4.data, n_replicates, seed)

    def compute(self, orientations):
        """Calculates the per-particle and global order parameter.

        Args:
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Orientations as angles to use in computation.
        """
        orientations = freud.common.convert_array(
            orientations, 2, dtype=np.float32, contiguous=True,
            array_name="orientations")
        if orientations.shape[1] != 4:
            raise TypeError('orientations should be an Nx4 array')

        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef unsigned int num_particles = <unsigned int> orientations.shape[0]

        with nogil:
            self.thisptr.compute(
                <quat[float]*> l_orientations.data, num_particles, 1)
        return self

    def get_t_initial(self):
        """Get initial temperature.

        Returns:
            float: Value of initial temperature.
        """
        return self.thisptr.getTInitial()

    def get_t_final(self):
        """Get final temperature.

        Returns:
            float: Value of final temperature.
        """
        return self.thisptr.getTFinal()

    def get_scale(self):
        """Get scale.

        Returns:
            float: Value of scale.
        """
        return self.thisptr.getScale()

    def get_cubatic_order_parameter(self):
        """Get cubatic order parameter.

        Returns:
            float: Cubatic order parameter.
        """
        return self.thisptr.getCubaticOrderParameter()

    def get_orientation(self):
        """Get orientations.

        Returns:
            :math:`\\left(4 \\right)` :class:`numpy.ndarray`:
                Orientation of global orientation.
        """
        cdef quat[float] q = self.thisptr.getCubaticOrientation()
        cdef np.ndarray[float, ndim=1] result = np.array(
            [q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)
        return result

    def get_particle_op(self):
        """Get per-particle order parameter.

        Returns:
            :class:`np.ndarray`: Cubatic order parameter.
        """
        cdef float * particle_op = \
            self.thisptr.getParticleCubaticOrderParameter().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32,
                                         <void*> particle_op)
        return result

    def get_particle_tensor(self):
        """Get per-particle cubatic tensor.

        Returns:
            :math:`\\left(N_{particles}, 3, 3, 3, 3 \\right)` \
            :class:`numpy.ndarray`:
                Rank 5 tensor corresponding to each individual particle
                orientation.
        """
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

    def get_global_tensor(self):
        """Get global tensor.

        Returns:
            :math:`\\left(3, 3, 3, 3 \\right)` :class:`numpy.ndarray`:
                Rank 4 tensor corresponding to global orientation.
        """
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

    def get_cubatic_tensor(self):
        """Get cubatic tensor.

        Returns:
            :math:`\\left(3, 3, 3, 3 \\right)` :class:`numpy.ndarray`:
                Rank 4 tensor corresponding to cubatic tensor.
        """
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

    def get_gen_r4_tensor(self):
        """Get R4 Tensor.

        Returns:
            :math:`\\left(3, 3, 3, 3 \\right)` :class:`numpy.ndarray`:
                Rank 4 tensor corresponding to each individual particle
                orientation.
        """
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


cdef class NematicOrderParameter:
    """Compute the nematic order parameter for a system of particles.

    .. moduleauthor:: Jens Glaser <jsglaser@umich.edu>

    .. versionadded:: 0.7.0

    Args:
        u (:math:`\\left(3 \\right)` :class:`numpy.ndarray`):
            The nematic director of a single particle in the reference state
            (without any rotation applied).
    """
    cdef freud._order.NematicOrderParameter *thisptr

    def __cinit__(self, u):
        # run checks
        if len(u) != 3:
            raise ValueError('u needs to be a three-dimensional vector')

        cdef np.ndarray[np.float32_t, ndim=1] l_u = \
            np.array(u, dtype=np.float32)
        self.thisptr = new freud._order.NematicOrderParameter(
            (<vec3[float]*> l_u.data)[0])

    def compute(self, orientations):
        """Calculates the per-particle and global order parameter.

        Args:
            orientations (:math:`\\left(N_{particles}, 4 \\right)` \
            :class:`numpy.ndarray`):
                Orientations to calculate the order parameter.
        """
        orientations = freud.common.convert_array(
            orientations, 2, dtype=np.float32, contiguous=True,
            array_name="orientations")
        if orientations.shape[1] != 4:
            raise TypeError('orientations should be an Nx4 array')

        cdef np.ndarray[float, ndim=2] l_orientations = orientations
        cdef unsigned int num_particles = <unsigned int> orientations.shape[0]

        with nogil:
            self.thisptr.compute(<quat[float]*> l_orientations.data,
                                 num_particles)

    def get_nematic_order_parameter(self):
        """The nematic order parameter.

        Returns:
            float: Nematic order parameter.
        """
        return self.thisptr.getNematicOrderParameter()

    def get_director(self):
        """The director (eigenvector corresponding to the order parameter).

        Returns:
            :math:`\\left(3 \\right)` :class:`numpy.ndarray`:
                The average nematic director.
        """
        cdef vec3[float] n = self.thisptr.getNematicDirector()
        cdef np.ndarray[np.float32_t, ndim=1] result = np.array(
            [n.x, n.y, n.z], dtype=np.float32)
        return result

    def get_particle_tensor(self):
        """The full per-particle tensor of orientation information.

        Returns:
            :math:`\\left(N_{particles}, 3, 3 \\right)` :class:`numpy.ndarray`:
                3x3 matrix corresponding to each individual particle
                orientation.
        """
        cdef float *particle_tensor = self.thisptr.getParticleTensor().get()
        cdef np.npy_intp nbins[3]
        nbins[0] = <np.npy_intp> self.thisptr.getNumParticles()
        nbins[1] = <np.npy_intp> 3
        nbins[2] = <np.npy_intp> 3
        cdef np.ndarray[np.float32_t, ndim=3] result = \
            np.PyArray_SimpleNewFromData(3, nbins, np.NPY_FLOAT32,
                                         <void*> particle_tensor)
        return result

    def get_nematic_tensor(self):
        """The nematic Q tensor.

        Returns:
            :math:`\\left(3, 3 \\right)` :class:`numpy.ndarray`:
                3x3 matrix corresponding to the average particle orientation.
        """
        cdef float *nematic_tensor = self.thisptr.getNematicTensor().get()
        cdef np.npy_intp nbins[2]
        nbins[0] = <np.npy_intp> 3
        nbins[1] = <np.npy_intp> 3
        cdef np.ndarray[np.float32_t, ndim=2] result = \
            np.PyArray_SimpleNewFromData(2, nbins, np.NPY_FLOAT32,
                                         <void*> nematic_tensor)
        return result

cdef class HexOrderParameter:
    """Calculates the :math:`k`-atic order parameter for each particle in the
    system.

    The :math:`k`-atic order parameter for a particle :math:`i` and its
    :math:`n` neighbors :math:`j` is given by:

    :math:`\\psi_k \\left( i \\right) = \\frac{1}{n}
    \\sum_j^n e^{k i \\phi_{ij}}`

    The parameter :math:`k` governs the symmetry of the order parameter while
    the parameter :math:`n` governs the number of neighbors of particle
    :math:`i` to average over. :math:`\\phi_{ij}` is the angle between the
    vector :math:`r_{ij}` and :math:`\\left( 1,0 \\right)`.

    .. note::
        2D: :py:class:`freud.cluster.Cluster` properly handles 2D boxes.
        The points must be passed in as :code:`[x, y, 0]`.
        Failing to set z=0 will lead to undefined behavior.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        rmax (float):
            +/- r distance to search for neighbors.
        k (unsigned int):
            Symmetry of order parameter (:math:`k=6` is hexatic).
        n (unsigned int):
            Number of neighbors (:math:`n=k` if :math:`n` not specified).

    Attributes:
        psi (:math:`\\left(N_{particles} \\right)` :class:`numpy.ndarray`):
            Order parameter.
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        k (unsigned int):
            Symmetry of the order parameter.
    """
    cdef freud._order.HexOrderParameter * thisptr
    cdef num_neigh
    cdef rmax

    def __cinit__(self, rmax, k=int(6), n=int(0)):
        self.thisptr = new freud._order.HexOrderParameter(rmax, k, n)
        self.rmax = rmax
        self.num_neigh = (n if n else int(k))

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, nlist=None):
        """Calculates the correlation function and adds to the current
        histogram.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        box = freud.common.convert_box(box)
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist_nn(
            box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef freud._box.Box l_box = freud._box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        with nogil:
            self.thisptr.compute(
                l_box, nlist_ptr, <vec3[float]*> l_points.data, nP)
        return self

    @property
    def psi(self):
        return self.getPsi()

    def getPsi(self):
        """Get the order parameter.

        Returns:
            :math:`\\left(N_{particles} \\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float complex * psi = self.thisptr.getPsi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> psi)
        return result

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
          :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(< freud._box.Box > self.thisptr.getBox())

    @property
    def num_particles(self):
        return self.getNP()

    def getNP(self):
        """Get the number of particles.

        Returns:
          unsigned int: :math:`N_{particles}`.
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

    @property
    def k(self):
        return self.getK()

    def getK(self):
        """Get the symmetry of the order parameter.

        Returns:
          unsigned int: :math:`k`.
        """
        cdef unsigned int k = self.thisptr.getK()
        return k

cdef class TransOrderParameter:
    """Compute the translational order parameter for each particle.

    .. moduleauthor:: Michael Engel <engelmm@umich.edu>

    Args:
        rmax (float):
            +/- r distance to search for neighbors.
        k (float):
            Symmetry of order parameter (:math:`k=6` is hexatic).
        n (unsigned int):
            Number of neighbors (:math:`n=k` if :math:`n` not specified).

    Attributes:
        d_r (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            Reference to the last computed translational order array.
        box (:py:class:`freud.box.Box`):
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
        """Calculates the local descriptors.

        Args:
            box (:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        box = freud.common.convert_box(box)
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True, array_name="points")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef freud._box.Box l_box = freud._box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist_nn(
            box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        with nogil:
            self.thisptr.compute(
                l_box, nlist_ptr, <vec3[float]*> l_points.data, nP)
        return self

    @property
    def d_r(self):
        return self.getDr()

    def getDr(self):
        """Get a reference to the last computed spherical harmonic array.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float complex * dr = self.thisptr.getDr().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> dr)
        return result

    @property
    def box(self):
        return self.getBox()

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
            :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(< freud._box.Box > self.thisptr.getBox())

    @property
    def num_particles(self):
        return self.getNP()

    def getNP(self):
        """Get the number of particles.

        Returns:
            unsigned int: :math:`N_{particles}`.
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class LocalQl:
    """
    Compute the local Steinhardt [Steinhardt1983]_ rotationally invariant
    :math:`Q_l` [Lechner2008]_ order parameter for a set of points.

    Implements the local rotationally invariant :math:`Q_l` order parameter
    described by Steinhardt. For a particle i, we calculate the average
    :math:`Q_l` by summing the spherical harmonics between particle :math:`i`
    and its neighbors :math:`j` in a local region:
    :math:`\\overline{Q}_{lm}(i) = \\frac{1}{N_b}
    \\displaystyle\\sum_{j=1}^{N_b} Y_{lm}(\\theta(\\vec{r}_{ij}),
    \\phi(\\vec{r}_{ij}))`.

    This is then combined in a rotationally invariant fashion to remove local
    orientational order as follows: :math:`Q_l(i)=\\sqrt{\\frac{4\pi}{2l+1}
    \\displaystyle\\sum_{m=-l}^{l} |\\overline{Q}_{lm}|^2 }`.

    Added first/second shell combined average :math:`Q_l` order parameter for
    a set of points:

    * Variation of the Steinhardt :math:`Q_l` order parameter
    * For a particle i, we calculate the average :math:`Q_l` by summing the
      spherical harmonics between particle i and its neighbors j and the
      neighbors k of neighbor j in a local region.

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:py:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near the first
            minimum of the RDF are recommended.
        l (unsigned int):
            Spherical harmonic quantum number l. Must be a positive number.
        rmin (float):
            Can look at only the second shell or some arbitrary RDF region.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`\\bar{Q_l}` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Ql (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The last computed :math:`\\bar{Q_l}` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).

    .. todo:: move box to compute, this is old API
    """ # noqa
    cdef freud._order.LocalQl * qlptr
    cdef m_box
    cdef rmax

    def __cinit__(self, box, rmax, l, rmin=0):
        box = freud.common.convert_box(box)
        cdef freud._box.Box l_box
        if type(self) is LocalQl:
            l_box = freud._box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
            self.m_box = box
            self.rmax = rmax
            self.qlptr = new freud._order.LocalQl(l_box, rmax, l, rmin)

    def __dealloc__(self):
        if type(self) is LocalQl:
            del self.qlptr
            self.qlptr = NULL

    @property
    def box(self):
        return self.getBox()

    @box.setter
    def box(self, value):
        self.setBox(value)

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
            :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(< freud._box.Box > self.qlptr.getBox())

    def setBox(self, box):
        """Reset the simulation box.

        Args:
            box (:class:`freud.box.Box`): Simulation box.
        """
        box = freud.common.convert_box(box)
        cdef freud._box.Box l_box = freud._box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.qlptr.setBox(l_box)

    @property
    def num_particles(self):
        return self.getNP()

    def getNP(self):
        """Get the number of particles.

        Returns:
            unsigned int: :math:`N_{particles}`.
        """
        cdef unsigned int np = self.qlptr.getNP()
        return np

    @property
    def Ql(self):
        return self.getQl()

    def getQl(self):
        """Get a reference to the last computed :math:`Q_l` for each particle.
        Returns NaN instead of :math:`Q_l` for particles with no neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float * Ql = self.qlptr.getQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[float, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> Ql)
        return result

    @property
    def ave_Ql(self):
        return self.getAveQl()

    def getAveQl(self):
        """Get a reference to the last computed :math:`Q_l` for each particle.
        Returns NaN instead of :math:`Q_l` for particles with no neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float * Ql = self.qlptr.getAveQl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[float, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> Ql)
        return result

    @property
    def norm_Ql(self):
        return self.getQlNorm()

    def getQlNorm(self):
        """Get a reference to the last computed :math:`Q_l` for each particle.
        Returns NaN instead of :math:`Q_l` for particles with no neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float * Ql = self.qlptr.getQlNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[float, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> Ql)
        return result

    @property
    def ave_norm_Ql(self):
        return self.getQlAveNorm()

    def getQlAveNorm(self):
        """Get a reference to the last computed :math:`Q_l` for each particle.
        Returns NaN instead of :math:`Q_l` for particles with no neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float * Ql = self.qlptr.getQlAveNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[float, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_FLOAT32, <void*> Ql)
        return result

    def compute(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

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
        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        self.qlptr.compute(nlist_ptr, <vec3[float]*> l_points.data, nP)
        return self

    def computeAve(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

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

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        self.qlptr.compute(nlist_ptr, <vec3[float]*> l_points.data, nP)
        self.qlptr.computeAve(nlist_ptr, <vec3[float]*> l_points.data, nP)
        return self

    def computeNorm(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

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

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        self.qlptr.compute(nlist_ptr, <vec3[float]*> l_points.data, nP)
        self.qlptr.computeNorm(<vec3[float]*> l_points.data, nP)
        return self

    def computeAveNorm(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

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

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        self.qlptr.compute(nlist_ptr, <vec3[float]*> l_points.data, nP)
        self.qlptr.computeAve(nlist_ptr, <vec3[float]*> l_points.data, nP)
        self.qlptr.computeAveNorm(<vec3[float]*> l_points.data, nP)
        return self

cdef class LocalQlNear(LocalQl):
    """
    Compute the local Steinhardt [Steinhardt1983]_ rotationally invariant
    :math:`Q_l` order parameter [Lechner2008]_ for a set of points.

    Implements the local rotationally invariant :math:`Q_l` order parameter
    described by Steinhardt. For a particle i, we calculate the average
    :math:`Q_l` by summing the spherical harmonics between particle :math:`i`
    and its neighbors :math:`j` in a local region:
    :math:`\\overline{Q}_{lm}(i) = \\frac{1}{N_b}
    \\displaystyle\\sum_{j=1}^{N_b} Y_{lm}(\\theta(\\vec{r}_{ij}),
    \\phi(\\vec{r}_{ij}))`

    This is then combined in a rotationally invariant fashion to remove local
    orientational order as follows: :math:`Q_l(i)=\\sqrt{\\frac{4\pi}{2l+1}
    \\displaystyle\\sum_{m=-l}^{l} |\\overline{Q}_{lm}|^2 }`

    Added first/second shell combined average :math:`Q_l` order parameter for
    a set of points:

    * Variation of the Steinhardt :math:`Q_l` Order parameter.
    * For a particle i, we calculate the average :math:`Q_l` by summing the
      spherical harmonics between particle i and its neighbors j and the
      neighbors k of neighbor j in a local region.

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:py:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near the first
            minimum of the RDF are recommended.
        l (unsigned int):
            Spherical harmonic quantum number l. Must be a positive number.
        kn (unsigned int):
            Number of nearest neighbors. must be a positive integer.

    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`\\bar{Q_l}` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Ql (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The last computed :math:`\\bar{Q_l}` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).

    .. todo:: move box to compute, this is old API
    """ # noqa
    cdef num_neigh

    def __cinit__(self, box, rmax, l, kn=12):
        # Note that we cannot leverage super here because the
        # type conditional in the parent will prevent it.
        # Unfortunately, this is necessary for proper memory
        # management in this inheritance structure.
        cdef freud._box.Box l_box
        if type(self) == LocalQlNear:
            box = freud.common.convert_box(box)
            l_box = freud._box.Box(
                box.getLx(), box.getLy(), box.getLz(),
                box.getTiltFactorXY(), box.getTiltFactorXZ(),
                box.getTiltFactorYZ(), box.is2D())
            self.qlptr = new freud._order.LocalQl(l_box, rmax, l, 0)
            self.m_box = box
            self.rmax = rmax
            self.num_neigh = kn

    def __dealloc__(self):
        if type(self) == LocalQlNear:
            del self.qlptr
            self.qlptr = NULL

    def computeAve(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalQlNear, self).computeAve(points, nlist_)

    def computeNorm(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalQlNear, self).computeNorm(points, nlist_)

    def computeAveNorm(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalQlNear, self).computeAveNorm(points, nlist_)

cdef class LocalWl(LocalQl):
    """
    Compute the local Steinhardt [Steinhardt1983]_ rotationally invariant
    :math:`W_l` order parameter [Lechner2008]_ for a set of points.

    Implements the local rotationally invariant :math:`W_l` order parameter
    described by Steinhardt that can aid in distinguishing  between FCC, HCP,
    and BCC.

    Added first/second shell combined average :math:`W_l` order parameter for
    a set of points:

    * Variation of the Steinhardt :math:`W_l` order parameter.
    * For a particle i, we calculate the average :math:`W_l` by summing the
      spherical harmonics between particle i and its neighbors j and the
      neighbors k of neighbor j in a local region.

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:py:class:`freud.box.Box`):
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
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`\\bar{Q_l}` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Ql (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The last computed :math:`\\bar{Q_l}` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).
        Wl (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`W_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Wl (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`\\bar{W}_l` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Wl (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`W_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Wl (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The last computed :math:`\\bar{W}_l` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).

    .. todo:: move box to compute, this is old API
    """
    cdef freud._order.LocalWl * thisptr

    # List of Ql attributes to remove
    delattrs = ['Ql', 'getQl',
                'ave_Ql', 'getAveQl',
                'norm_Ql', 'getQlNorm',
                'ave_norm_Ql', 'getQlAveNorm']

    def __cinit__(self, box, rmax, l, rmin=0, *args, **kwargs):
        cdef freud._box.Box l_box
        if type(self) is LocalWl:
            l_box = freud._box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
            self.thisptr = self.qlptr = new freud._order.LocalWl(
                l_box, rmax, l, rmin)
            self.m_box = box
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
        """Get a reference to the last computed :math:`W_l` for each particle.
        Returns NaN instead of :math:`W_l` for particles with no neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float complex * Wl = self.thisptr.getWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Wl)
        return result

    @property
    def ave_Wl(self):
        return self.getAveWl()

    def getAveWl(self):
        """Get a reference to the last computed :math:`W_l` for each particle.
        Returns NaN instead of :math:`W_l` for particles with no neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float complex * Wl = self.thisptr.getAveWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Wl)
        return result

    @property
    def norm_Wl(self):
        return self.getWlNorm()

    def getWlNorm(self):
        """Get a reference to the last computed :math:`W_l` for each particle.
        Returns NaN instead of :math:`W_l` for particles with no neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float complex * Wl = self.thisptr.getWlNorm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Wl)
        return result

    @property
    def ave_norm_Wl(self):
        return self.getWlAveNorm()

    def getWlAveNorm(self):
        """Get a reference to the last computed :math:`W_l` for each particle.
        Returns NaN instead of :math:`W_l` for particles with no neighbors.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float complex * Wl = self.thisptr.getAveNormWl().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.qlptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Wl)
        return result

cdef class LocalWlNear(LocalWl):
    """
    Compute the local Steinhardt [Steinhardt1983]_ rotationally invariant
    :math:`W_l` order parameter [Lechner2008]_ for a set of points.

    Implements the local rotationally invariant :math:`W_l` order parameter
    described by Steinhardt that can aid in distinguishing between FCC, HCP,
    and BCC.

    Added first/second shell combined average :math:`W_l` order parameter for a
    set of points:

    * Variation of the Steinhardt :math:`W_l` order parameter.
    * For a particle i, we calculate the average :math:`W_l` by summing the
      spherical harmonics between particle i and its neighbors j and the
      neighbors k of neighbor j in a local region.

    .. moduleauthor:: Xiyu Du <xiyudu@umich.edu>
    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>

    Args:
        box (:py:class:`freud.box.Box`):
            Simulation box.
        rmax (float):
            Cutoff radius for the local order parameter. Values near the first
            minimum of the RDF are recommended.
        l (unsigned int):
            Spherical harmonic quantum number l. Must be a positive number
        kn (unsigned int):
            Number of nearest neighbors. Must be a positive number.


    Attributes:
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        num_particles (unsigned int):
            Number of particles.
        Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`\\bar{Q_l}` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Ql (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Ql (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The last computed :math:`\\bar{Q_l}` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).
        Wl (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`W_l` for each particle (filled with NaN
            for particles with no neighbors).
        ave_Wl (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`\\bar{W}_l` for each particle (filled with
            NaN for particles with no neighbors).
        norm_Wl (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`W_l` for each particle normalized by the
            value over all particles (filled with NaN for particles with no
            neighbors).
        ave_norm_Wl (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The last computed :math:`\\bar{W}_l` for each particle normalized
            by the value over all particles (filled with NaN for particles with
            no neighbors).

    .. todo:: move box to compute, this is old API
    """
    cdef num_neigh

    def __cinit__(self, box, rmax, l, kn=12):
        box = freud.common.convert_box(box)
        cdef freud._box.Box l_box
        if type(self) is LocalWlNear:
            l_box = freud._box.Box(
                box.getLx(), box.getLy(), box.getLz(),
                box.getTiltFactorXY(), box.getTiltFactorXZ(),
                box.getTiltFactorYZ(), box.is2D())
            self.thisptr = self.qlptr = new freud._order.LocalWl(l_box, rmax, l, 0)
            self.m_box = box
            self.rmax = rmax
            self.num_neigh = kn

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def computeAve(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalWlNear, self).computeAve(points, nlist_)

    def computeNorm(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalWlNear, self).computeNorm(points, nlist_)

    def computeAveNorm(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`, optional):
                Neighborlist to use to find bonds (Default value = None).
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return super(LocalWlNear, self).computeAveNorm(points, nlist_)

cdef class SolLiq:
    """
    Computes dot products of :math:`Q_{lm}` between particles and uses these
    for clustering.

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>

    Args:
        box (:py:class:`freud.box.Box`):
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
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        largest_cluster_size (unsigned int):
            The largest cluster size. Must call a compute method first.
        cluster_sizes (unsigned int):
            The sizes of all clusters.
        largest_cluster_size (unsigned int):
            The largest cluster size. Must call a compute method first.
        Ql_mi (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_{lmi}` for each particle.
        clusters (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed set of solid-like cluster indices for each
            particle.
        num_connections (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The number of connections per particle.
        Ql_dot_ij (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            Reference to the qldot_ij values.
        num_particles (unsigned int):
            Number of particles.

    .. todo:: move box to compute, this is old API
    """
    cdef freud._order.SolLiq * thisptr
    cdef m_box
    cdef rmax

    def __init__(self, box, rmax, Qthreshold, Sthreshold, l):
        box = freud.common.convert_box(box)
        cdef freud._box.Box l_box = freud._box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new freud._order.SolLiq(
            l_box, rmax, Qthreshold, Sthreshold, l)
        self.m_box = box
        self.rmax = rmax

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def compute(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

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

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        self.thisptr.compute(nlist_ptr, <vec3[float]*> l_points.data, nP)
        return self

    def computeSolLiqVariant(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

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

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        self.thisptr.computeSolLiqVariant(
            nlist_ptr, <vec3[float]*> l_points.data, nP)
        return self

    def computeSolLiqNoNorm(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

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

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef unsigned int nP = <unsigned int> points.shape[0]

        defaulted_nlist = make_default_nlist(
            self.m_box, points, points, self.rmax, nlist, True)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        cdef freud._locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        self.thisptr.computeSolLiqNoNorm(
            nlist_ptr, <vec3[float]*> l_points.data, nP)
        return self

    @property
    def box(self):
        return self.getBox()

    @box.setter
    def box(self, value):
        self.setBox(value)

    def getBox(self):
        """Get the box used in the calculation.

        Returns:
            :class:`freud.box.Box`: freud Box.
        """
        return BoxFromCPP(< freud._box.Box > self.thisptr.getBox())

    def setClusteringRadius(self, rcutCluster):
        """Reset the clustering radius.

        Args:
            rcutCluster (float): Radius for the cluster finding.
        """
        self.thisptr.setClusteringRadius(rcutCluster)

    def setBox(self, box):
        """Reset the simulation box.

        Args:
            box(:class:`freud.box.Box`): Simulation box.
        """
        box = freud.common.convert_box(box)
        cdef freud._box.Box l_box = freud._box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr.setBox(l_box)

    @property
    def largest_cluster_size(self):
        return self.getLargestClusterSize()

    def getLargestClusterSize(self):
        """Returns the largest cluster size. Must call a compute method first.

        Returns:
            unsigned int: Largest cluster size.
        """
        cdef unsigned int clusterSize = self.thisptr.getLargestClusterSize()
        return clusterSize

    @property
    def cluster_sizes(self):
        return self.getClusterSizes()

    def getClusterSizes(self):
        """Return the sizes of all clusters.

        Returns:
            :math:`\\left(N_{clusters}\\right)` :class:`numpy.ndarray`:
                The cluster sizes.

        .. todo:: unsure of the best way to pass back, as this doesn't do
                  what I want
        """
        cdef vector[unsigned int] clusterSizes = self.thisptr.getClusterSizes()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNumClusters()
        cdef np.ndarray[np.uint32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32,
                                         <void*> &clusterSizes)
        return result

    @property
    def Ql_mi(self):
        return self.getQlmi()

    def getQlmi(self):
        """Get a reference to the last computed :math:`Q_{lmi}` for each
        particle.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Order parameter.
        """
        cdef float complex * Qlmi = self.thisptr.getQlmi().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> Qlmi)
        return result

    @property
    def clusters(self):
        return self.getClusters()

    def getClusters(self):
        """Get a reference to the last computed set of solid-like cluster
        indices for each particle.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Clusters.
        """
        cdef unsigned int * clusters = self.thisptr.getClusters().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32,
                                         <void*> clusters)
        return result

    @property
    def num_connections(self):
        return self.getNumberOfConnections()

    def getNumberOfConnections(self):
        """Get a reference to the number of connections per particle.

        Returns:
            :math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
                Clusters.
        """
        cdef unsigned int * connections = \
            self.thisptr.getNumberOfConnections().get()
        cdef np.npy_intp nbins[1]
        # this is the correct number
        nbins[0] = <np.npy_intp> self.thisptr.getNP()
        cdef np.ndarray[np.uint32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_UINT32,
                                         <void*> connections)
        return result

    @property
    def Ql_dot_ij(self):
        return self.getNumberOfConnections()

    def getQldot_ij(self):
        """Get a reference to the qldot_ij values.

        Returns:
            :math:`\\left(N_{clusters}\\right)` :class:`numpy.ndarray`:
                The qldot values.

        .. todo:: Figure out the size of this because apparently its size is
                  just its size
        """
        cdef vector[float complex] Qldot = self.thisptr.getQldot_ij()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp> self.thisptr.getNumClusters()
        cdef np.ndarray[np.complex64_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(1, nbins, np.NPY_COMPLEX64,
                                         <void*> &Qldot)
        return result

    @property
    def num_particles(self):
        return self.getNP()

    def getNP(self):
        """Get the number of particles.

        Returns:
          unsigned int: :math:`N_p`.
        """
        cdef unsigned int np = self.thisptr.getNP()
        return np

cdef class SolLiqNear(SolLiq):
    """
    Computes dot products of :math:`Q_{lm}` between particles and uses these
    for clustering.

    .. moduleauthor:: Richmond Newman <newmanrs@umich.edu>

    Args:
        box (:py:class:`freud.box.Box`):
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
        box (:py:class:`freud.box.Box`):
            Box used in the calculation.
        largest_cluster_size (unsigned int):
            The largest cluster size. Must call a compute method first.
        cluster_sizes (unsigned int):
            The sizes of all clusters.
        largest_cluster_size (unsigned int):
            The largest cluster size. Must call a compute method first.
        Ql_mi (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed :math:`Q_{lmi}` for each particle.
        clusters (:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`):
            The last computed set of solid-like cluster indices for each
            particle.
        num_connections (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            The number of connections per particle.
        Ql_dot_ij (:math:`\\left(N_{particles}\\right)` \
        :class:`numpy.ndarray`):
            Reference to the qldot_ij values.
        num_particles (unsigned int):
            Number of particles.

    .. todo:: move box to compute, this is old API
    """
    cdef num_neigh

    def __init__(self, box, rmax, Qthreshold, Sthreshold, l, kn=12):
        box = freud.common.convert_box(box)
        cdef freud._box.Box l_box = freud._box.Box(
            box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
            box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())
        self.thisptr = new freud._order.SolLiq(
            l_box, rmax, Qthreshold, Sthreshold, l)
        self.m_box = box
        self.rmax = rmax
        self.num_neigh = kn

    def __dealloc__(self):
        del self.thisptr
        self.thisptr = NULL

    def compute(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return SolLiq.compute(self, points, nlist_)

    def computeSolLiqVariant(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return SolLiq.computeSolLiqVariant(self, points, nlist_)

    def computeSolLiqNoNorm(self, points, nlist=None):
        """Compute the local rotationally invariant :math:`Q_l` order
        parameter.

        Args:
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the order parameter.
            nlist (:class:`freud.locality.NeighborList`):
                Neighborlist to use to find bonds.
        """
        defaulted_nlist = make_default_nlist_nn(
            self.m_box, points, points, self.num_neigh, nlist, True, self.rmax)
        cdef NeighborList nlist_ = defaulted_nlist[0]
        return SolLiq.computeSolLiqNoNorm(self, points, nlist_)


class BondOrder(_EBO):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:class:`freud.environment.BondOrder` instead.

    .. deprecated:: 0.8.2
       Use :py:class:`freud.environment.BondOrder` instead.

    """
    def __init__(self, rmax, k, n, n_bins_t, n_bins_p):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.BondOrder instead!",
                      FreudDeprecationWarning)


class LocalDescriptors(_ELD):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:class:`freud.environment.LocalDescriptors` instead.

    .. deprecated:: 0.8.2
       Use :py:class:`freud.environment.LocalDescriptors` instead.

    """
    def __init__(self, num_neighbors, lmax, rmax, negative_m=True):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.LocalDescriptors instead!",
                      FreudDeprecationWarning)


class MatchEnv(_EME):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:class:`freud.environment.MatchEnv` instead.

    .. deprecated:: 0.8.2
       Use :py:class:`freud.environment.MatchEnv` instead.

    """
    def __init__(self, box, rmax, k):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.MatchEnv instead!",
                      FreudDeprecationWarning)


class Pairing2D(_EP):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:mod:`freud.bond` instead.

    .. deprecated:: 0.8.2
       Use :py:mod:`freud.bond` instead.

    """
    def __init__(self, rmax, k, compDotTol):
        warnings.warn("This class is deprecated, use "
                      "freud.bond instead!", FreudDeprecationWarning)


class AngularSeparation(_EAS):
    """
    .. note::
        This class is only retained for backwards compatibility.
        Please use :py:class:`freud.environment.AngularSeparation` instead.

    .. deprecated:: 0.8.2
       Use :py:class:`freud.environment.AngularSeparation` instead.

    """
    def __init__(self, rmax, n):
        warnings.warn("This class is deprecated, use "
                      "freud.environment.AngularSeparation instead!",
                      FreudDeprecationWarning)
