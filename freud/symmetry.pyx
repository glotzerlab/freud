# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The symmetry module computes symmetries of a system.
"""

import numpy as np
import logging
import freud.common

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from cython.operator cimport dereference

cimport freud._symmetry
cimport freud.box
cimport numpy as np

logger = logging.getLogger(__name__)


# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class SymmetryCollection:
    """Compute the global or local symmetric orientation for the system of
    particles.

    .. moduleauthor:: Bradley Dice <bdice@umich.edu>
    .. moduleauthor:: Yezhi Jin <jinyezhi@umich.edu>

    """
    cdef symmetry.SymmetryCollection *thisptr
    cdef num_neigh

    def __cinit__(self, maxL=int(30)):
        self.thisptr = new symmetry.SymmetryCollection(maxL)
        self.num_neigh = maxL

    def __dealloc__(self):
        del self.thisptr

    def compute(self, box, points, nlist):
        """Compute symmetry axes.

        Args:
            box (py:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the symmetry axes.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = None).
        """
        cdef freud.box.Box b = freud.common.convert_box(box)
        points = freud.common.convert_array(
            points, 2, dtype=np.float32, contiguous=True,
            array_name="points")

        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef np.ndarray[float, ndim=2] l_points = points
        cdef NeighborList nlist_ = nlist

        cdef unsigned int n_p = <unsigned int> points.shape[0]

        self.thisptr.compute(
            dereference(b.thisptr),
            <vec3[float]*> l_points.data,
            nlist_.get_ptr(),
            n_p)
        return self

    def measure(self, int n):
        """Compute symmetry axes.

        Args:
            box (py:class:`freud.box.Box`):
                Simulation box.
            points ((:math:`N_{particles}`, 3) :class:`numpy.ndarray`):
                Points to calculate the symmetry axes.
            nlist (:class:`freud.locality.NeighborList`, optional):
                NeighborList to use to find bonds (Default value = None).
        """
        return self.thisptr.measure(n)

    def getMlm(self):
        """Get a reference to ``Mlm``.

        Returns:
            (:math:`(l_{max} + 1)^2 - 1`) :class:`numpy.ndarray`:
                Spherical harmonic values computed over all bonds.
        """
        cdef float * Mlm = self.thisptr.getMlm().get()
        cdef np.npy_intp Mlm_shape[1]
        Mlm_shape[0] = <np.npy_intp> (self.thisptr.getMaxL() + 1)**2 - 1
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, Mlm_shape, np.NPY_FLOAT32, <void*> Mlm)
        return result

    def getMlm_rotated(self):
        """Get a reference to ``Mlm_rotated``.

        Returns:
            (:math:`(l_{max} + 1)^2 - 1`) :class:`numpy.ndarray`:
                Spherical harmonic values computed over all bonds.
        """
        cdef float * Mlm_rotated = self.thisptr.getMlm_rotated().get()
        cdef np.npy_intp Mlm_shape[1]
        Mlm_shape[0] = <np.npy_intp> (self.thisptr.getMaxL() + 1)**2 - 1
        cdef np.ndarray[np.float32_t, ndim=1] result = \
            np.PyArray_SimpleNewFromData(
                1, Mlm_shape, np.NPY_FLOAT32, <void*> Mlm_rotated)
        return result

    @property
    def l_max(self):
        return self.getMaxL()

    def getMaxL(self):
        """Returns :math:`l_{max}`.

        Returns:
            int: Maximum :math:`l` value.
        """
        return self.thisptr.getMaxL()

    def rotate(self, q):
        """Rotate Mlm by q.

        Args:
            q (:math:`\\left(4 \\right)` :class:`numpy.ndarray`):
                The rotation quaternion.
        """

        q = freud.common.convert_array(
            q, 1, dtype=np.float32, contiguous=True, array_name="q")
        if q.shape[0] != 4:
            raise TypeError('q should be an 1x4 array')
        cdef np.ndarray[float, ndim=1] l_q = q

        self.thisptr.rotate(<const quat[float] &> l_q[0])
        return self

    @property
    def symmetries(self):
        """Return the found symmetries.
        """
        return self.getSymmetries()

    def getSymmetries(self):
        """Return the found symmetries. Each symmetry is a ``dict`` with keys
        ``n`` (for an :math:`n`-fold axis), ``vertex`` (a unit vector in the
        direction of the symmetry axis), ``quaternion`` (a unit quaternion that
        rotates the box frame to the symmetry frame), and ``measured_order``
        (a quantity describing the amount of symmetry order, generally but not
        always between 0 and 1).

        Returns:
            list[dict]: Detected symmetry axes.
        """
        cdef vector[symmetry.FoundSymmetry] cpp_symmetries = \
            self.thisptr.getSymmetries()
        cdef np.ndarray[float, ndim=1] vert = np.zeros(3, dtype=np.float32)
        cdef np.ndarray[float, ndim=1] quat = np.zeros(4, dtype=np.float32)

        symmetries = []
        for symm in cpp_symmetries:
            vert = np.array(
                [symm.v.x, symm.v.y, symm.v.z], dtype=np.float32)
            quat = np.array(
                [symm.q.s, symm.q.v.x, symm.q.v.y, symm.q.v.z],
                dtype=np.float32)
            symmetries.append({
                'n': symm.n,
                'vertex': vert,
                'quaternion': quat,
                'measured_order': symm.measured_order})
        return symmetries

    def getLaueGroup(self):
        """Identify Laue Group.

        Returns:
            string: Laue group name.
        """
        cdef string cpp_string = self.thisptr.getLaueGroup()
        return cpp_string.decode('UTF-8')

    def getCrystalSystem(self):
        """Identify Crystal System.

        Returns:
            string: Crystal system name.
        """
        cdef string cpp_string = self.thisptr.getCrystalSystem()
        return cpp_string.decode('UTF-8')


cdef class Geodesation:
    """Computes a geodesation of the sphere, starting from an icosahedron with
    faces iteratively broken into triangles.

    .. moduleauthor:: Bradley Dice <bdice@umich.edu>
    .. moduleauthor:: Yezhi Jin <jinyezhi@umich.edu>
    """
    cdef symmetry.Geodesation *thisptr
    cdef iterations

    def __cinit__(self, iterations):
        self.thisptr = new symmetry.Geodesation(iterations)
        self.iterations = iterations

    def __dealloc__(self):
        del self.thisptr

    def getNVertices(self):
        """Returns the number of vertices.

        Returns:
            int: Length of the vertex list.
        """
        return self.thisptr.getNVertices()

    @property
    def n_vertices(self):
        return self.getNVertices()

    def getVertexList(self):
        """Return the vertex positions.

        Returns:
            :math:`\\left(N_{vertices}, 3\\right)` :class:`numpy.ndarray`:
                Array of vertex positions.
        """
        cdef vector[vec3[float]] *vertices = self.thisptr.getVertexList().get()
        cdef np.npy_intp nVerts[2]
        nVerts[0] = <np.npy_intp> vertices.size()
        nVerts[1] = 3
        cdef np.ndarray[float, ndim=2] result = \
            np.PyArray_SimpleNewFromData(
                2, nVerts, np.NPY_FLOAT32,
                <void*> dereference(vertices).data())
        return result

    def getNeighborList(self):
        """Return the neighbor list.

        Returns:
            list:
                List of ``[i, j]`` pairs of neighboring sites of the
                geodesation.
        """
        cdef vector[unordered_set[int]] network = \
            dereference(self.thisptr.getNeighborList().get())

        result = []
        for i, neighbors in enumerate(network):
            for j in neighbors:
                result.append([i, j])
        return result
