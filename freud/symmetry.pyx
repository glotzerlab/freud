# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The symmetry module computes symmetries of a system.
"""

import logging

import numpy as np

import freud.common

cimport numpy as np
from cython.operator cimport dereference
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

cimport freud._symmetry
cimport freud.box
cimport freud.locality
from freud.util._VectorMath cimport quat, vec3

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
    cdef freud._symmetry.SymmetryCollection *thisptr

    def __cinit__(self, maxL=int(30)):
        if maxL < 0:
            raise ValueError("maxL must be 0 or greater")
        self.thisptr = new freud._symmetry.SymmetryCollection(maxL)

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

        cdef float[:, ::1] l_points = points
        cdef freud.locality.NeighborList nlist_ = nlist

        cdef unsigned int n_p = l_points.shape[0]

        self.thisptr.compute(
            dereference(b.thisptr),
            <vec3[float]*> &l_points[0, 0],
            nlist_.get_ptr(),
            n_p)
        return self

    def measure(self, int n):
        """Compute symmetry axes.

        Args:
            n (int):
                Order of symmetry to measure along rotated frame's z-axis.
        Returns:
            float:
                Measured :math:`n`-fold symmetry for the rotated frame.
        """
        return self.thisptr.measure(n)

    @property
    def Mlm(self):
        """Get the system average spherical harmonics.

        Returns:
            (:math:`\\left(l_{max} + 1\\right)^2 - 1`) :class:`numpy.ndarray`:
                Spherical harmonic values computed over all bonds.
        """
        cdef unsigned int Mlm_size = (self.thisptr.getMaxL() + 1)**2 - 1
        cdef float[::1] Mlm = <float[:Mlm_size]> self.thisptr.getMlm().get()
        return np.asarray(Mlm)

    @property
    def Mlm_rotated(self):
        """Get the system average spherical harmonics, rotated by a quaternion.

        Returns:
            (:math:`(l_{max} + 1)^2 - 1`) :class:`numpy.ndarray`:
                Spherical harmonic values computed over all bonds.
        """
        cdef unsigned int Mlm_size = (self.thisptr.getMaxL() + 1)**2 - 1
        cdef float[::1] Mlm_rotated = \
            <float[:Mlm_size]> self.thisptr.getMlm_rotated().get()
        return np.asarray(Mlm_rotated)

    @property
    def l_max(self):
        """Returns :math:`l_{max}`.

        Returns:
            int: Maximum :math:`l` value.
        """
        return self.thisptr.getMaxL()

    def rotate(self, q):
        """Rotate ``Mlm`` by ``q``, accessed via the ``Mlm_rotated`` property.

        Args:
            q (:math:`\\left(4 \\right)` :class:`numpy.ndarray`):
                The rotation quaternion.
        """

        q = freud.common.convert_array(
            q, 1, dtype=np.float32, contiguous=True, array_name="q")
        if q.shape[0] != 4:
            raise TypeError('q should be an 1x4 array')
        cdef float[::1] l_q = q
        self.thisptr.rotate(<const quat[float] &> l_q[0])
        return self

    @property
    def symmetries(self):
        """Return the found symmetries. Each symmetry is a ``dict`` with keys
        ``n`` (for an :math:`n`-fold axis), ``vertex`` (a unit vector in the
        direction of the symmetry axis), ``quaternion`` (a unit quaternion that
        rotates the box frame to the symmetry frame), and ``measured_order``
        (a quantity describing the amount of symmetry order, generally but not
        always between 0 and 1).

        Returns:
            list[dict]: Detected symmetry axes.
        """
        cdef vector[freud._symmetry.FoundSymmetry] cpp_symmetries = \
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

    @property
    def laue_group(self):
        """Identify Laue group.

        Returns:
            string: Laue group name.
        """
        cdef unicode laue_group = \
            self.thisptr.getLaueGroup().decode('UTF-8')
        return laue_group

    @property
    def crystal_system(self):
        """Identify crystal system.

        Returns:
            string: Crystal system name.
        """
        cdef unicode crystal_system = \
            self.thisptr.getCrystalSystem().decode('UTF-8')
        return crystal_system


cdef class Geodesation:
    """Computes a geodesation of the sphere, starting from an icosahedron with
    faces iteratively broken into triangles.

    .. moduleauthor:: Bradley Dice <bdice@umich.edu>
    .. moduleauthor:: Yezhi Jin <jinyezhi@umich.edu>
    """
    cdef freud._symmetry.Geodesation *thisptr
    cdef iterations

    def __cinit__(self, iterations):
        self.thisptr = new freud._symmetry.Geodesation(iterations)
        self.iterations = iterations

    def __dealloc__(self):
        del self.thisptr

    @property
    def n_vertices(self):
        """Returns the number of vertices.

        Returns:
            int: Number of vertices
        """
        return self.thisptr.getNVertices()

    @property
    def vertices(self):
        """Return the vertex positions.

        Returns:
            :math:`\\left(N_{vertices}, 3\\right)` :class:`numpy.ndarray`:
                Array of vertex positions.
        """
        cdef vector[vec3[float]] *vertex_list = \
            self.thisptr.getVertexList().get()
        cdef unsigned int vertices_size = vertex_list.size()
        cdef float[:, ::1] vertices = <float[:vertices_size, :3]> (
            <float*> dereference(vertex_list).data())
        return np.asarray(vertices)

    @property
    def neighbor_pairs(self):
        """Return the neighbor pairs.

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
