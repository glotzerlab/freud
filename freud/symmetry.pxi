# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._box as _box
cimport freud._symmetry as symmetry
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
import numpy as np
cimport numpy as np
import time
from cython.operator cimport dereference

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class SymmetryCollection:
    """Compute the global or local symmetric orientation for the system of particles.

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

        :param box: simulation box
        :param points: points to calculate the symmetry axes
        :param nlist: :py:class:`freud.symmetry.SymmetryCollection` object defining bonds
        :type box: :py:class:`freud.box.Box`
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3 \\right)`,
                      dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.symmetry.SymmetryCollection`
        """
        points = freud.common.convert_array(
                points, 2, dtype=np.float32, contiguous=True,
                dim_message="points must be a 2 dimensional array")
        if points.shape[1] != 3:
            raise TypeError('points should be an Nx3 array')

        cdef _box.Box l_box = _box.Box(
                box.getLx(), box.getLy(), box.getLz(), box.getTiltFactorXY(),
                box.getTiltFactorXZ(), box.getTiltFactorYZ(), box.is2D())

        cdef np.ndarray[float, ndim = 2] l_points = points

        cdef NeighborList nlist_ = nlist
        cdef locality.NeighborList * nlist_ptr = nlist_.get_ptr()

        cdef unsigned int nP = <unsigned int > points.shape[0]

        self.thisptr.compute(l_box, < vec3[float]*>l_points.data, nlist_ptr, nP)
        return self


    def measure(self, int n):
        """Compute symmetry axes.

        :param box: simulation box
        :param points: points to calculate the symmetry axes
        :param nlist: :py:class:`freud.symmetry.SymmetryCollection` object defining bonds
        :type box: :py:class:`freud.box.Box`
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3 \\right)`,
                      dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.symmetry.SymmetryCollection`
        """
        return self.thisptr.measure(n)


    def getMlm(self):
        """Get a reference to Mlm.

        :return: Mlm
        :rtype: :class:`numpy.ndarray`,
                shape= :math: (MAXL + 1) ** 2 - 1,
                dtype= :class:`numpy.float32
        """
        cdef float * Mlm = self.thisptr.getMlm().get()
        cdef np.npy_intp Mlm_shape[1]
        Mlm_shape[0] = <np.npy_intp > (self.thisptr.getMaxL() + 1)**2 - 1
        cdef np.ndarray[np.float32_t, ndim= 1
                ] result = np.PyArray_SimpleNewFromData(
                        1, Mlm_shape, np.NPY_FLOAT32, < void*>Mlm)
        return result



    def getMlm_rotated(self):
        """Get a reference to Mlm_rotated.

        :return: Mlm
        :rtype: :class:`numpy.ndarray`,
                shape= :math: (MAXL + 1) ** 2 - 1,
                dtype= :class:`numpy.float32
        """
        cdef float * Mlm_rotated = self.thisptr.getMlm_rotated().get()
        cdef np.npy_intp Mlm_shape[1]
        Mlm_shape[0] = <np.npy_intp > (self.thisptr.getMaxL() + 1)**2 - 1
        cdef np.ndarray[np.float32_t, ndim= 1
                ] result = np.PyArray_SimpleNewFromData(
                        1, Mlm_shape, np.NPY_FLOAT32, < void*>Mlm_rotated)
        return result


    def getMaxL(self):
        """Maximum l value

        :return: maximum l value
        :rtype: int
        """
        return self.thisptr.getMaxL()


    def rotate(self, q):
        """rotate Mlm by q.

        :param q: rotation quaternion
        :type q: class:`numpy.ndarray`,
                      shape= 4 (r, v.x, v.y, v.z),
                      dtype= :class:`numpy.float32`
        :rtype: void
        """

        q = freud.common.convert_array(q, 1, dtype=np.float32, contiguous=True,
                dim_message="q must be a 1x4 dimensional array")
        if q.shape[0] != 4:
            raise TypeError('q should be an 1x4 array')
        cdef np.ndarray[float, ndim= 1] l_q = q

        self.thisptr.rotate(< const quat[float] &> l_q[0])
        return self


   # def getHighestSymmetryQuat(self):

        """find quaternion to the highest-symmetry axis.

        :return: orientation of highest symmetry axis
        :rtype: class:`numpy.ndarray`,
                       shape= 4 (r, v.x, v.y, v.z),
                       dtype= :class:`numpy.float32`

        """
    #    cdef quat[float] q = self.thisptr.getHighestSymmetryQuat()
     #   cdef np.ndarray[float, ndim = 1] result = np.array(
      #          [q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)
       # return result


    @property
    def symmetries(self):
        """Return the found symmetries.
        """
        return self.getSymmetries()

    def getSymmetries(self):
        """Return the found symmetries.

        :return: Symmetry axes detected
        :rtype: :class:`numpy.ndarray`,
                shape=(:math:`N_{particles}`, varying),
                dtype= :class:`numpy.uint32`
        """
        cdef vector[symmetry.FoundSymmetry] cpp_symmetries = self.thisptr.getSymmetries()
        cdef np.ndarray[float, ndim = 1] vert = np.zeros(3, dtype=np.float32)
        cdef np.ndarray[float, ndim = 1] quat = np.zeros(4, dtype=np.float32)
        symmetries = []
        for symm in cpp_symmetries:
            vert = np.array(
                [symm.v.x, symm.v.y, symm.v.z], dtype=np.float32)
            quat = np.array(
                [symm.q.s, symm.q.v.x, symm.q.v.y, symm.q.v.z], dtype=np.float32)
            symmetries.append({
                'n': symm.n,
                'vertex': vert,
                'quaternion': quat,
                'measured_order': symm.measured_order})
        return symmetries


    def getLaueGroup(self):
        cdef string cpp_string = self.thisptr.getLaueGroup()
        return cpp_string.decode('UTF-8')


    # def get_symmetric_orientation(self):
    #     """
    #     :return: orientation of highest symmetry axis
    #     :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(4 \\right)`, dtype= :class:`numpy.float32`
    #     """
    #     cdef quat[float] q = self.thisptr.getSymmetricOrientation()
    #     cdef np.ndarray[float, ndim=1] result = np.array([q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)
    #     return result



cdef class Geodesation:
    """Compute the

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

        :return: the size of vertex list
        :rtype: int
        """
        return self.thisptr.getNVertices()

    @property
    def NVertices(self):

        return self.getNVertices()


    def getVertexList(self):
        """Return the vertex list.

        :return: list of vertices
        :rtype: :class:`numpy.ndarray`,
                shape=(:math:`N_{particles}`, varying),
                dtype= :class:`numpy.uint32`
        """
        cdef vector[vec3[float]] *vertices = self.thisptr.getVertexList().get()
        cdef np.npy_intp nVerts[2]
        nVerts[0] = <np.npy_intp > vertices.size()
        nVerts[1] = 3
        cdef np.ndarray[float, ndim=2
                        ] result = np.PyArray_SimpleNewFromData(
                        2, nVerts, np.NPY_FLOAT32, < void*> dereference(vertices).data())
        return result

    def getNeighborList(self):
        """Return the neighbor list.

        :return: list of neighbors
        :rtype: :class:`numpy.ndarray`,
                shape=(:math:`N_{particles}`, varying),
                dtype= :class:`numpy.uint32`
        """
        cdef vector[unordered_set[int]] network = dereference(self.thisptr.getNeighborList().get())

        result = []
        for i, neighbors in enumerate(network):
            for j in neighbors:
                result.append([i, j])
        return result


