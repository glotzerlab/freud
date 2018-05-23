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
        :param nlist: :py:class:`freud.locality.NeighborList` object defining bonds
        :type box: :py:class:`freud.box.Box`
        :type points: :class:`numpy.ndarray`,
                      shape= :math:`\\left(N_{particles}, 3 \\right)`,
                      dtype= :class:`numpy.float32`
        :type nlist: :py:class:`freud.locality.NeighborList`
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

    #cdef measure(self, shared_ptr[float] Mlm, unsigned int type):
    #    return self.thisptr.measure(Mlm, type)


    def measure(self, Mlm, int type):
        #cdef np.npy_intp nbins[1]
       # nbins[0] = <np.npy_intp > self.thisptr.getMaxL()

        return self.thisptr.measure(self.thisptr.getMlm(), type)


    def getMlm(self):
        """Get a reference to Mlm.

        :return: order parameter
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}\\right)`,
                dtype= :class:`numpy.float64
        """
        cdef float * Mlm = self.thisptr.getMlm().get()
        cdef np.npy_intp Mlm_shape[1]
        Mlm_shape[0] = <np.npy_intp > (self.thisptr.getMaxL() + 1)**2 - 1
        cdef np.ndarray[np.float64_t, ndim= 1
                ] result = np.PyArray_SimpleNewFromData(
                        1, Mlm_shape, np.NPY_FLOAT64, < void*>Mlm)
        return result


    def getMaxL(self):
        """Maximum l value

        :return: maximum l value
        :rtype: int
        """
        return self.thisptr.getMaxL()


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

    def createVertex(self, x, y, z):
        """Returns the index of vertex.

        :return: the index of vertex added to the list
        :rtype: int
        """
        return self.thisptr.createVertex(x, y, z)

    def createSimplex(self, v0, v1, v2):
        """Returns the index of simplex in the list.

        :return: the index of simplex added to the list
        :rtype: int
        """
        return self.thisptr.createSimplex(v0, v1, v2)


    def getNVertices(self):
        """Returns the index of vertex.

        :return: the index of vertex added to the list
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

    def createMidVertex(self, i0, i1):
        """Returns the index of vertex.

        :return: the index of vertex added to the list
        :rtype: int
        """
        return self.thisptr.createMidVertex(i0, i1)

    def connectSimplices(self, s0, s1):
        """Returns the index of vertex.

        :return: the index of vertex added to the list
        :rtype: int
        """
        return self.thisptr.connectSimplices(s0, s1)

    def findNeighborMidVertex(self, points, s):
        """Returns the index of vertex.

        :return: the index of vertex added to the list
        :rtype: int
        """
        return self.thisptr.findNeighborMidVertex(points, s)

    def geodesate(self):
        """Returns the index of vertex.

        :return: the index of vertex added to the list
        :rtype: int
        """
        return self.thisptr.geodesate()

