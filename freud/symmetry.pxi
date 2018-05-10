# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
cimport freud._box as _box
cimport freud._symmetry as symmetry
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
import numpy as np
cimport numpy as np
import time

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class SymmetryCollection:
    """Compute the global or local symmetric orientation for the system of particles.

    .. moduleauthor:: Bradley Dice <bdice@umich.edu>
    .. moduleauthor:: Yezhi Jin <jinyezhi@umich.edu>

    """
    cdef symmetry.SymmetryCollection *thisptr

    def __cinit__(self):
        self.thisptr = new symmetry.SymmetryCollection()

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


    def getMlm(self):
        """Get a reference to Mlm.

        :return: order parameter
        :rtype: :class:`numpy.ndarray`,
                shape= :math:`\\left(N_{particles}\\right)`,
                dtype= :class:`numpy.complex64`
        """
        cdef float complex * Mlm = self.thisptr.getMlm().get()
        cdef np.npy_intp nbins[1]
        nbins[0] = <np.npy_intp > self.thisptr.getNP()
        cdef np.ndarray[np.complex64_t, ndim= 1
                ] result = np.PyArray_SimpleNewFromData(
                        1, nbins, np.NPY_COMPLEX64, < void*>Mlm)
        return result


    # def get_symmetric_orientation(self):
    #     """
    #     :return: orientation of highest symmetry axis
    #     :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(4 \\right)`, dtype= :class:`numpy.float32`
    #     """
    #     cdef quat[float] q = self.thisptr.getSymmetricOrientation()
    #     cdef np.ndarray[float, ndim=1] result = np.array([q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)
    #     return result
