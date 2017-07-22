# Copyright (c) 2010-2017 The Regents of the University of Michigan
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

cdef class SymmetricOrientation:
    """Compute the global or local symmetric orientation for the system of particles.

    .. moduleauthor:: Bradley Dice <bdice@umich.edu>

    """
    cdef symmetry.SymmetricOrientation *thisptr

    def __cinit__(self):
        self.thisptr = new symmetry.SymmetricOrientation()

    def __dealloc__(self):
        del self.thisptr

    def get_symmetric_orientation(self):
        """
        :return: orientation of highest symmetry axis
        :rtype: :class:`numpy.ndarray`, shape= :math:`\\left(4 \\right)`, dtype= :class:`numpy.float32`
        """
        cdef quat[float] q = self.thisptr.getSymmetricOrientation()
        cdef np.ndarray[float, ndim=1] result = np.array([q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)
        return result
