# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud._util cimport vec3, quat, ManagedArray
cimport freud.util

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class ManagedArrayWrapper:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr

    def __array__(self):
        cdef unsigned int dim1 = self.thisptr.size()
        cdef np.npy_intp nP[1]
        nP[0] = dim1
        cdef np.ndarray[np.uint32_t, ndim=1] cluster_idx = np.PyArray_SimpleNewFromData(
            1, nP, np.NPY_UINT32, <void*>self.get())

        freud.util.set_base(cluster_idx, self)
        return cluster_idx
