# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

cimport freud.util

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class ManagedArrayWrapper:
    def __cinit__(self, typenum, ndim):
        self.var_typenum = typenum
        self.ndim = ndim
        self.thisptr = NULL

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr

    def __array__(self):
        cdef unsigned int dim1 = self.thisptr.size()
        cdef np.npy_intp nP[1]
        nP[0] = dim1
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(
            self.ndim, nP, self.var_typenum, <void*>self.get())

        self.set_as_base(arr)
        return arr
