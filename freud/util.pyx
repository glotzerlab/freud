# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from cython.operator cimport dereference
cimport freud.util

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class ManagedArrayWrapper:
    def __cinit__(self, typenum, ndim):
        self.var_typenum = typenum
        self.ndim = ndim
        if typenum == np.NPY_UINT32:
            self.thisptr.uint_ptr = new ManagedArray[uint]()
            self.sourceptr.uint_ptr = NULL

    def acquire(self):
        if self.var_typenum == np.NPY_UINT32:
            self.thisptr.uint_ptr.acquire(dereference(self.sourceptr.uint_ptr))

    def release(self):
        if self.var_typenum == np.NPY_UINT32:
            self.sourceptr.uint_ptr.acquire(dereference(self.thisptr.uint_ptr))

    def __dealloc__(self):
        if self.var_typenum == np.NPY_UINT32:
            del self.thisptr.uint_ptr

    def __array__(self):
        """Convert the underlying data array into a read-only numpy array."""
        cdef unsigned int dim1
        if self.var_typenum == np.NPY_UINT32:
            dim1 = self.thisptr.uint_ptr.size()
        cdef np.npy_intp nP[1]
        nP[0] = dim1
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(
            self.ndim, nP, self.var_typenum, self.get())
        arr.setflags(write=False)

        self.set_as_base(arr)
        return arr
