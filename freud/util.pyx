# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from cython.operator cimport dereference
cimport freud.util

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class ManagedArrayWrapper:
    def __cinit__(self):
        # This class should be initialized via the factory "init" function.
        pass

    def acquire(self):
        if self.var_typenum == np.NPY_UINT32:
            self.thisptr.uint_ptr.acquire(dereference(self.sourceptr.uint_ptr))

    def release(self):
        if self.var_typenum == np.NPY_UINT32:
            self.sourceptr.uint_ptr.acquire(dereference(self.thisptr.uint_ptr))

    def set_shape(self, shape):
        self.shape = shape

    def __dealloc__(self):
        if self.var_typenum == np.NPY_UINT32:
            del self.thisptr.uint_ptr

    def __array__(self):
        """Convert the underlying data array into a read-only numpy array."""
        if self.shape == tuple():
            raise ValueError("You must specify the shape of the numpy array to be created by calling set_shape.")
        cdef unsigned int ndim = len(self.shape)
        cdef np.npy_intp nP1[1]
        cdef np.npy_intp nP2[2]
        cdef np.npy_intp nP3[3]
        cdef np.ndarray arr
        if len(self.shape) == 1:
            nP1[0] = self.shape[0]
            arr = np.PyArray_SimpleNewFromData(
                ndim, nP1, self.var_typenum, self.get())
        elif len(self.shape) == 2:
            nP2[0] = self.shape[0]
            nP2[1] = self.shape[1]
            arr = np.PyArray_SimpleNewFromData(
                ndim, nP2, self.var_typenum, self.get())
        elif len(self.shape) == 3:
            nP3[0] = self.shape[0]
            nP3[1] = self.shape[1]
            nP3[2] = self.shape[2]
            arr = np.PyArray_SimpleNewFromData(
                ndim, nP3, self.var_typenum, self.get())

        arr.setflags(write=False)

        self.set_as_base(arr)
        return arr
