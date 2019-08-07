# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Directly expose vec3 and quat since they're ubiquitous in constructing
# arguments to interface with the C++ implementations of all methods.
from freud._util cimport vec3, quat, ManagedArray, PyArray_SetBaseObject
from cpython cimport Py_INCREF
cimport numpy as np

ctypedef unsigned int uint

ctypedef enum arr_type_t:
    UNSIGNED_INT

ctypedef union arr_ptr_t:
    ManagedArray[uint] *uint_ptr


cdef class ManagedArrayManager:
    cdef int var_typenum
    cdef arr_ptr_t thisptr
    cdef arr_ptr_t sourceptr
    cdef tuple _shape

    cdef inline void set_as_base(self, arr):
        """Sets the base of arr to be this object and increases the
        reference count."""
        PyArray_SetBaseObject(arr, self)
        Py_INCREF(self)

    cdef inline void *get(self):
        """Return the raw pointer to the underlying data array.

        Since the primary purpose of this function is to be passed to the
        Python array generation function, we can just return a void pointer to
        simplify the code.
        """
        return self.thisptr.uint_ptr.get()

    @staticmethod
    cdef inline ManagedArrayManager init(
            void *array, arr_type_t arr_type):
        cdef ManagedArrayManager obj = ManagedArrayManager()

        obj.shape = tuple()
        if arr_type == arr_type_t.UNSIGNED_INT:
            obj.var_typenum = np.NPY_UINT32
            obj.thisptr.uint_ptr = new ManagedArray[uint]()
            obj.sourceptr.uint_ptr = <ManagedArray[uint] *>array

        return obj
