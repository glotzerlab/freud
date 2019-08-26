# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Directly expose vec3 and quat since they're ubiquitous in constructing
# arguments to interface with the C++ implementations of all methods.
import freud
import numpy as np

from freud._util cimport vec3, quat, ManagedArray, PyArray_SetBaseObject
from cpython cimport Py_INCREF
from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr

cimport numpy as np

ctypedef unsigned int uint

ctypedef enum arr_type_t:
    UNSIGNED_INT

ctypedef union arr_ptr_t:
    const void *null_ptr
    const ManagedArray[uint] *uint_ptr


cdef class ManagedArrayManager:
    cdef int var_typenum
    cdef arr_ptr_t thisptr
    cdef arr_type_t data_type

    cdef void set_as_base(self, arr)
    cdef void *get(self)

    @staticmethod
    cdef inline ManagedArrayManager init(
            const void *array, arr_type_t arr_type):
        cdef ManagedArrayManager obj

        if arr_type == arr_type_t.UNSIGNED_INT:
            obj = ManagedArrayManager(arr_type_t.UNSIGNED_INT, np.NPY_UINT32)
            obj.thisptr.uint_ptr = new const ManagedArray[uint](
                dereference(<const ManagedArray[uint] *>array))

        return obj

cdef inline make_managed_numpy_array(
        const void *array, arr_type_t arr_type):
    """Make a ManagedArrayManager and return an array pointing to its data."""
    return np.asarray(ManagedArrayManager.init(
        array, arr_type_t.UNSIGNED_INT))
