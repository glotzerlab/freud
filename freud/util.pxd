# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Directly expose vec3 and quat since they're ubiquitous in constructing
# arguments to interface with the C++ implementations of all methods.
import freud
import numpy as np

from freud._util cimport vec3, quat, ManagedArray, PyArray_SetBaseObject
from cpython cimport Py_INCREF
from libcpp.complex cimport complex
from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr

cimport numpy as np

ctypedef unsigned int uint

ctypedef enum arr_type_t:
    UNSIGNED_INT
    FLOAT
    COMPLEX

ctypedef union arr_ptr_t:
    const void *null_ptr
    const ManagedArray[uint] *uint_ptr
    const ManagedArray[float] *float_ptr
    const ManagedArray[float complex] *complex_ptr


cdef class _ManagedArrayContainer:
    cdef uint _element_size
    cdef int var_typenum
    cdef arr_ptr_t thisptr
    cdef arr_type_t data_type

    cdef void set_as_base(self, arr)
    cdef void *get(self)

    @staticmethod
    cdef inline _ManagedArrayContainer init(
            const void *array, arr_type_t arr_type, uint element_size=1):
        cdef _ManagedArrayContainer obj

        if arr_type == arr_type_t.UNSIGNED_INT:
            obj = _ManagedArrayContainer(arr_type, np.NPY_UINT32,
                                         element_size)
            obj.thisptr.uint_ptr = new const ManagedArray[uint](
                dereference(<const ManagedArray[uint] *>array))
        elif arr_type == arr_type_t.FLOAT:
            obj = _ManagedArrayContainer(arr_type, np.NPY_FLOAT,
                                         element_size)
            obj.thisptr.float_ptr = new const ManagedArray[float](
                dereference(<const ManagedArray[float] *>array))
        elif arr_type == arr_type_t.COMPLEX:
            obj = _ManagedArrayContainer(arr_type, np.NPY_COMPLEX64,
                                         element_size)
            obj.thisptr.complex_ptr = new const ManagedArray[float complex](
                dereference(<const ManagedArray[float complex] *>array))

        return obj

cdef inline make_managed_numpy_array(
        const void *array, arr_type_t arr_type, uint element_size=1):
    """Make a _ManagedArrayContainer and return an array pointing to its
    data."""
    return np.asarray(
        _ManagedArrayContainer.init(array, arr_type, element_size))
