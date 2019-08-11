# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Directly expose vec3 and quat since they're ubiquitous in constructing
# arguments to interface with the C++ implementations of all methods.
from freud._util cimport vec3, quat, ManagedArray, PyArray_SetBaseObject
from cpython cimport Py_INCREF
from cython.operator cimport dereference
from libcpp.memory cimport shared_ptr

cimport numpy as np

ctypedef unsigned int uint

ctypedef enum arr_type_t:
    UNSIGNED_INT

ctypedef union arr_ptr_t:
    void *null_ptr
    ManagedArray[uint] *uint_ptr


cdef class ManagedArrayManager:
    cdef int var_typenum
    cdef arr_ptr_t thisptr
    cdef tuple _shape
    cdef arr_type_t data_type

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
        if self.data_type == arr_type_t.UNSIGNED_INT:
            return self.thisptr.uint_ptr.get()

    cdef inline void dissociate(self):
        """Decouple the underlying ManagedArray from other ManagedArrays
        pointing to the same data.

        Since ManagedArrays are implemented as pointers to pointers to arrays,
        copying a ManagedArray by value copies the first pointer, but all such
        arrays are pointing to the same second level pointer. This mechanisms
        supports all such ManagedArrays sharing ownership of the underlying
        data for the purposes of resizing and reallocation. Here, we take
        advantage of this structure by actually creating a completely new
        ManagedArray, so the second level pointer is also distinct from the
        existing structure. When we reassign it to point to the same data as
        the current ManagedArrayManager's thisptr, we will have a completely
        separate pointer->pointer->array hierarchy pointing to the same array,
        so we can safely resize or reallocate any of the original ManagedArrays
        and this new instance will retain a reference to the original data,
        keeping it alive.
        """
        if self.data_type == arr_type_t.UNSIGNED_INT:
            self.thisptr.uint_ptr = <ManagedArray[uint] *> \
                self.thisptr.uint_ptr.dissociate()

    cdef inline void reallocate(self):
        """Reallocate the data in the underlying array."""
        if self.data_type == arr_type_t.UNSIGNED_INT:
            self.thisptr.uint_ptr.reallocate()

    @staticmethod
    cdef inline ManagedArrayManager init(
            void *array, arr_type_t arr_type):
        cdef ManagedArrayManager obj

        if arr_type == arr_type_t.UNSIGNED_INT:
            obj = ManagedArrayManager(arr_type_t.UNSIGNED_INT, np.NPY_UINT32)
            obj.thisptr.uint_ptr = new ManagedArray[uint](
                dereference(<ManagedArray[uint] *>array))

        return obj
