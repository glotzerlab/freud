# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Directly expose vec3 and quat since they're ubiquitous in constructing
# arguments to interface with the C++ implementations of all methods.
from freud._util cimport vec3, quat, ManagedArray, PyArray_SetBaseObject
from cpython cimport Py_INCREF

ctypedef unsigned int uint

cdef class ManagedArrayWrapper:
    cdef int var_typenum
    cdef unsigned int ndim
    cdef ManagedArray[uint] *thisptr

    cdef inline void set_as_base(self, arr):
        """Sets the base of arr to be this object and increases the
        reference count."""
        PyArray_SetBaseObject(arr, self)
        Py_INCREF(self)

    cdef inline uint *get(self):
        return self.thisptr.get()

    @staticmethod
    cdef inline ManagedArrayWrapper init(ManagedArray[uint] &other, int typenum, unsigned int ndim):
        obj = ManagedArrayWrapper(typenum, ndim)

        obj.thisptr = ManagedArray[uint].createAndAcquire(other)

        return obj
