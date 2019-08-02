# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Directly expose vec3 and quat since they're ubiquitous in constructing
# arguments to interface with the C++ implementations of all methods.
from freud._util cimport vec3, quat, ManagedArray, PyArray_SetBaseObject
from cpython cimport Py_INCREF

ctypedef unsigned int uint

cdef class ManagedArrayWrapper:
    cdef ManagedArray[uint] *thisptr

    @staticmethod
    cdef inline ManagedArrayWrapper init(ManagedArray[uint] &other):
        obj = ManagedArrayWrapper()

        # obj.thisptr = ManagedArray[uint].copyAndAcquire(other)
        obj.thisptr = new ManagedArray[uint](other.get(), other.size())

        return obj

    cdef inline uint *get(self):
        return self.thisptr.get()


cdef inline set_base(arr, obj):
    """Sets the base of arr to be this object and increases the
    reference count"""
    PyArray_SetBaseObject(arr, obj)
    Py_INCREF(obj)
