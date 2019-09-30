# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

cimport numpy

cdef extern from "VectorMath.h":
    cdef cppclass vec3[Real]:
        vec3(Real, Real, Real)
        vec3()
        Real x
        Real y
        Real z

    cdef cppclass quat[Real]:
        quat(Real, vec3[Real])
        quat()
        Real s
        vec3[Real] v

cdef extern from "ManagedArray.h" namespace "freud::util":
    cdef cppclass ManagedArray[T]:
        ManagedArray()
        ManagedArray(const ManagedArray[T] &)
        T *get()
        unsigned int size() const
        vector[unsigned int] shape() const


cdef extern from "numpy/arrayobject.h":
    cdef int PyArray_SetBaseObject(numpy.ndarray arr, obj)
