# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

cimport numpy
from libcpp cimport bool

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
        ManagedArray(T*, unsigned int)
        T *get()
        unsigned int size()

        @staticmethod
        ManagedArray *copyAndAcquire(ManagedArray &other)

        bool m_managed

cdef extern from "numpy/arrayobject.h":
    cdef int PyArray_SetBaseObject(numpy.ndarray arr, obj)
