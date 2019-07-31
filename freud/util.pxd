# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

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

cdef extern from "NumericalArray.h" namespace "freud::util":
    cdef cppclass NumericalArray[T]:
        NumericalArray()
        NumericalArray(T*, unsigned int)
        T *get()
