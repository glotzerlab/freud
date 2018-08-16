# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp.memory cimport shared_ptr
from freud.util._VectorMath cimport vec3
from libcpp.vector cimport vector
cimport freud._box
cimport freud._locality

ctypedef unsigned int uint

cdef extern from "InterfaceMeasure.cc" namespace "freud::interface":
    pass

cdef extern from "InterfaceMeasure.h" namespace "freud::interface":
    cdef cppclass InterfaceMeasure:
        InterfaceMeasure(const freud._box.Box &, float)
        void compute(
            const freud._locality.NeighborList*,
            const vec3[float]*,
            const unsigned int,
            const vec3[float]*,
            const unsigned int) nogil except +
        unsigned int getInterfaceCount()
        shared_ptr[vector[uint]] getInterfaceIds()
