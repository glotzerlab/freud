# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from libcpp.memory cimport shared_ptr
cimport freud._box
cimport freud._locality

cdef extern from "InterfaceMeasure.cc" namespace "freud::interface":
    pass

cdef extern from "InterfaceMeasure.h" namespace "freud::interface":
    cdef cppclass InterfaceMeasure:
        InterfaceMeasure(const freud._box.Box &, float)
        void compute(
            const freud._locality.NeighborList*,
            const vec3[float]*,
            unsigned int,
            const vec3[float]*,
            unsigned int)
        unsigned int getInterfaceCount()
        shared_ptr[unsigned int] getInterfaceIdx()
