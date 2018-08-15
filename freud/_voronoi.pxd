# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp.memory cimport shared_ptr
from freud.util._VectorMath cimport vec3
from libcpp.vector cimport vector
cimport freud._box

ctypedef unsigned int uint

cdef extern from "VoronoiBuffer.cc" namespace "freud::voronoi":
    pass

cdef extern from "VoronoiBuffer.h" namespace "freud::voronoi":
    cdef cppclass VoronoiBuffer:
        VoronoiBuffer(const freud._box.Box &)
        const freud._box.Box & getBox() const
        void compute(
            const vec3[float]*,
            const unsigned int,
            const float) nogil except +
        shared_ptr[vector[vec3[float]]] getBufferParticles()
        shared_ptr[vector[uint]] getBufferIds()
