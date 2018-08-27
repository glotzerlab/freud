# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp.memory cimport shared_ptr
from freud.util._VectorMath cimport vec3
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t
cimport freud._box

ctypedef unsigned int uint

cdef extern from "ParticleBuffer.cc" namespace "freud::util":
    pass

cdef extern from "ParticleBuffer.h" namespace "freud::util":
    cdef cppclass ParticleBuffer:
        ParticleBuffer(const freud._box.Box &)
        const freud._box.Box & getBox() const
        void compute(
            const vec3[float]*,
            const unsigned int,
            const float,
            const bool_t) nogil except +
        shared_ptr[vector[vec3[float]]] getBufferParticles()
        shared_ptr[vector[uint]] getBufferIds()
