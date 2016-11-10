# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from freud.util._Boost cimport shared_ptr
from freud.util._cudaTypes cimport float3
cimport freud._box as box
from libcpp.vector cimport vector

cdef extern from "VoronoiBuffer.h" namespace "freud::voronoi":
    cdef cppclass VoronoiBuffer:
        VoronoiBuffer(const box.Box&)
        const box.Box &getBox() const
        void compute(const float3*, const unsigned int, const float) nogil except +
        shared_ptr[vector[float3]] getBufferParticles()
