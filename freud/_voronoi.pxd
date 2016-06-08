from freud.util._Boost cimport shared_ptr
from freud.util._cudaTypes cimport float3
cimport freud._box as box
from libcpp.vector cimport vector

cdef extern from "VoronoiBuffer.h" namespace "freud::voronoi":
    cdef cppclass VoronoiBuffer:
        VoronoiBuffer(const box.Box&)
        const box.Box &getBox() const
        void compute(const float3*, const unsigned int, const float)
        shared_ptr[vector[float3]] getBufferParticles()
