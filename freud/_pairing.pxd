# from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
from libcpp.complex cimport complex
from libcpp.vector cimport vector
cimport freud._trajectory as trajectory

cdef extern from "Pairing2D.h" namespace "freud::pairing":
    cdef cppclass Pairing2D:
        Pairing2D(const float, const unsigned int, float)
        const trajectory.Box &getBox() const
        void resetBondOrder()
        void compute(trajectory.Box &,
                     vec3[float]*,
                     float*,
                     float*,
                     unsigned int,
                     unsigned int)
        shared_array[unsigned int] getMatch()
        shared_array[unsigned int] getPair()
        unsigned int getNumParticles()
