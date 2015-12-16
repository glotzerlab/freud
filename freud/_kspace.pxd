
from freud.util._Boost cimport shared_array
from freud.util._VectorMath cimport vec3, quat
from libcpp.complex cimport complex

cdef extern from "kspace.h" namespace "freud::kspace":
    cdef cppclass FTdelta:
        FTDelta()
        void set_K(vec3[float]*, unsigned int)
        void set_rq(unsigned int, vec3[float]*, quat[float]*)
        void set_density(float complex)
        void compute()
        shared_array[float complex] getFT()

    cdef cppclass FTsphere:
        FTsphere()
        void set_K(vec3[float]*, unsigned int)
        void set_rq(unsigned int, vec3[float]*, quat[float]*)
        void set_density(float complex)
        void compute()
        shared_array[float complex] getFT()
        void set_radius(const float)
