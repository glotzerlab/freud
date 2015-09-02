# from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._Boost cimport shared_array
cimport freud._trajectory as trajectory

cdef extern from "RDF.h" namespace "freud::density":
    cdef cppclass RDF:
        RDF(float, float)
        # ~RDF()

        const trajectory.Box& getBox() const
        void resetRDF()
        void accumulate(trajectory.Box&,
                        const vec3[float]*,
                        unsigned int,
                        const vec3[float]*,
                        unsigned int) nogil
        void compute(trajectory.Box&,
                     const vec3[float]*,
                     unsigned int,
                     const vec3[float]*,
                     unsigned int)
        void reduceRDF()
        shared_array[float] getRDF()
        shared_array[float] getR()
        shared_array[float] getNr()
        unsigned int getNBins()
