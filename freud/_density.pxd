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

cdef extern from "GaussianDensity.h" namespace "freud::density":
    cdef cppclass GaussianDensity:
        GaussianDensity(unsigned int, float, float)
        GaussianDensity(unsigned int, unsigned int, unsigned int, float, float)
        const trajectory.Box &getBox() const

        void resetDensity()
        void reduceDensity()

        void accumulate(const trajectory.Box &, const vec3[float]*) nogil

        shared_array[float] getDensity()

cdef extern from "CorrelationFunction.h" namespace "freud::density":
    cdef cppclass CorrelationFunction[T]:
        CorrelationFunction(float, float)
        const trajectory.Box &getBox() const
        void resetCorrelationFunction()

        void accumulate(const trajectory.Box &, const vec3[float]*, const T*,
            unsigned int, const vec3[float]*, const T*, unsigned int) nogil

        void reduceCorrelationFunction()
        shared_array[T] getRDF()
        shared_array[unsigned int] getCounts()
        shared_array[float] getR()
        unsigned int getNBins() const
