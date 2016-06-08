# from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._Boost cimport shared_array
cimport freud._box as box

cdef extern from "CorrelationFunction.h" namespace "freud::density":
    cdef cppclass CorrelationFunction[T]:
        CorrelationFunction(float, float)
        const box.Box &getBox() const
        void resetCorrelationFunction()
        void accumulate(const box.Box &, const vec3[float]*, const T*,
            unsigned int, const vec3[float]*, const T*, unsigned int) nogil
        void reduceCorrelationFunction()
        shared_array[T] getRDF()
        shared_array[unsigned int] getCounts()
        shared_array[float] getR()
        unsigned int getNBins() const

cdef extern from "GaussianDensity.h" namespace "freud::density":
    cdef cppclass GaussianDensity:
        GaussianDensity(unsigned int, float, float)
        GaussianDensity(unsigned int, unsigned int, unsigned int, float, float)
        const box.Box &getBox() const
        void resetDensity()
        void reduceDensity()
        void compute(const box.Box &, const vec3[float]*, unsigned int) nogil
        shared_array[float] getDensity()
        unsigned int getWidthX()
        unsigned int getWidthY()
        unsigned int getWidthZ()

cdef extern from "LocalDensity.h" namespace "freud::density":
    cdef cppclass LocalDensity:
        LocalDensity(float, float, float)
        const box.Box &getBox() const
        void compute(const box.Box &, const vec3[float]*, unsigned int, const vec3[float]*, unsigned int) nogil
        unsigned int getNRef()
        shared_array[float] getDensity()
        shared_array[float] getNumNeighbors()

cdef extern from "RDF.h" namespace "freud::density":
    cdef cppclass RDF:
        RDF(float, float)
        const box.Box& getBox() const
        void resetRDF()
        void accumulate(box.Box&,
                        const vec3[float]*,
                        unsigned int,
                        const vec3[float]*,
                        unsigned int) nogil
        void reduceRDF()
        shared_array[float] getRDF()
        shared_array[float] getR()
        shared_array[float] getNr()
        unsigned int getNBins()
