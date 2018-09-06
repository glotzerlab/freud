# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3
from libcpp.memory cimport shared_ptr
cimport freud._box
cimport freud._locality

cdef extern from "CorrelationFunction.h" namespace "freud::density":
    cdef cppclass CorrelationFunction[T]:
        CorrelationFunction(float, float)
        const freud._box.Box & getBox() const
        void reset()
        void accumulate(const freud._box.Box &,
                        const freud._locality.NeighborList*,
                        const vec3[float]*, const T*,
                        unsigned int,
                        const vec3[float]*,
                        const T*,
                        unsigned int) nogil except +
        void reduceCorrelationFunction()
        shared_ptr[T] getRDF()
        shared_ptr[unsigned int] getCounts()
        shared_ptr[float] getR()
        unsigned int getNBins() const

cdef extern from "GaussianDensity.h" namespace "freud::density":
    cdef cppclass GaussianDensity:
        GaussianDensity(unsigned int, float, float)
        GaussianDensity(unsigned int,
                        unsigned int,
                        unsigned int,
                        float,
                        float)
        const freud._box.Box & getBox() const
        void reset()
        void reduceDensity()
        void compute(
            const freud._box.Box &,
            const vec3[float]*,
            unsigned int) nogil except +
        shared_ptr[float] getDensity()
        unsigned int getWidthX()
        unsigned int getWidthY()
        unsigned int getWidthZ()

cdef extern from "LocalDensity.h" namespace "freud::density":
    cdef cppclass LocalDensity:
        LocalDensity(float, float, float)
        const freud._box.Box & getBox() const
        void compute(
            const freud._box.Box &,
            const freud._locality.NeighborList *,
            const vec3[float]*,
            unsigned int,
            const vec3[float]*,
            unsigned int) nogil except +
        unsigned int getNRef()
        shared_ptr[float] getDensity()
        shared_ptr[float] getNumNeighbors()

cdef extern from "RDF.h" namespace "freud::density":
    cdef cppclass RDF:
        RDF(float, float, float)
        const freud._box.Box & getBox() const
        void reset()
        void accumulate(freud._box.Box &,
                        const freud._locality.NeighborList*,
                        const vec3[float]*,
                        unsigned int,
                        const vec3[float]*,
                        unsigned int) nogil except +
        void reduceRDF()
        shared_ptr[float] getRDF()
        shared_ptr[float] getR()
        shared_ptr[float] getNr()
        unsigned int getNBins()
