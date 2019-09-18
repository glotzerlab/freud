# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util cimport vec3
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
cimport freud._box
cimport freud._locality
cimport freud.util

ctypedef unsigned int uint

cdef extern from "CorrelationFunction.h" namespace "freud::density":
    cdef cppclass CorrelationFunction[T]:
        CorrelationFunction(float, float) except +
        const freud._box.Box & getBox() const
        void reset()
        void accumulate(const freud._locality.NeighborQuery*, const T*,
                        const vec3[float]*,
                        const T*,
                        unsigned int, const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[T] &getRDF()
        const freud.util.ManagedArray[unsigned int] &getCounts()
        const freud.util.ManagedArray[float] &getR()
        unsigned int getNBins() const

cdef extern from "GaussianDensity.h" namespace "freud::density":
    cdef cppclass GaussianDensity:
        GaussianDensity(vec3[unsigned int], float, float) except +
        const freud._box.Box & getBox() const
        void reset()
        void compute(
            const freud._box.Box &,
            const vec3[float]*,
            unsigned int) except +
        const freud.util.ManagedArray[float] &getDensity()
        vec3[unsigned int] getWidth()
        float getSigma()

cdef extern from "LocalDensity.h" namespace "freud::density":
    cdef cppclass LocalDensity:
        LocalDensity(float, float, float)
        const freud._box.Box & getBox() const
        void compute(
            const freud._locality.NeighborQuery*,
            const vec3[float]*,
            unsigned int, const freud._locality.NeighborList *,
            freud._locality.QueryArgs) except +
        unsigned int getNPoints()
        const freud.util.ManagedArray[float] &getDensity()
        const freud.util.ManagedArray[float] &getNumNeighbors()

cdef extern from "RDF.h" namespace "freud::density":
    cdef cppclass RDF:
        RDF(float, float, float) except +
        const freud._box.Box & getBox() const
        float getRMax() const
        float getRMin() const
        float getDr() const
        void reset()
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getRDF()
        vector[float] getBinCenters()
        const freud.util.ManagedArray[float] &getNr()
        const freud.util.ManagedArray[uint] &getBinCounts()
        vector[float] getBins()
