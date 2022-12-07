# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from libcpp.vector cimport vector

cimport freud._locality
cimport freud._box
cimport freud.util
from freud.util cimport vec3


cdef extern from "StructureFactor.h" namespace "freud::diffraction":
    cdef cppclass StructureFactor:
        const vector[float] getBinEdges() const
        const vector[float] getBinCenters() const
        float getMinValidK() const

cdef extern from "StaticStructureFactorDebye.h" namespace "freud::diffraction":
    cdef cppclass StaticStructureFactorDebye(StructureFactor):
        StaticStructureFactorDebye(unsigned int, float, float) except +
        const freud.util.ManagedArray[float] &getStructureFactor()
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*, unsigned int, unsigned int) except +
        void reset()

cdef extern from "StructureFactorDirect.h" namespace "freud::diffraction":
    cdef cppclass StructureFactorDirect(StructureFactor):
        unsigned int getNumSampledKPoints() const
        vector[vec3[float]] getKPoints() const

cdef extern from "StaticStructureFactorDirect.h" namespace "freud::diffraction":
    cdef cppclass StaticStructureFactorDirect(StructureFactorDirect):
        StaticStructureFactorDirect(unsigned int, float, float, unsigned int) except +
        const freud.util.ManagedArray[float] &getStructureFactor()
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*, unsigned int, unsigned int) except +
        void reset()

cdef extern from "IntermediateScattering.h" namespace "freud::diffraction":
    cdef cppclass IntermediateScattering(StructureFactorDirect):
        IntermediateScattering(freud._box.Box box, unsigned int, float, float, unsigned int) except +
        void compute(const vec3[float]*, unsigned int, const vec3[float]*, unsigned int, unsigned int, unsigned int)
        const freud.util.ManagedArray[float] &getSelfFunction() const
        const freud.util.ManagedArray[float] &getDistinctFunction() const
