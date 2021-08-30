# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp.complex cimport complex
from libcpp.vector cimport vector

cimport freud._locality
cimport freud.util
from freud.util cimport vec3


cdef extern from "StaticStructureFactorDebye.h" namespace "freud::diffraction":
    cdef cppclass StaticStructureFactorDebye:
        StaticStructureFactorDebye(unsigned int, float, float) except +
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*,
                        unsigned int, unsigned int) except +
        void reset()
        const freud.util.ManagedArray[float] &getStructureFactor()
        const vector[float] getBinEdges() const
        const vector[float] getBinCenters() const
        float getMinValidK() const

cdef extern from "StaticStructureFactorDirect.h" namespace "freud::diffraction":
    cdef cppclass StaticStructureFactorDirect:
        StaticStructureFactorDirect(unsigned int, float, float, unsigned int) except +
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*, unsigned int, unsigned int) except +
        void reset()
        const freud.util.ManagedArray[float] &getStructureFactor()
        const vector[float] getBinEdges() const
        const vector[float] getBinCenters() const
        unsigned int getMaxKPoints() const
        float getMinValidK() const
        vector[vec3[float]] getKPoints() const

    vector[vec3[float]] reciprocal_isotropic(const freud._box.Box &, float, float, unsigned int)
