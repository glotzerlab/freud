# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util cimport vec3
from freud._locality cimport BondHistogramCompute
from libcpp cimport bool
from libcpp.vector cimport vector

cimport freud._box
cimport freud._locality
cimport freud.util

ctypedef unsigned int uint

cdef extern from "StructureFactor.h" namespace "freud::diffraction":
    cdef cppclass StructureFactor:
        StructureFactor(uint, float, float, bool) except +
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*,
                        unsigned int, const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getStructureFactor()
        const vector[float] getBinEdges() const
        const vector[float] getBinCenters() const
        float &getMinValidK() const
