# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
cimport freud._box
cimport freud._locality

cdef extern from "BondingAnalysis.cc" namespace "freud::bond":
    pass

cdef extern from "BondingAnalysis.h" namespace "freud::bond":
    cdef cppclass BondingAnalysis:
        BondingAnalysis(unsigned int, unsigned int)
        void reduceArrays()
        void initialize(unsigned int*) nogil
        void compute(unsigned int*, unsigned int*) nogil
        vector[vector[uint]] getBondLifetimes()
        # vector[vector[uint]] getOverallLifetimes()
        vector[uint] getOverallLifetimes()
        shared_ptr[uint] getTransitionMatrix()
        unsigned int getNumFrames()
        unsigned int getNumParticles()
        unsigned int getNumBonds()

cdef extern from "BondingR12.cc" namespace "freud::bond":
    pass

cdef extern from "BondingR12.h" namespace "freud::bond":
    cdef cppclass BondingR12:
        BondingR12(float, unsigned int, unsigned int, unsigned int,
                   unsigned int, unsigned int *, unsigned int *)
        const freud._box.Box & getBox() const
        void compute(freud._box.Box &,
                     const freud._locality.NeighborList*,
                     vec3[float]*,
                     float*,
                     unsigned int,
                     vec3[float]*,
                     float*,
                     unsigned int) nogil except +
        shared_ptr[uint] getBonds()
        unsigned int getNumParticles()
        unsigned int getNumBonds()
        map[uint, uint] getListMap()
        map[uint, uint] getRevListMap()

cdef extern from "BondingXY2D.cc" namespace "freud::bond":
    pass

cdef extern from "BondingXY2D.h" namespace "freud::bond":
    cdef cppclass BondingXY2D:
        BondingXY2D(float, float, unsigned int, unsigned int, unsigned int,
                    unsigned int *, unsigned int *)
        const freud._box.Box & getBox() const
        void compute(freud._box.Box &,
                     const freud._locality.NeighborList*,
                     vec3[float]*,
                     float*,
                     unsigned int,
                     vec3[float]*,
                     float*,
                     unsigned int) nogil except +
        shared_ptr[uint] getBonds()
        unsigned int getNumParticles()
        unsigned int getNumBonds()
        map[uint, uint] getListMap()
        map[uint, uint] getRevListMap()

cdef extern from "BondingXYT.cc" namespace "freud::bond":
    pass

cdef extern from "BondingXYT.h" namespace "freud::bond":
    cdef cppclass BondingXYT:
        BondingXYT(float, float, unsigned int, unsigned int, unsigned int,
                   unsigned int, unsigned int *, unsigned int *)
        const freud._box.Box & getBox() const
        void compute(freud._box.Box &,
                     const freud._locality.NeighborList*,
                     vec3[float]*,
                     float*,
                     unsigned int,
                     vec3[float]*,
                     float*,
                     unsigned int) nogil except +
        shared_ptr[uint] getBonds()
        unsigned int getNumParticles()
        unsigned int getNumBonds()
        map[uint, uint] getListMap()
        map[uint, uint] getRevListMap()

cdef extern from "BondingXYZ.cc" namespace "freud::bond":
    pass

cdef extern from "BondingXYZ.h" namespace "freud::bond":
    cdef cppclass BondingXYZ:
        BondingXYZ(float, float, float, unsigned int, unsigned int,
                   unsigned int, unsigned int, unsigned int *,
                   unsigned int *)
        const freud._box.Box & getBox() const
        void compute(freud._box.Box &,
                     const freud._locality.NeighborList*,
                     vec3[float]*,
                     quat[float]*,
                     unsigned int,
                     vec3[float]*,
                     quat[float]*,
                     unsigned int) nogil except +
        shared_ptr[uint] getBonds()
        unsigned int getNumParticles()
        unsigned int getNumBonds()
        map[uint, uint] getListMap()
        map[uint, uint] getRevListMap()
