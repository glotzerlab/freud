# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libcpp.memory cimport shared_ptr
cimport freud._box
cimport freud._locality

cdef extern from "PMFT.cc" namespace "freud::pmft":
    pass

cdef extern from "PMFT.h" namespace "freud::pmft":
    cdef cppclass PMFT:
        PMFT()

        const freud._box.Box & getBox() const
        void reset()
        void reducePCF()
        shared_ptr[unsigned int] getBinCounts()
        shared_ptr[float] getPCF()
        float getRCut()

cdef extern from "PMFTR12.cc" namespace "freud::pmft":
    pass

cdef extern from "PMFTR12.h" namespace "freud::pmft":
    cdef cppclass PMFTR12(PMFT):
        PMFTR12(float, unsigned int, unsigned int, unsigned int)

        void accumulate(freud._box.Box &,
                        const freud._locality.NeighborList*,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        vec3[float]*,
                        float*,
                        unsigned int) nogil
        shared_ptr[float] getR()
        shared_ptr[float] getT1()
        shared_ptr[float] getT2()
        shared_ptr[float] getInverseJacobian()
        unsigned int getNBinsR()
        unsigned int getNBinsT1()
        unsigned int getNBinsT2()

cdef extern from "PMFTXYT.cc" namespace "freud::pmft":
    pass

cdef extern from "PMFTXYT.h" namespace "freud::pmft":
    cdef cppclass PMFTXYT(PMFT):
        PMFTXYT(float, float, unsigned int, unsigned int, unsigned int)

        void accumulate(freud._box.Box &,
                        const freud._locality.NeighborList*,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        vec3[float]*,
                        float*,
                        unsigned int) nogil
        shared_ptr[float] getX()
        shared_ptr[float] getY()
        shared_ptr[float] getT()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()
        unsigned int getNBinsT()

cdef extern from "PMFTXY2D.cc" namespace "freud::pmft":
    pass

cdef extern from "PMFTXY2D.h" namespace "freud::pmft":
    cdef cppclass PMFTXY2D(PMFT):
        PMFTXY2D(float, unsigned int, unsigned int, unsigned int)

        void accumulate(freud._box.Box &,
                        const freud._locality.NeighborList*,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        vec3[float]*,
                        float*,
                        unsigned int) nogil
        shared_ptr[float] getX()
        shared_ptr[float] getY()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()

cdef extern from "PMFTXYZ.cc" namespace "freud::pmft":
    pass

cdef extern from "PMFTXYZ.h" namespace "freud::pmft":
    cdef cppclass PMFTXYZ(PMFT):
        PMFTXYZ(float, float, float, unsigned int, unsigned int,
                unsigned int, vec3[float])

        void accumulate(freud._box.Box &,
                        const freud._locality.NeighborList*,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        quat[float]*,
                        unsigned int) nogil
        shared_ptr[float] getX()
        shared_ptr[float] getY()
        shared_ptr[float] getZ()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()
        unsigned int getNBinsZ()
