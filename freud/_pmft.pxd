# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util cimport vec3, quat
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
cimport freud._box
cimport freud._locality
cimport freud.util

cdef extern from "PMFT.h" namespace "freud::pmft":
    cdef cppclass PMFT:
        PMFT() except +

        const freud._box.Box & getBox() const
        void reset()
        const freud.util.ManagedArray[unsigned int] &getBinCounts()
        const freud.util.ManagedArray[float] &getPCF()
        float getRMax()
        vector[vector[float]] getBins() const
        vector[vector[float]] getBinCenters() const

cdef extern from "PMFTR12.h" namespace "freud::pmft":
    cdef cppclass PMFTR12(PMFT):
        PMFTR12(float, unsigned int, unsigned int, unsigned int) except +

        void accumulate(const freud._locality.NeighborQuery*,
                        float*,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getR()
        const freud.util.ManagedArray[float] &getT1()
        const freud.util.ManagedArray[float] &getT2()
        const freud.util.ManagedArray[float] &getInverseJacobian()
        unsigned int getNBinsR()
        unsigned int getNBinsT1()
        unsigned int getNBinsT2()

cdef extern from "PMFTXYT.h" namespace "freud::pmft":
    cdef cppclass PMFTXYT(PMFT):
        PMFTXYT(float, float,
                unsigned int, unsigned int, unsigned int) except +

        void accumulate(const freud._locality.NeighborQuery*,
                        float*,
                        vec3[float]*,
                        float*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getX()
        const freud.util.ManagedArray[float] &getY()
        const freud.util.ManagedArray[float] &getT()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()
        unsigned int getNBinsT()

cdef extern from "PMFTXY2D.h" namespace "freud::pmft":
    cdef cppclass PMFTXY2D(PMFT):
        PMFTXY2D(float, float, unsigned int, unsigned int) except +

        void accumulate(const freud._locality.NeighborQuery*,
                        float*,
                        vec3[float]*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getX()
        const freud.util.ManagedArray[float] &getY()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()

cdef extern from "PMFTXYZ.h" namespace "freud::pmft":
    cdef cppclass PMFTXYZ(PMFT):
        PMFTXYZ(float, float, float, unsigned int, unsigned int,
                unsigned int, vec3[float]) except +

        void accumulate(const freud._locality.NeighborQuery*,
                        quat[float]*,
                        vec3[float]*,
                        unsigned int,
                        quat[float]*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getX()
        const freud.util.ManagedArray[float] &getY()
        const freud.util.ManagedArray[float] &getZ()
        float getJacobian()
        unsigned int getNBinsX()
        unsigned int getNBinsY()
        unsigned int getNBinsZ()
