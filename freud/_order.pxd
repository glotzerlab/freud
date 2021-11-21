# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from libcpp.complex cimport complex
from libcpp.vector cimport vector

cimport freud._box
cimport freud._locality
cimport freud.util
from freud.util cimport quat, vec3

ctypedef float complex fcomplex

cdef extern from "Cubatic.h" namespace "freud::order":
    cdef cppclass Cubatic:
        Cubatic(float,
                float,
                float,
                unsigned int,
                unsigned int) except +
        void reset()
        void compute(quat[float]*,
                     unsigned int) except +
        unsigned int getNumParticles() const
        float getCubaticOrderParameter() const
        const freud.util.ManagedArray[float] &getParticleOrderParameter() const
        const freud.util.ManagedArray[float] &getGlobalTensor() const
        const freud.util.ManagedArray[float] &getCubaticTensor() const
        quat[float] getCubaticOrientation() const
        float getTInitial() const
        float getTFinal() const
        float getScale() const
        unsigned int getNReplicates() const
        unsigned int getSeed() const


cdef extern from "Nematic.h" namespace "freud::order":
    cdef cppclass Nematic:
        Nematic(vec3[float])
        void reset()
        void compute(quat[float]*,
                     unsigned int) except +
        unsigned int getNumParticles() const
        float getNematicOrderParameter() const
        const freud.util.ManagedArray[float] &getParticleTensor() const
        const freud.util.ManagedArray[float] &getNematicTensor() const
        vec3[float] getNematicDirector() const
        vec3[float] getU() const


cdef extern from "HexaticTranslational.h" namespace "freud::order":
    cdef cppclass Hexatic:
        Hexatic(unsigned int, bool)
        void compute(const freud._locality.NeighborList*,
                     const freud._locality.NeighborQuery*,
                     freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[fcomplex] &getOrder()
        unsigned int getK()
        bool isWeighted() const

    cdef cppclass Translational:
        Translational(float, bool)
        void compute(const freud._locality.NeighborList*,
                     const freud._locality.NeighborQuery*,
                     freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[fcomplex] &getOrder() const
        float getK() const
        bool isWeighted() const


cdef extern from "Steinhardt.h" namespace "freud::order":
    cdef cppclass Steinhardt:
        Steinhardt(vector[unsigned int], bool, bool, bool, bool) except +
        unsigned int getNP() const
        void compute(const freud._locality.NeighborList*,
                     const freud._locality.NeighborQuery*,
                     freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getQl() const
        const vector[freud.util.ManagedArray[fcomplex]] &getQlm() const
        const freud.util.ManagedArray[float] &getParticleOrder() const
        vector[float] getOrder() const
        bool isAverage() const
        bool isWl() const
        bool isWeighted() const
        bool isWlNormalized() const
        vector[unsigned int] getL() const


cdef extern from "SolidLiquid.h" namespace "freud::order":
    cdef cppclass SolidLiquid:
        SolidLiquid(unsigned int, float, unsigned int, bool) except +
        unsigned int getL() const
        float getQThreshold() const
        unsigned int getSolidThreshold() const
        bool getNormalizeQ() const
        void compute(const freud._locality.NeighborList*,
                     const freud._locality.NeighborQuery*,
                     freud._locality.QueryArgs) nogil except +
        unsigned int getLargestClusterSize() const
        vector[unsigned int] getClusterSizes() const
        const freud.util.ManagedArray[unsigned int] &getClusterIdx() const
        const freud.util.ManagedArray[unsigned int] &getNumberOfConnections() \
            const
        unsigned int getNumClusters() const
        const freud.util.ManagedArray[float] &getQlm() const
        freud._locality.NeighborList * getNList()
        const freud.util.ManagedArray[float] &getQlij() const


cdef extern from "RotationalAutocorrelation.h" namespace "freud::order":
    cdef cppclass RotationalAutocorrelation:
        RotationalAutocorrelation()
        RotationalAutocorrelation(unsigned int)
        unsigned int getL() const
        const freud.util.ManagedArray[fcomplex] &getRAArray() const
        float getRotationalAutocorrelation() const
        void compute(quat[float]*, quat[float]*, unsigned int) except +
