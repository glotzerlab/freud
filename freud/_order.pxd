# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util cimport vec3, quat
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
cimport freud._box
cimport freud._locality
cimport freud.util

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
        unsigned int getNumParticles()
        float getCubaticOrderParameter()
        const freud.util.ManagedArray[float] &getParticleOrderParameter()
        const freud.util.ManagedArray[float] &getGlobalTensor()
        const freud.util.ManagedArray[float] &getCubaticTensor()
        float getTInitial()
        float getTFinal()
        float getScale()
        quat[float] getCubaticOrientation()


cdef extern from "Nematic.h" namespace "freud::order":
    cdef cppclass Nematic:
        Nematic(vec3[float])
        void reset()
        void compute(quat[float]*,
                     unsigned int) except +
        unsigned int getNumParticles()
        float getNematicOrderParameter()
        const freud.util.ManagedArray[float] &getParticleTensor()
        const freud.util.ManagedArray[float] &getNematicTensor()
        vec3[float] getNematicDirector()
        vec3[float] getU()


cdef extern from "HexaticTranslational.h" namespace "freud::order":
    cdef cppclass Hexatic:
        Hexatic(unsigned int)
        const freud._box.Box & getBox() const
        void compute(const freud._locality.NeighborList*,
                     const freud._locality.NeighborQuery*,
                     freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float complex] &getOrder()
        unsigned int getK()

    cdef cppclass Translational:
        Translational(float)
        const freud._box.Box & getBox() const,
        void compute(const freud._locality.NeighborList*,
                     const freud._locality.NeighborQuery*,
                     freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float complex] &getOrder()
        float getK()


cdef extern from "Steinhardt.h" namespace "freud::order":
    cdef cppclass Steinhardt:
        Steinhardt(unsigned int, bool, bool, bool) except +
        unsigned int getNP()
        void compute(const freud._locality.NeighborList*,
                     const freud._locality.NeighborQuery*,
                     freud._locality.QueryArgs) except +
        shared_ptr[float] getQl()
        shared_ptr[float] getOrder()
        float getNorm()
        bool isAverage()
        bool isWl()
        bool isWeighted()


cdef extern from "SolLiq.h" namespace "freud::order":
    cdef cppclass SolLiq:
        SolLiq(const freud._box.Box &, float,
               float, unsigned int, unsigned int) except +
        const freud._box.Box & getBox() const
        void setBox(const freud._box.Box)
        void setClusteringRadius(float) except +
        void compute(const freud._locality.NeighborList *,
                     const vec3[float]*,
                     unsigned int) except +
        void computeSolLiqVariant(const freud._locality.NeighborList *,
                                  const vec3[float]*,
                                  unsigned int) except +
        void computeSolLiqNoNorm(const freud._locality.NeighborList *,
                                 const vec3[float]*,
                                 unsigned int) except +
        unsigned int getLargestClusterSize()
        vector[unsigned int] getClusterSizes()
        shared_ptr[float complex] getQlmi()
        shared_ptr[unsigned int] getClusters()
        shared_ptr[unsigned int] getNumberOfConnections()
        vector[float complex] getQldot_ij()
        unsigned int getNP()
        unsigned int getNumClusters()


cdef extern from "RotationalAutocorrelation.h" namespace "freud::order":
    cdef cppclass RotationalAutocorrelation:
        RotationalAutocorrelation()
        RotationalAutocorrelation(unsigned int)
        unsigned int getL()
        unsigned int getN()
        const freud.util.ManagedArray[float complex] &getRAArray()
        float getRotationalAutocorrelation()
        void compute(quat[float]*, quat[float]*, unsigned int) except +
