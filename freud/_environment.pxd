# Copyright (c) 2010-2023 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from libcpp.complex cimport complex
from libcpp.map cimport map
from libcpp.vector cimport vector

cimport freud._box
cimport freud._locality
cimport freud.util
from freud._locality cimport BondHistogramCompute
from freud.util cimport quat, vec3


cdef extern from "BondOrder.h" namespace "freud::environment":
    ctypedef enum BondOrderMode:
        bod
        lbod
        obcd
        oocd

    cdef cppclass BondOrder(BondHistogramCompute):
        BondOrder(unsigned int, unsigned int, BondOrderMode) except +
        void accumulate(
            const freud._locality.NeighborQuery*,
            quat[float]*,
            vec3[float]*,
            quat[float]*,
            unsigned int,
            const freud._locality.NeighborList*,
            freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getBondOrder()
        BondOrderMode getMode() const

cdef extern from "LocalDescriptors.h" namespace "freud::environment":
    ctypedef enum LocalDescriptorOrientation:
        LocalNeighborhood
        Global
        ParticleLocal

    cdef cppclass LocalDescriptors:
        LocalDescriptors(unsigned int,
                         bool, LocalDescriptorOrientation)
        unsigned int getNSphs() const
        unsigned int getLMax() const
        unsigned int getSphWidth() const
        void compute(
            const freud._locality.NeighborQuery*,
            const vec3[float]*, unsigned int,
            const quat[float]*,
            const freud._locality.NeighborList*,
            freud._locality.QueryArgs,
            unsigned int) except +
        const freud.util.ManagedArray[float complex] &getSph() const
        freud._locality.NeighborList * getNList()
        LocalDescriptorOrientation getMode() const
        bool getNegativeM() const

cdef extern from "MatchEnv.h" namespace "freud::environment":
    map[unsigned int, unsigned int] minimizeRMSD(
        const freud._box.Box &, const vec3[float]*, vec3[float]*, unsigned int,
        float &, bool) except +

    map[unsigned int, unsigned int] isSimilar(const freud._box.Box &,
                                              const vec3[float]*,
                                              vec3[float]*,
                                              unsigned int,
                                              float,
                                              bool) except +

    cdef cppclass MatchEnv:
        MatchEnv() except +
        const freud.util.ManagedArray[vec3[float]] &getPointEnvironments()

    cdef cppclass EnvironmentMotifMatch(MatchEnv):
        EnvironmentMotifMatch() except +
        void compute(const freud._locality.NeighborQuery*,
                     const freud._locality.NeighborList*,
                     freud._locality.QueryArgs,
                     const vec3[float]*,
                     unsigned int,
                     float,
                     bool) except +
        const freud.util.ManagedArray[bool] &getMatches()

    cdef cppclass EnvironmentRMSDMinimizer(MatchEnv):
        EnvironmentRMSDMinimizer() except +
        void compute(
            const freud._locality.NeighborQuery*,
            const freud._locality.NeighborList*,
            freud._locality.QueryArgs,
            const vec3[float]*,
            unsigned int,
            bool) except +
        const freud.util.ManagedArray[float] &getRMSDs()

    cdef cppclass EnvironmentCluster(MatchEnv):
        EnvironmentCluster() except +
        void compute(const freud._locality.NeighborQuery*,
                     const freud._locality.NeighborList*,
                     freud._locality.QueryArgs,
                     const freud._locality.NeighborList*,
                     freud._locality.QueryArgs,
                     float,
                     bool,
                     bool) except +
        unsigned int getNumClusters()
        const freud.util.ManagedArray[unsigned int] &getClusters()
        vector[vector[vec3[float]]] &getClusterEnvironments()

cdef extern from "AngularSeparation.h" namespace "freud::environment":
    cdef cppclass AngularSeparationGlobal:
        AngularSeparationGlobal()
        void compute(quat[float]*,
                     unsigned int,
                     quat[float]*,
                     unsigned int,
                     quat[float]*,
                     unsigned int) except +
        const freud.util.ManagedArray[float] &getAngles() const

    cdef cppclass AngularSeparationNeighbor:
        AngularSeparationNeighbor()
        void compute(
            const freud._locality.NeighborQuery*,
            const quat[float]*,
            const vec3[float] *,
            const quat[float]*, unsigned int,
            const quat[float]*, unsigned int,
            const freud._locality.NeighborList*,
            freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getAngles() const
        freud._locality.NeighborList * getNList()

cdef extern from "LocalBondProjection.h" namespace "freud::environment":
    cdef cppclass LocalBondProjection:
        LocalBondProjection()
        void compute(const freud._locality.NeighborQuery*, quat[float]*,
                     vec3[float]*, unsigned int, vec3[float]*, unsigned int,
                     quat[float]*, unsigned int, const
                     freud._locality.NeighborList*,
                     freud._locality.QueryArgs) except +

        const freud.util.ManagedArray[float] &getProjections() const
        const freud.util.ManagedArray[float] &getNormedProjections() const
        freud._locality.NeighborList * getNList()
