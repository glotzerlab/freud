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

cdef extern from "BondOrder.h" namespace "freud::environment":
    cdef cppclass BondOrder:
        BondOrder(unsigned int, unsigned int) except +
        const freud._box.Box & getBox() const
        void reset()
        void accumulate(
            const freud._locality.NeighborQuery*,
            quat[float]*,
            vec3[float]*,
            quat[float]*,
            unsigned int,
            unsigned int,
            const freud._locality.NeighborList*,
            freud._locality.QueryArgs)
        const freud.util.ManagedArray[float] &getBondOrder()
        const freud.util.ManagedArray[float] &getTheta()
        const freud.util.ManagedArray[float] &getPhi()
        unsigned int getNBinsTheta()
        unsigned int getNBinsPhi()

cdef extern from "LocalDescriptors.h" namespace "freud::environment":
    ctypedef enum LocalDescriptorOrientation:
        LocalNeighborhood
        Global
        ParticleLocal

    cdef cppclass LocalDescriptors:
        LocalDescriptors(unsigned int,
                         bool)
        unsigned int getNSphs() const
        unsigned int getLMax() const
        unsigned int getSphWidth() const
        unsigned int getNPoints()
        void compute(
            const freud._box.Box &, unsigned int,
            const vec3[float]*, unsigned int,
            const vec3[float]*, unsigned int,
            const quat[float]*, LocalDescriptorOrientation,
            const freud._locality.NeighborList*) except +
        const freud.util.ManagedArray[float complex] &getSph()

cdef extern from "MatchEnv.h" namespace "freud::environment":
    cdef cppclass MatchEnv:
        MatchEnv(const freud._box.Box &, float, unsigned int) except +
        void setBox(const freud._box.Box)
        void cluster(const freud._locality.NeighborList*,
                     const freud._locality.NeighborList*,
                     const vec3[float]*,
                     unsigned int,
                     float,
                     bool,
                     bool,
                     bool) except +
        void matchMotif(const freud._locality.NeighborList*,
                        const vec3[float]*,
                        unsigned int,
                        const vec3[float]*,
                        unsigned int,
                        float,
                        bool) except +
        vector[float] minRMSDMotif(
            const freud._locality.NeighborList*,
            const vec3[float]*,
            unsigned int,
            const vec3[float]*,
            unsigned int,
            bool) except +
        map[unsigned int, unsigned int] isSimilar(const vec3[float]*,
                                                  vec3[float]*,
                                                  unsigned int,
                                                  float,
                                                  bool) except +
        map[unsigned int, unsigned int] minimizeRMSD(const vec3[float]*,
                                                     vec3[float]*,
                                                     unsigned int,
                                                     float &,
                                                     bool) except +
        shared_ptr[unsigned int] getClusters()
        shared_ptr[vec3[float]] getEnvironment(unsigned int)
        shared_ptr[vec3[float]] getTotEnvironment()
        unsigned int getNP()
        unsigned int getNumClusters()
        unsigned int getNumNeighbors()
        unsigned int getMaxNumNeighbors()

cdef extern from "AngularSeparation.h" namespace "freud::environment":
    cdef cppclass AngularSeparation:
        AngularSeparation()
        void computeNeighbor(
            quat[float]*, unsigned int,
            quat[float]*, unsigned int,
            quat[float]*, unsigned int,
            const freud._locality.NeighborList*) except +
        void computeGlobal(quat[float]*,
                           unsigned int,
                           quat[float]*,
                           unsigned int,
                           quat[float]*,
                           unsigned int) except +
        const freud.util.ManagedArray[float] &getNeighborAngles()
        const freud.util.ManagedArray[float] &getGlobalAngles()

cdef extern from "LocalBondProjection.h" namespace "freud::environment":
    cdef cppclass LocalBondProjection:
        LocalBondProjection()
        void compute(freud._box.Box &,
                     vec3[float]*, unsigned int,
                     vec3[float]*, quat[float]*, unsigned int,
                     vec3[float]*, unsigned int,
                     quat[float]*, unsigned int,
                     const freud._locality.NeighborList*) except +

        const freud.util.ManagedArray[float] &getProjections()
        const freud.util.ManagedArray[float] &getNormedProjections()
        unsigned int getNPoints()
        unsigned int getNQueryPoints()
        unsigned int getNproj()
        const freud._box.Box & getBox() const
