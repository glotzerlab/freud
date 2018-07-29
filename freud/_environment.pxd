# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
cimport freud._box as box
cimport freud._locality

cdef extern from "BondOrder.h" namespace "freud::environment":
    cdef cppclass BondOrder:
        BondOrder(float, float, unsigned int, unsigned int, unsigned int)
        const box.Box & getBox() const
        void reset()
        void accumulate(
            box.Box &,
            const freud._locality.NeighborList*,
            vec3[float]*,
            quat[float]*,
            unsigned int,
            vec3[float]*,
            quat[float]*,
            unsigned int,
            unsigned int) nogil
        void reduceBondOrder()
        shared_ptr[float] getBondOrder()
        shared_ptr[float] getTheta()
        shared_ptr[float] getPhi()
        unsigned int getNBinsTheta()
        unsigned int getNBinsPhi()

cdef extern from "LocalDescriptors.h" namespace "freud::environment":
    ctypedef enum LocalDescriptorOrientation:
        LocalNeighborhood
        Global
        ParticleLocal

    cdef cppclass LocalDescriptors:
        LocalDescriptors(unsigned int,
                         unsigned int,
                         float,
                         bool)
        unsigned int getNSphs() const
        unsigned int getLMax() const
        unsigned int getSphWidth() const
        float getRMax() const
        unsigned int getNP()
        void computeNList(const box.Box &, const vec3[float]*, unsigned int,
                          const vec3[float]*, unsigned int) nogil except +
        void compute(
            const box.Box &, const freud._locality.NeighborList*,
            unsigned int, const vec3[float]*,
            unsigned int, const vec3[float]*, unsigned int,
            const quat[float]*, LocalDescriptorOrientation) nogil except +
        shared_ptr[float complex] getSph()

cdef extern from "MatchEnv.h" namespace "freud::environment":
    cdef cppclass MatchEnv:
        MatchEnv(const box.Box &, float, unsigned int) nogil except +
        void setBox(const box.Box)
        void cluster(const freud._locality.NeighborList*,
                     const freud._locality.NeighborList*,
                     const vec3[float]*,
                     unsigned int,
                     float,
                     bool,
                     bool,
                     bool) nogil except +
        void matchMotif(const freud._locality.NeighborList*,
                        const vec3[float]*,
                        unsigned int,
                        const vec3[float]*,
                        unsigned int,
                        float,
                        bool) nogil except +
        vector[float] minRMSDMotif(
            const freud._locality.NeighborList*,
            const vec3[float]*,
            unsigned int,
            const vec3[float]*,
            unsigned int,
            bool) nogil except +
        map[unsigned int, unsigned int] isSimilar(const vec3[float]*,
                                                  vec3[float]*,
                                                  unsigned int,
                                                  float,
                                                  bool) nogil except +
        map[unsigned int, unsigned int] minimizeRMSD(const vec3[float]*,
                                                     vec3[float]*,
                                                     unsigned int,
                                                     float &,
                                                     bool) nogil except +
        shared_ptr[unsigned int] getClusters()
        shared_ptr[vec3[float]] getEnvironment(unsigned int)
        shared_ptr[vec3[float]] getTotEnvironment()
        unsigned int getNP()
        unsigned int getNumClusters()
        unsigned int getNumNeighbors()
        unsigned int getMaxNumNeighbors()

cdef extern from "Pairing2D.h" namespace "freud::environment":
    cdef cppclass Pairing2D:
        Pairing2D(const float, const unsigned int, float)
        const box.Box & getBox() const
        void reset()
        void compute(box.Box &,
                     const freud._locality.NeighborList*,
                     vec3[float]*,
                     float*,
                     float*,
                     unsigned int,
                     unsigned int) nogil except +
        shared_ptr[unsigned int] getMatch()
        shared_ptr[unsigned int] getPair()
        unsigned int getNumParticles()

cdef extern from "AngularSeparation.h" namespace "freud::environment":
    cdef cppclass AngularSeparation:
        AngularSeparation()
        void computeNeighbor(const freud._locality.NeighborList*,
                             quat[float]*,
                             quat[float]*,
                             quat[float]*,
                             unsigned int,
                             unsigned int,
                             unsigned int) nogil except +
        void computeGlobal(quat[float]*,
                           quat[float]*,
                           quat[float]*,
                           unsigned int,
                           unsigned int,
                           unsigned int) nogil except +

        shared_ptr[float] getNeighborAngles()
        shared_ptr[float] getGlobalAngles()
        unsigned int getNP()
        unsigned int getNref()
        unsigned int getNglobal()
