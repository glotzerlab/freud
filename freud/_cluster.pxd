# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util cimport vec3
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
cimport freud._box
cimport freud._locality
cimport freud.util

ctypedef unsigned int uint

cdef extern from "Cluster.h" namespace "freud::cluster":
    cdef cppclass Cluster:
        Cluster(float) except +
        void compute(const freud._locality.NeighborQuery*,
                     const freud._locality.NeighborList*,
                     const freud.util.ManagedArray[vec3[float]],
                     freud._locality.QueryArgs) nogil except +
        void computeClusterMembership(const unsigned int*) nogil except +
        unsigned int getNumClusters()
        unsigned int getNumParticles()
        const freud.util.ManagedArray[unsigned int] &getClusterIdx()
        const vector[vector[uint]] getClusterKeys()

cdef extern from "ClusterProperties.h" namespace "freud::cluster":
    cdef cppclass ClusterProperties:
        ClusterProperties()
        void computeProperties(
            const freud._box.Box &,
            const vec3[float]*,
            const unsigned int*,
            unsigned int) nogil except +
        unsigned int getNumClusters()
        const freud.util.ManagedArray[vec3[float]] getClusterCOM()
        const freud.util.ManagedArray[float] getClusterG()
        const freud.util.ManagedArray[unsigned int] getClusterSize()
