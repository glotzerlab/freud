# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util cimport vec3, uint
from libcpp.vector cimport vector
cimport freud._box
cimport freud._locality
cimport freud.util

cdef extern from "Cluster.h" namespace "freud::cluster":
    cdef cppclass Cluster:
        Cluster() except +
        void compute(const freud._locality.NeighborQuery*,
                     const freud._locality.NeighborList*,
                     freud._locality.QueryArgs,
                     const unsigned int*) except +
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
            unsigned int) except +
        unsigned int getNumClusters()
        const freud.util.ManagedArray[vec3[float]] &getClusterCOM()
        const freud.util.ManagedArray[float] &getClusterG()
        const freud.util.ManagedArray[unsigned int] &getClusterSize()
