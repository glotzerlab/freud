# Copyright (c) 2010-2023 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp.vector cimport vector

cimport freud._locality
cimport freud.util
from freud.util cimport uint, vec3


cdef extern from "Cluster.h" namespace "freud::cluster":
    cdef cppclass Cluster:
        Cluster() except +
        void compute(const freud._locality.NeighborQuery*,
                     const freud._locality.NeighborList*,
                     freud._locality.QueryArgs,
                     const unsigned int*) except +
        unsigned int getNumClusters() const
        const freud.util.ManagedArray[unsigned int] &getClusterIdx() const
        const vector[vector[uint]] getClusterKeys() const

cdef extern from "ClusterProperties.h" namespace "freud::cluster":
    cdef cppclass ClusterProperties:
        ClusterProperties()
        void compute(const freud._locality.NeighborQuery*,
                     const unsigned int*, const float*) except +
        const freud.util.ManagedArray[vec3[float]] &getClusterCenters() const
        const freud.util.ManagedArray[vec3[float]] &getClusterCentersOfMass() const
        const freud.util.ManagedArray[float] &getClusterMomentsOfInertia() const
        const freud.util.ManagedArray[float] &getClusterGyrations() const
        const freud.util.ManagedArray[unsigned int] &getClusterSizes() const
        const freud.util.ManagedArray[float] &getClusterMasses() const
