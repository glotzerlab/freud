# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._Boost cimport shared_array
cimport freud._box as box
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
cimport freud._locality

cdef extern from "Cluster.h" namespace "freud::cluster":
    cdef cppclass Cluster:
        Cluster(const box.Box&, float)
        const box.Box &getBox() const
        void computeClusters(const freud._locality.NeighborList*, const vec3[float]*, unsigned int) nogil except +
        void computeClusterMembership(const unsigned int*) nogil except +
        unsigned int getNumClusters()
        unsigned int getNumParticles()
        shared_array[unsigned int] getClusterIdx()
        const vector[vector[uint]] getClusterKeys()

cdef extern from "ClusterProperties.h" namespace "freud::cluster":
    cdef cppclass ClusterProperties:
        ClusterProperties(const box.Box&)
        const box.Box &getBox() const
        void computeProperties(const vec3[float]*, const unsigned int*, unsigned int) nogil except +
        unsigned int getNumClusters()
        shared_array[vec3[float]] getClusterCOM()
        shared_array[float] getClusterG()
        shared_array[unsigned int] getClusterSize()
