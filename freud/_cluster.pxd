from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._Boost cimport shared_array
cimport freud._box as box
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t

cdef extern from "Cluster.h" namespace "freud::cluster":
    cdef cppclass Cluster:
        Cluster(const box.Box&, float)
        const box.Box &getBox() const
        void computeClusters(const vec3[float]*, unsigned int) nogil
        void computeClusterMembership(const unsigned int*) nogil
        unsigned int getNumClusters()
        unsigned int getNumParticles()
        shared_array[unsigned int] getClusterIdx()
        const vector[unsigned int] getClusterKeys()

cdef extern from "ClusterProperties.h" namespace "freud::cluster":
    cdef cppclass ClusterProperties:
        ClusterProperties(const box.Box&)
        const box.Box &getBox() const
        void computeProperties(const vec3[float]*, const unsigned int*, unsigned int) nogil
        unsigned int getNumClusters()
        shared_array[vec3[float]] getClusterCOM()
        shared_array[float] getClusterG()
        shared_array[unsigned int] getClusterSize()
