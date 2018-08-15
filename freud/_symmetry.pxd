# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp.memory cimport shared_ptr
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string
cimport freud._box
cimport freud._locality

cdef extern from "SymmetryCollection.cc" namespace "freud::symmetry":
    pass

cdef extern from "SymmetryCollection.h" namespace "freud::symmetry":
    struct FoundSymmetry:
        int n
        vec3[float] v
        quat[float] q
        float measured_order

    cdef cppclass SymmetryCollection:
        SymmetryCollection(unsigned int)
        void compute(freud._box.Box &,
                     vec3[float]*,
                     const freud._locality.NeighborList*,
                     unsigned int) nogil except +
        shared_ptr[quat[float]] getOrderQuats()
        shared_ptr[float] getMlm()
        shared_ptr[float] getMlm_rotated()
        float measure(int)
        int getNP()
        int getMaxL()
        void rotate(const quat[float]&)
        vector[FoundSymmetry] getSymmetries()
        string getLaueGroup()
        string getCrystalSystem()

cdef extern from "Geodesation.cc" namespace "freud::symmetry":
    pass

cdef extern from "Geodesation.h" namespace "freud::symmetry":
    cdef cppclass Geodesation:
        Geodesation(unsigned int)
        unsigned int getNVertices()
        shared_ptr[vector[vec3[float]]] getVertexList()
        shared_ptr[vector[unordered_set[int]]] getNeighborList()
