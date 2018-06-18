# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.unordered_set cimport unordered_set
cimport freud._box as box
cimport freud._locality

cdef extern from "SymmetryCollection.h" namespace "freud::symmetry":
    cdef cppclass SymmetryCollection:
        SymmetryCollection(unsigned int)
        void compute(box.Box & ,
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
        int searchSymmetry(bool)
        void symmetrize(bool)
        quat[float] getHighestSymmetryQuat()

cdef extern from "Geodesation.h" namespace "freud::symmetry":
    cdef cppclass Geodesation:
        Geodesation(unsigned int)
        unsigned int getNVertices()
        shared_ptr[vector[vec3[float]]] getVertexList()
        shared_ptr[vector[unordered_set[int]]] getNeighborList()
