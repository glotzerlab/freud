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
cimport freud._box as box
cimport freud._locality

cdef extern from "SymmetryCollection.h" namespace "freud::symmetry":
    cdef cppclass SymmetryCollection:
        SymmetryCollection()
        void compute(box.Box & ,
                     vec3[float]*,
                     const freud._locality.NeighborList*,
                     unsigned int) nogil except +
        quat[float] getHighestOrderQuat()
        shared_ptr[quat[float]] getOrderQuats()
        shared_ptr[float complex] getMlm()
        int getNP()
