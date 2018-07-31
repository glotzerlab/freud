# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import time
import warnings
from .errors import FreudDeprecationWarning
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
cimport freud._box as _box
cimport freud._environment as environment
cimport numpy as np
cimport freud._locality as locality

cdef class BondOrder:
    cdef environment.BondOrder * thisptr
    cdef num_neigh
    cdef rmax

cdef class LocalDescriptors:
    cdef environment.LocalDescriptors * thisptr
    cdef num_neigh
    cdef rmax

cdef class MatchEnv:
    cdef environment.MatchEnv * thisptr
    cdef rmax
    cdef num_neigh
    cdef m_box

cdef class Pairing2D:
    cdef environment.Pairing2D * thisptr
    cdef rmax
    cdef num_neigh

cdef class AngularSeparation:
    cdef environment.AngularSeparation * thisptr
    cdef num_neigh
    cdef rmax
    cdef nlist_
