# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
from freud.errors import FreudDeprecationWarning

cimport freud._environment
cimport numpy as np

cdef class BondOrder:
    cdef freud._environment.BondOrder * thisptr
    cdef num_neigh
    cdef rmax

cdef class LocalDescriptors:
    cdef freud._environment.LocalDescriptors * thisptr
    cdef num_neigh
    cdef rmax

cdef class MatchEnv:
    cdef freud._environment.MatchEnv * thisptr
    cdef rmax
    cdef num_neigh
    cdef m_box

cdef class Pairing2D:
    cdef freud._environment.Pairing2D * thisptr
    cdef rmax
    cdef num_neigh

cdef class AngularSeparation:
    cdef freud._environment.AngularSeparation * thisptr
    cdef num_neigh
    cdef rmax
    cdef nlist_
