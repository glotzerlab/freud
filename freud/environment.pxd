# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
from freud.errors import FreudDeprecationWarning

cimport freud._environment
cimport freud.locality
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

cdef class AngularSeparation:
    cdef freud._environment.AngularSeparation * thisptr
    cdef unsigned int num_neigh
    cdef float rmax
    cdef freud.locality.NeighborList nlist_

cdef class LocalBondProjection:
    cdef freud._environment.LocalBondProjection * thisptr
    cdef float rmax
    cdef unsigned int num_neigh
    cdef freud.locality.NeighborList nlist_
