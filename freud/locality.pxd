# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

from . cimport _locality

cdef class NeighborList:
    cdef _locality.NeighborList * thisptr
    cdef char _managed
    cdef base

    cdef refer_to(self, _locality.NeighborList * other)
    cdef _locality.NeighborList * get_ptr(self)
    cdef void copy_c(self, NeighborList other)

cdef class IteratorLinkCell:
    cdef _locality.IteratorLinkCell * thisptr

    cdef void copy(self, const _locality.IteratorLinkCell & rhs)

cdef class LinkCell:
    cdef _locality.LinkCell * thisptr
    cdef NeighborList _nlist

cdef class NearestNeighbors:
    cdef _locality.NearestNeighbors * thisptr
    cdef NeighborList _nlist
    cdef _cached_points
    cdef _cached_ref_points
    cdef _cached_box
