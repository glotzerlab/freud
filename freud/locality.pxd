# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool as cbool

cimport freud._locality
cimport freud.box

cdef class NeighborQuery:
    cdef freud._locality.NeighborQuery * spdptr
    cdef cbool queryable
    cdef freud.box.Box box
    cdef float[:, ::1] ref_points

cdef class NeighborList:
    cdef freud._locality.NeighborList * thisptr
    cdef char _managed
    cdef base

    cdef refer_to(self, freud._locality.NeighborList * other)
    cdef freud._locality.NeighborList * get_ptr(self) nogil
    cdef void copy_c(self, NeighborList other)

cdef class IteratorLinkCell:
    cdef freud._locality.IteratorLinkCell * thisptr

    cdef void copy(self, const freud._locality.IteratorLinkCell & rhs)

cdef class LinkCell(NeighborQuery):
    cdef freud._locality.LinkCell * thisptr
    cdef NeighborList _nlist

cdef class NearestNeighbors:
    cdef freud._locality.NearestNeighbors * thisptr
    cdef NeighborList _nlist
    cdef _cached_points
    cdef _cached_ref_points
    cdef _cached_box

cdef class AABBQuery(NeighborQuery):
    cdef freud._locality.AABBQuery * thisptr
    cdef NeighborList _nlist
