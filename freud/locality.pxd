# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr

cimport freud._locality
cimport freud.box

cdef class NeighborQueryResult:
    cdef freud._locality.NeighborQuery * spdptr
    cdef shared_ptr[freud._locality.NeighborQueryIterator] iterator
    cdef float[:, ::1] points
    cdef float r
    cdef unsigned int k
    cdef unsigned int Np
    cdef cbool exclude_ii
    cdef str query_type

    @staticmethod
    cdef inline NeighborQueryResult init(freud._locality.NeighborQuery * spdptr, float[:, ::1] points, cbool exclude_ii, float r=0, unsigned int k=0):
        # Internal API only
        assert r != 0 or k != 0

        obj = NeighborQueryResult()

        obj.spdptr = spdptr
        obj.points = points
        obj.exclude_ii = exclude_ii
        obj.Np = points.shape[0]

        obj.r = r
        obj.k = k

        if obj.r != 0:
            obj.query_type = 'ball'
        else:
            obj.query_type = 'nn'

        return obj


cdef class AABBQueryResult(NeighborQueryResult):
    cdef freud._locality.AABBQuery * aabbptr
    cdef float r_guess
    cdef float scale

    @staticmethod
    cdef inline AABBQueryResult init2(freud._locality.AABBQuery * aabbptr, float[:, ::1] points, cbool exclude_ii, unsigned int k, float r_guess, float scale):
        # Internal API only
        assert k != 0

        obj = AABBQueryResult()

        obj.aabbptr = obj.spdptr = aabbptr
        obj.points = points
        obj.exclude_ii = exclude_ii
        obj.Np = points.shape[0]

        obj.r = 0  # Only for kN queries
        obj.k = k

        if obj.r != 0:
            obj.query_type = 'ball'
        else:
            obj.query_type = 'nn'

        obj.r_guess = r_guess
        obj.scale = scale

        return obj


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
