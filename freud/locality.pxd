# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool as cbool
from libcpp.memory cimport shared_ptr

cimport freud._locality
cimport freud.box


cdef class NeighborQueryResult:
    cdef NeighborQuery nq
    cdef const float[:, ::1] points
    cdef float r_max
    cdef unsigned int num_neighbors
    cdef unsigned int Np
    cdef cbool exclude_ii
    cdef str query_type

    cdef shared_ptr[
        freud._locality.NeighborQueryIterator] _getIterator(self) except *

    # This had to be implemented as a factory because the constructors will
    # always get called with Python objects as arguments, and we need typed
    # objects. See the link below for more information.
    # https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#initialisation-methods-cinit-and-init
    # This needed to be declared inline because it needed to be in the pxd,
    # which in turn was the only way to get the staticmethod decorator to
    # compile with a cdef method.
    @staticmethod
    cdef inline NeighborQueryResult init(
            NeighborQuery nq, const float[:, ::1] points,
            cbool exclude_ii, float r_max=0, unsigned int num_neighbors=0):
        # Internal API only
        assert r_max != 0 or num_neighbors != 0

        obj = NeighborQueryResult()

        obj.nq = nq
        obj.points = points
        obj.exclude_ii = exclude_ii
        obj.Np = points.shape[0]

        obj.r_max = r_max
        obj.num_neighbors = num_neighbors

        if obj.r_max != 0:
            obj.query_type = 'ball'
        else:
            obj.query_type = 'nearest'

        return obj


cdef class AABBQueryResult(NeighborQueryResult):
    cdef AABBQuery aabbq
    cdef float scale

    @staticmethod
    cdef inline AABBQueryResult init_aabb_nn(
            AABBQuery aabbq, const float[:, ::1] points,
            cbool exclude_ii,
            unsigned int num_neighbors, float r_max,
            float scale):
        # Internal API only
        assert num_neighbors != 0

        obj = AABBQueryResult()
        obj.aabbq = obj.nq = aabbq
        obj.points = points
        obj.exclude_ii = exclude_ii
        obj.Np = points.shape[0]

        # For AABBs, even kN queries require a distance cutoff
        obj.r_max = r_max
        obj.num_neighbors = num_neighbors

        obj.query_type = 'nearest'

        obj.scale = scale

        return obj

cdef class NlistptrWrapper:
    cdef freud._locality.NeighborList * nlistptr
    cdef freud._locality.NeighborList * get_ptr(self) nogil

cdef class NeighborQuery:
    cdef freud._locality.NeighborQuery * nqptr
    cdef cbool queryable
    cdef freud.box.Box _box
    cdef const float[:, ::1] points
    cdef freud._locality.NeighborQuery * get_ptr(self) nogil

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
    cdef _cached_query_points
    cdef _cached_box

cdef class AABBQuery(NeighborQuery):
    cdef freud._locality.AABBQuery * thisptr
    cdef NeighborList _nlist

cdef class RawPoints(NeighborQuery):
    cdef freud._locality.RawPoints * thisptr

cdef class _QueryArgs:
    cdef freud._locality.QueryArgs * thisptr

cdef class _Voronoi:
    cdef freud._locality.Voronoi * thisptr
    cdef NeighborList _nlist
    cdef _volumes
    cdef _polytopes
