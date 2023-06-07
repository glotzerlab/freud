# Copyright (c) 2010-2023 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

cimport freud._locality
cimport freud.box
from freud.util cimport _Compute


cdef NeighborList _nlist_from_cnlist(freud._locality.NeighborList *c_nlist)

cdef class NeighborQueryResult:
    cdef NeighborQuery nq
    cdef const float[:, ::1] points
    cdef _QueryArgs query_args

    # This had to be implemented as a factory because the constructors will
    # always get called with Python objects as arguments, and we need typed
    # objects. See the link below for more information.
    # https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#initialisation-methods-cinit-and-init
    # This needed to be declared inline because it needed to be in the pxd,
    # which in turn was the only way to get the staticmethod decorator to
    # compile with a cdef method.
    @staticmethod
    cdef inline NeighborQueryResult init(NeighborQuery nq, const float[:, ::1]
                                         points, _QueryArgs query_args):

        obj = NeighborQueryResult()

        obj.nq = nq
        obj.points = points
        obj.query_args = query_args

        return obj

cdef class NeighborQuery:
    cdef freud._locality.NeighborQuery * nqptr
    cdef const float[:, ::1] points
    cdef freud._locality.NeighborQuery * get_ptr(self)

cdef class NeighborList:
    cdef freud._locality.NeighborList * thisptr
    cdef char _managed
    cdef freud.util._Compute _compute

    cdef freud._locality.NeighborList * get_ptr(self)
    cdef void copy_c(self, NeighborList other)

cdef class LinkCell(NeighborQuery):
    cdef freud._locality.LinkCell * thisptr

cdef class AABBQuery(NeighborQuery):
    cdef freud._locality.AABBQuery * thisptr

cdef class _RawPoints(NeighborQuery):
    cdef freud._locality.RawPoints * thisptr

cdef class _QueryArgs:
    cdef freud._locality.QueryArgs * thisptr

cdef class _PairCompute(_Compute):
    pass

cdef class _SpatialHistogram(_PairCompute):
    cdef float r_max
    cdef freud._locality.BondHistogramCompute *histptr

cdef class _SpatialHistogram1D(_SpatialHistogram):
    pass

cdef class PeriodicBuffer(_Compute):
    cdef freud._locality.PeriodicBuffer * thisptr

cdef class Voronoi(_Compute):
    cdef freud._locality.Voronoi * thisptr
    cdef NeighborList _nlist
    cdef freud.box.Box _box

cdef class Filter(_PairCompute):
    cdef freud._locality.Filter *_filterptr

cdef class FilterSANN(Filter):
    cdef freud._locality.FilterSANN *_thisptr

cdef class FilterRAD(Filter):
    cdef freud._locality.FilterRAD *_thisptr
