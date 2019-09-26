# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util cimport vec3
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.pair cimport pair
cimport freud._box
cimport freud.util

cdef extern from "NeighborBond.h" namespace "freud::locality":
    cdef cppclass NeighborBond:
        unsigned int id
        unsigned int ref_id
        float distance
        float weight
        bool operator==(NeighborBond)
        bool operator!=(NeighborBond)
        bool operator<(NeighborBond)

cdef extern from "NeighborQuery.h" namespace "freud::locality":

    ctypedef enum QueryType "freud::locality::QueryArgs::QueryType":
        none "freud::locality::QueryArgs::QueryType::none"
        ball "freud::locality::QueryArgs::QueryType::ball"
        nearest "freud::locality::QueryArgs::QueryType::nearest"

    cdef cppclass QueryArgs:
        QueryType mode
        int num_neighbors
        float r_max
        float r_min
        float r_guess
        float scale
        bool exclude_ii

    cdef cppclass NeighborQuery:
        NeighborQuery() except +
        NeighborQuery(const freud._box.Box &, const vec3[float]*, unsigned int)
        shared_ptr[NeighborQueryIterator] query(
            const vec3[float]*, unsigned int, QueryArgs) except +
        const freud._box.Box & getBox() const
        const vec3[float]* getPoints const
        const unsigned int getNPoints const
        const vec3[float] operator[](unsigned int) const

    NeighborBond ITERATOR_TERMINATOR \
        "freud::locality::NeighborQueryIterator::ITERATOR_TERMINATOR"

    cdef cppclass NeighborQueryIterator:
        NeighborQueryIterator()
        NeighborQueryIterator(NeighborQuery*, vec3[float]*, unsigned int)
        bool end()
        NeighborBond next()
        NeighborList *toNeighborList()

cdef extern from "RawPoints.h" namespace "freud::locality":

    cdef cppclass RawPoints(NeighborQuery):
        RawPoints() except +
        RawPoints(const freud._box.Box, const vec3[float]*, unsigned int)

cdef extern from "NeighborList.h" namespace "freud::locality":
    cdef cppclass NeighborList:
        NeighborList()
        NeighborList(unsigned int)
        NeighborList(unsigned int, const unsigned int*, unsigned int,
                     const unsigned int*, unsigned int, const float*,
                     const float*) except +

        freud.util.ManagedArray[unsigned int] &getNeighbors()
        freud.util.ManagedArray[float] &getDistances()
        freud.util.ManagedArray[float] &getWeights()
        freud.util.ManagedArray[float] &getSegments()
        freud.util.ManagedArray[float] &getCounts()

        unsigned int getNumBonds() const
        unsigned int getNumPoints() const
        unsigned int getNumQueryPoints() const
        void setNumBonds(unsigned int, unsigned int, unsigned int)
        unsigned int filter(const bool*) except +
        unsigned int filter_r(float, float) except +

        unsigned int find_first_index(unsigned int)

        void resize(unsigned int)
        void copy(const NeighborList &)
        void validate(unsigned int, unsigned int) except +

cdef extern from "LinkCell.h" namespace "freud::locality":
    cdef cppclass IteratorLinkCell:
        IteratorLinkCell()
        IteratorLinkCell(
            const shared_ptr[unsigned int] &,
            unsigned int,
            unsigned int,
            unsigned int)
        void copy(const IteratorLinkCell &)
        bool atEnd()
        unsigned int next()
        unsigned int begin()

    cdef cppclass LinkCell(NeighborQuery):
        LinkCell() except +
        LinkCell(const freud._box.Box &,
                 float,
                 const vec3[float]*,
                 unsigned int) except +
        const vec3[unsigned int] computeDimensions(
            const freud._box.Box &,
            float) const
        unsigned int getNumCells() const
        float getCellWidth() const
        unsigned int getCell(const vec3[float] &) const
        IteratorLinkCell itercell(unsigned int) const
        vector[unsigned int] getCellNeighbors(unsigned int) const
        void computeCellList(
            const freud._box.Box &,
            const vec3[float]*,
            unsigned int) except +

cdef extern from "AABBQuery.h" namespace "freud::locality":
    cdef cppclass AABBQuery(NeighborQuery):
        AABBQuery() except +
        AABBQuery(const freud._box.Box,
                  const vec3[float]*,
                  unsigned int) except +

cdef extern from "Voronoi.h" namespace "freud::locality":
    cdef cppclass Voronoi:
        Voronoi()
        void compute(
            const freud._box.Box &,
            const vec3[double]*,
            const int*,
            const int*,
            unsigned int,
            unsigned int,
            const int*,
            const vec3[double]*,
            const int*) except +
        NeighborList * getNeighborList()

cdef extern from "BondHistogramCompute.h" namespace "freud::locality":
    cdef cppclass BondHistogramCompute:
        BondHistogramCompute()

        const freud._box.Box & getBox() const
        void reset()
        const freud.util.ManagedArray[unsigned int] &getBinCounts()
        vector[vector[float]] getBinEdges() const
        vector[vector[float]] getBinCenters() const
        vector[pair[float, float]] getBounds() const
        vector[unsigned int] getAxisSizes() const
