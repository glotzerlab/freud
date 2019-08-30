# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util cimport vec3
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
cimport freud._box

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
        float scale
        bool exclude_ii

    cdef cppclass NeighborQuery:
        NeighborQuery()
        NeighborQuery(const freud._box.Box &, const vec3[float]*, unsigned int)
        shared_ptr[NeighborQueryIterator] query(
            const vec3[float]*, unsigned int, QueryArgs) except +
        const freud._box.Box & getBox() const
        const vec3[float]* getPoints const
        const unsigned int getNPoints const
        const vec3[float] operator[](unsigned int) const
        void validateQueryArgs(QueryArgs) except +

    NeighborBond ITERATOR_TERMINATOR \
        "freud::locality::NeighborQueryIterator::ITERATOR_TERMINATOR"

    cdef cppclass NeighborQueryIterator:
        NeighborQueryIterator()
        NeighborQueryIterator(NeighborQuery*, vec3[float]*, unsigned int)
        bool end()
        NeighborBond next()
        NeighborList *toNeighborList()

    cdef cppclass RawPoints(NeighborQuery):
        RawPoints()
        RawPoints(const freud._box.Box, const vec3[float]*, unsigned int)

cdef extern from "NeighborList.h" namespace "freud::locality":
    cdef cppclass NeighborList:
        NeighborList()
        NeighborList(size_t)

        size_t * getNeighbors()
        float * getWeights()
        float * getDistances()

        size_t getNumI() const
        size_t getNumBonds() const
        void setNumBonds(size_t, size_t, size_t)
        size_t filter(const bool*)
        size_t filter_r(
            const freud._box.Box &,
            const vec3[float]*,
            const vec3[float]*,
            float, float)

        size_t find_first_index(size_t)

        # Include separate definitions for resize with and without optional
        # parameter
        void resize(size_t)
        void resize(size_t, bool)
        void copy(const NeighborList &)
        void validate(size_t, size_t) except +

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
        LinkCell()
        LinkCell(const freud._box.Box &, float) except +
        LinkCell(const freud._box.Box &, float,
                 const vec3[float]*, unsigned int) except +

        setCellWidth(float) except +
        updateBox(const freud._box.Box &) except +
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
        NeighborList * getNeighborList()

cdef extern from "NearestNeighbors.h" namespace "freud::locality":
    cdef cppclass NearestNeighbors:
        NearestNeighbors()
        NearestNeighbors(float, unsigned int, float, bool)

        void setRMax(float)
        const freud._box.Box & getBox() const
        unsigned int getNumNeighbors() const
        float getRMax() const
        unsigned int getUINTMAX() const
        unsigned int getNref() const
        unsigned int getNp() const
        void setCutMode(const bool)
        void compute(
            const freud._box.Box &,
            const vec3[float]*,
            unsigned int,
            const vec3[float]*,
            unsigned int,
            bool) except +
        NeighborList * getNeighborList()

cdef extern from "AABBQuery.h" namespace "freud::locality":
    cdef cppclass AABBQuery(NeighborQuery):
        AABBQuery()
        AABBQuery(const freud._box.Box, const vec3[float]*, unsigned int)

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
