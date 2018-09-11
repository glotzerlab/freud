# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._Index1D cimport Index3D
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
cimport freud._box

cdef extern from "NeighborList.h" namespace "freud::locality":
    cdef cppclass NeighborList:
        NeighborList()
        NeighborList(size_t)

        size_t * getNeighbors()
        float * getWeights()

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

    cdef cppclass LinkCell:
        LinkCell(const freud._box.Box &, float) except +
        LinkCell()

        setCellWidth(float)
        updateBox(const freud._box.Box &)
        const vec3[unsigned int] computeDimensions(
            const freud._box.Box &,
            float) const
        const freud._box.Box & getBox() const
        const Index3D & getCellIndexer() const
        unsigned int getNumCells() const
        float getCellWidth() const
        unsigned int getCell(const vec3[float] &) const
        IteratorLinkCell itercell(unsigned int) const
        vector[unsigned int] getCellNeighbors(unsigned int) const
        void computeCellList(
            const freud._box.Box &,
            const vec3[float]*,
            unsigned int) nogil except +
        void compute(
            const freud._box.Box &,
            const vec3[float]*,
            unsigned int,
            const vec3[float]*,
            unsigned int,
            bool) nogil except +
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
        void setCutMode(const bool)
        void compute(
            const freud._box.Box &,
            const vec3[float]*,
            unsigned int,
            const vec3[float]*,
            unsigned int,
            bool) nogil except +
        NeighborList * getNeighborList()
