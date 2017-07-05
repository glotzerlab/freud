# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._Index1D cimport Index3D
from freud.util._Boost cimport shared_array
cimport freud._box as box
from libcpp.vector cimport vector

cdef extern from "NeighborList.h" namespace "freud::locality":
    cdef cppclass NeighborList:
        NeighborList()
        NeighborList(unsigned int)

        unsigned int *getNeighbors() const
        float *getWeights() const

        unsigned int getNumBonds() const
        unsigned int filter(const bool*)
        unsigned int filter_r(const box.Box&, const vec3[float]*, const vec3[float]*, float, float)

        unsigned int find_first_index(unsigned int)

        void resize(unsigned int, bool)
        void copy(const NeighborList &)

cdef extern from "LinkCell.h" namespace "freud::locality":
    cdef cppclass IteratorLinkCell:
        IteratorLinkCell()
        IteratorLinkCell(const shared_array[unsigned int] &, unsigned int, unsigned int, unsigned int)
        void copy(const IteratorLinkCell&);
        bool atEnd()
        unsigned int next()
        unsigned int begin()

    cdef cppclass LinkCell:
        LinkCell(const box.Box&, float)
        LinkCell()

        setCellWidth(float)
        updateBox(const box.Box&)
        const vec3[unsigned int] computeDimensions(const box.Box&, float) const
        const box.Box &getBox() const
        const Index3D &getCellIndexer() const
        unsigned int getNumCells() const
        float getCellWidth() const
        unsigned int getCell(const vec3[float]&) const
        IteratorLinkCell itercell(unsigned int) const
        vector[unsigned int] getCellNeighbors(unsigned int) const
        void computeCellList(const box.Box&, const vec3[float]*, unsigned int) nogil except +
        void compute(const box.Box&, const vec3[float]*, unsigned int, const vec3[float]*, unsigned int, bool) nogil except +
        NeighborList *getNeighborList()

cdef extern from "NearestNeighbors.h" namespace "freud::locality":
    cdef cppclass NearestNeighbors:
        NearestNeighbors()
        NearestNeighbors(float, unsigned int, float, bool)

        void setRMax(float)
        const box.Box &getBox() const
        unsigned int getNumNeighbors() const
        float getRMax() const
        unsigned int getUINTMAX() const
        unsigned int getNref() const
        void setCutMode(const bool)
        void compute(const box.Box&, const vec3[float]*, unsigned int, const vec3[float]*, unsigned int, bool) nogil except +
        NeighborList *getNeighborList()
