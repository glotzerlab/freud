# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

cdef extern from "Index1D.h":
    cdef cppclass Index2D:
        Index2D(unsigned int)
        Index2D(unsigned int, unsigned int)
        unsigned int getIndex "operator()"(unsigned int, unsigned int)
        unsigned int getNumElements()

    cdef cppclass Index3D:
        Index3D(unsigned int)
        Index3D(unsigned int, unsigned int, unsigned int)
        unsigned int getIndex "operator()"(unsigned int, unsigned int, unsigned int)
        unsigned int getNumElements()
        unsigned int getW()
        unsigned int getH()
        unsigned int getD()
