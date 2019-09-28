# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

cimport freud._box

cdef class Box:
    cdef freud._box.Box * thisptr

cdef BoxFromCPP(const freud._box.Box & cppbox)

cdef class PeriodicBuffer:
    cdef freud._box.PeriodicBuffer * thisptr
