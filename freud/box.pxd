# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

from . cimport _box

cdef class Box:
    cdef _box.Box * thisptr

cdef BoxFromCPP(const _box.Box & cppbox)
