# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud._util cimport vec3, quat, ManagedArray

cdef class ManagedArrayWrapper:
    def __cinit__(self):
        self.thisptr = NULL

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
