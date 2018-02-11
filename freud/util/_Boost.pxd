# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

cdef extern from "boost/shared_array.hpp" namespace "boost":
    cdef cppclass shared_array[T]:
        T* get()

    cdef cppclass shared_ptr[T]:
        T* get()
