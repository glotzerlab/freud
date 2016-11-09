# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

cdef extern from "boost/shared_array.hpp" namespace "boost":
    cdef cppclass shared_array[T]:
        T* get()

    cdef cppclass shared_ptr[T]:
        T* get()
