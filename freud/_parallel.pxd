# Copyright (c) 2010-2016 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

cdef extern from "tbb_config.h" namespace "freud::parallel":
    void setNumThreads(unsigned int)
