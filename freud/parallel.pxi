# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

cimport freud._parallel as parallel

# Override TBB's default autoselection. This is necessary because once the
# automatic selection runs, the user cannot change it.

# On nyx/flux, default to 1 thread. On all other systems, default to as many
# cores as are available. Users on nyx/flux can opt in to more threads by
# calling setNumThreads again after initialization.

_numThreads = 0


def setNumThreads(nthreads=None):
    """Set the number of threads for parallel computation.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        nthreads(int, optional): Number of threads to use. If None (default), use all
    threads available.
    """
    if nthreads is None or nthreads < 0:
        nthreads = 0

    _numThreads = nthreads

    cdef unsigned int cNthreads = nthreads
    parallel.setNumThreads(cNthreads)
