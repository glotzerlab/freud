# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.parallel` module controls the parallelization behavior of
freud, determining how many threads the TBB-enabled parts of freud will use.
By default, freud tries to use all available threads for parallelization unless
directed otherwise, with one exception.
"""

import platform
import re
cimport freud._parallel

# Override TBB's default autoselection. This is necessary because once the
# automatic selection runs, the user cannot change it.

# On nyx/flux, default to 1 thread. On all other systems, default to as many
# cores as are available. Users on nyx/flux can opt in to more threads by
# calling setNumThreads again after initialization.

_numThreads = 0


def setNumThreads(nthreads=None):
    R"""Set the number of threads for parallel computation.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        nthreads(int, optional):
            Number of threads to use. If None (default), use all threads
            available.
    """
    if nthreads is None or nthreads < 0:
        nthreads = 0

    _numThreads = nthreads

    cdef unsigned int cNthreads = nthreads
    freud._parallel.setNumThreads(cNthreads)


if (re.match("flux.", platform.node()) is not None) or (
        re.match("nyx.", platform.node()) is not None):
    setNumThreads(1)


class NumThreads:
    R"""Context manager for managing the number of threads to use.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        N (int): Number of threads to use in this context. Defaults to
                 None, which will use all available threads.
    """

    def __init__(self, N=None):
        self.restore_N = _numThreads
        self.N = N

    def __enter__(self):
        setNumThreads(self.N)

    def __exit__(self, *args):
        setNumThreads(self.restore_N)
