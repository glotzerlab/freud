# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.parallel` module controls the parallelization behavior of
freud, determining how many threads the TBB-enabled parts of freud will use.
freud uses all available threads for parallelization unless directed otherwise.
"""

cimport freud._parallel

_num_threads = 0


def get_num_threads():
    R"""Get the number of threads for parallel computation.

    Returns:
        (int): Number of threads.
    """
    global _num_threads
    return _num_threads


def set_num_threads(nthreads=None):
    R"""Set the number of threads for parallel computation.

    Args:
        nthreads (int, optional):
            Number of threads to use. If :code:`None`, use all threads
            available. (Default value = :code:`None`).
    """
    global _num_threads
    if nthreads is None or nthreads < 0:
        nthreads = 0

    _num_threads = nthreads

    cdef unsigned int cNthreads = nthreads
    freud._parallel.setNumThreads(cNthreads)


class NumThreads:
    R"""Context manager for managing the number of threads to use.

    Args:
        N (int, optional): Number of threads to use in this context. If
            :code:`None`, which will use all available threads.
            (Default value = :code:`None`).
    """

    def __init__(self, N=None):
        global _num_threads
        self.restore_N = _num_threads
        self.N = N

    def __enter__(self):
        set_num_threads(self.N)
        return self

    def __exit__(self, *args):
        set_num_threads(self.restore_N)
