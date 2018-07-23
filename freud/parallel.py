# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

R"""
The :py:class:`freud.parallel` module controls the parallelization behavior of freud, determining how many threads the TBB-enabled parts of freud will use.
By default, freud tries to use all available threads for parallelization unless directed otherwise, with one exception.
"""

import platform
import re
from ._freud import setNumThreads, _numThreads

if (re.match("flux.", platform.node()) is not None) or (
        re.match("nyx.", platform.node()) is not None):
    setNumThreads(1)


class NumThreads:
    """Context manager for managing the number of threads to use.

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
