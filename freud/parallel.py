# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

import platform
import re
from ._freud import setNumThreads

if (re.match("flux.", platform.node()) is not None) or (
        re.match("nyx.", platform.node()) is not None):
    setNumThreads(1)


class NumThreads:
    """Context manager for managing the number of threads to use.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    :param N: Number of threads to use in this context. Defaults to
              :code:`None`, which will use all available threads.
    :type N: int or None
    """

    def __init__(self, N=None):
        self.restore_N = _freud._numThreads
        self.N = N

    def __enter__(self):
        setNumThreads(self.N)

    def __exit__(self, *args):
        setNumThreads(self.restore_N)
