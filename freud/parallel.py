## \package freud.parallel
#
# Methods to control parallel execution
#
import multiprocessing
import platform
import re
from ._freud import setNumThreads;

_numThreads = 0

if (re.match("flux*", platform.node()) is not None) or (re.match("nyx*", platform.node()) is not None):
    setNumThreads(1);

class NumThreads:
    def __init__(self, N=None):
        self.restore_N = _numThreads
        self.N = N

    def __enter__(self):
        setNumThreads(self.N)

    def __exit__(self, *args):
        setNumThreads(self.restore_N)
