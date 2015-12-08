import multiprocessing
import platform
import re

## \package freud.parallel
#
# Methods to control parallel execution
#

from _freud import setNumThreads;

# override TBB's default autoselection. This is necessary because once the automatic selection runs, the user cannot
# change it

# on nyx/flux, default to 1 thread. On all other systems, default to as many cores as are available.
# users on nyx/flux can opt in to more threads by calling setNumThreads again after initialization

if (re.match("flux*", platform.node()) is not None) or (re.match("nyx*", platform.node()) is not None):
    setNumThreads(1);
else:
    setNumThreads(multiprocessing.cpu_count());
