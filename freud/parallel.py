## \package freud.parallel
#
# Methods to control parallel execution
#
import multiprocessing
import platform
import re
from ._freud import setNumThreads;

if (re.match("flux*", platform.node()) is not None) or (re.match("nyx*", platform.node()) is not None):
    setNumThreads(1);
