import multiprocessing

## \package freud.parallel
#
# Methods to control parallel execution
#

from _freud import setNumThreads;

# override TBB's default autoselection. This is necessary because once the automatic selection runs, the user cannot
# change it
setNumThreads(multiprocessing.cpu_count());
# setNumThreads(1);
