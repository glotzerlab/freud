cimport freud._parallel as parallel
import freud, freud.parallel

# override TBB's default autoselection. This is necessary because once the automatic selection runs, the user cannot
# change it

# on nyx/flux, default to 1 thread. On all other systems, default to as many cores as are available.
# users on nyx/flux can opt in to more threads by calling setNumThreads again after initialization

def setNumThreads(nthreads=None):
    """Set the number of threads for parallel computation.

    :param nthreads: number of threads to use. If None (default), use all threads available
    :type nthreads: int or None
    """
    if nthreads is None or nthreads < 0:
        nthreads = 0

    freud.parallel._lastThreads = nthreads

    cdef unsigned int cNthreads = nthreads;
    parallel.setNumThreads(cNthreads)
