cimport freud._parallel as parallel

# override TBB's default autoselection. This is necessary because once the automatic selection runs, the user cannot
# change it

# on nyx/flux, default to 1 thread. On all other systems, default to as many cores as are available.
# users on nyx/flux can opt in to more threads by calling setNumThreads again after initialization

def setNumThreads(unsigned int nthreads):
    """Set the number of threads for parallel computation.

    :param nthreads: number of threads to use
    :type nthreads: unsigned int
    """
    parallel.setNumThreads(nthreads)
