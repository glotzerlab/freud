cdef extern from "tbb_config.h" namespace "freud::parallel":
    void setNumThreads(unsigned int)
