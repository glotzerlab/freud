cdef extern from "Index1D.h":
    cdef cppclass Index2D:
        Index2D(unsigned int)
        Index2D(unsigned int, unsigned int)
        unsigned int operator()(unsigned int, unsigned int)
        unsigned int getNumElements()

    cdef cppclass Index3D:
        Index3D(unsigned int)
        Index3D(unsigned int, unsigned int, unsigned int)
        unsigned int operator()(unsigned int, unsigned int, unsigned int)
        unsigned int getNumElements()
        unsigned int getW()
        unsigned int getH()
        unsigned int getD()
