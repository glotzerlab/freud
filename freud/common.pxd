cdef class Compute:
    cdef public _called_compute

cdef class PairCompute(Compute):
    pass
