
cdef extern from "cudacpu_vector_types.h":
    cdef struct float3:
        float x
        float y
        float z