
from freud.trajectory.trajectory cimport Box

cdef extern from "CorrelationFunction.h" namespace "freud::density":
    cdef cppclass CorrelationFunction[T]:
        CorrelationFunction(float, float)
        cdef const Box &getBox() const
        void resetCorrelationFunction()
        void
