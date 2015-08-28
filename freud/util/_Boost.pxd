cdef extern from "boost/shared_array.hpp" namespace "boost":
    cdef cppclass shared_array[T]:
        T* get()
