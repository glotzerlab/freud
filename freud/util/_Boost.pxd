cdef extern from "boost/shared_array.hpp" namespace "boost":
    cdef cppclass shared_array[T]:
        T* get()

    cdef cppclass shared_ptr[T]:
        T* get()
