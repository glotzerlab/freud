# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import sys
import numpy as np

from functools import wraps

from cython.operator cimport dereference
from libcpp.vector cimport vector

cimport freud.util
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class ManagedArrayManager:
    """Class responsible for synchronizing ownership between two ManagedArray
    instances.

    The purpose of this container is to minimize memory copies while avoiding
    reference rot. Users obtaining numpy arrays as outputs from a compute class
    have a reasonable expectation that these arrays will remain valid after the
    compute class recomputes or goes out of scope. To enforce this requirement,
    all such data members should be stored in using the ManagedArray class in
    C++ and then synchronized using a ManagedArrayManager in Python. This class
    retains a Python-level version of the array that maintains ownership of the
    data between compute calls and only relinquishes ownership to the C++ class
    if no outstanding references to the data exist in Python.

    This class should always be initialized using the static factory
    :meth:`~ManagedArrayManager.init` method, which creates the Python copy of
    a ManagedArray provided the instance member of the underlying C++ compute
    class.

    .. moduleauthor:: Vyas Ramasubramani <vramasub@umich.edu>
    """

    def __cinit__(self, arr_type, typenum):
        # This class should generally be initialized via the factory "init"
        # function, but some logic is included here for ease of use.
        self.data_type = arr_type
        self.var_typenum = typenum
        self.thisptr.null_ptr = NULL

    @property
    def shape(self):
        return tuple(self.thisptr.uint_ptr.shape())

    def __dealloc__(self):
        if self.var_typenum == np.NPY_UINT32:
            del self.thisptr.uint_ptr

    def __array__(self):
        """Convert the underlying data array into a read-only numpy array.

        To simplify the code, we allocate a single linear array and then
        reshape it on return. The reshape is just a view on the arr array
        created below, so it creates a chain reshaped_array->arr->self that
        ensures proper garbage collection.
        """
        cdef np.npy_intp size[1]
        cdef np.ndarray arr
        size[0] = np.prod(self.shape)
        arr = np.PyArray_SimpleNewFromData(
            1, size, self.var_typenum, self.get())

        arr.setflags(write=False)
        self.set_as_base(arr)
        return np.reshape(arr, self.shape)


def resolve_arrays(array_names):
    """Decorator that ensures that all ManagedArrays that are still referenced
    somewhere in Python are reallocated by the C++ instance.

    This function is actually a wrapper that parses the class members that are
    instances of ManagedArrayManager and then returns the decorator that loops
    over these and performs the necessary reallocation.

    Args:
        array_names (str or list(str)): ManagedArray attributes that should be
                                        reallocated in C++.
    Returns:
        callable: A function that behaves as a decorator for a compute call.
    """
    if type(array_names) is str:
        array_names = [array_names]

    def decorator(func):
        """The actual decorator that is called on the compute function.

        Args:
            func (callable): The compute function to manage arrays for.

        Returns:
            callable: An augmented compute function that manages arrays.
        """
        @wraps(func)
        def acquire_and_compute(self, *args, **kwargs):
            """This function is the replacement for compute.

            This function accepts all the arguments for the compute call and
            forwards them through, but first it loops over the
            ManagedArrayManagers attached to this compute function and checks
            their reference counts. If there are any references beyond the
            expected one (the class member itself), we reallocate the C++
            compute class's member array (which is possible since the
            ManagedArray allocates memory using a pointer to a pointer.

            Args:
                *args: Any positional arguments to the compute call.
                **kwargs: Any keyword arguments to the compute call.

            Returns:
                callable: A compute function that manages arrays.
            """
            cdef freud.util.ManagedArrayManager array, new_array
            for array_name in array_names:
                refcount = sys.getrefcount(getattr(self, array_name))
                array = <freud.util.ManagedArrayManager> getattr(
                    self, array_name)
                if refcount > 2:
                    new_array = freud.util.ManagedArrayManager.init(
                        array.thisptr.uint_ptr, arr_type_t.UNSIGNED_INT)
                    array.dissociate()
                    # For now I'll just call reallocate, but ideally I should
                    # intelligently choose to resize here if needed.
                    new_array.reallocate()
                    setattr(self, array_name, new_array)
            ret_val = func(self, *args, **kwargs)

            return ret_val
        return acquire_and_compute
    return decorator
