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
