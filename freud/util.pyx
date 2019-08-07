# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import sys
from functools import wraps

from cython.operator cimport dereference
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

    def __cinit__(self):
        # This class should be initialized via the factory "init" function.
        pass

    def acquire(self):
        """Acquire ownership of data from the synchronized C++ array."""
        if self.var_typenum == np.NPY_UINT32:
            self.thisptr.uint_ptr.acquire(dereference(self.sourceptr.uint_ptr))
        return self

    def release(self):
        """Return ownership of data to the synchronized C++ array."""
        if self.var_typenum == np.NPY_UINT32:
            self.sourceptr.uint_ptr.acquire(dereference(self.thisptr.uint_ptr))
        return self

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Set the shape of the output numpy array when :meth:`numpy.asarray`
        is called."""
        self._shape = shape

    def __dealloc__(self):
        if self.var_typenum == np.NPY_UINT32:
            del self.thisptr.uint_ptr

    def __array__(self):
        """Convert the underlying data array into a read-only numpy array."""
        if self.shape == tuple():
            raise ValueError("You must specify the shape of the numpy array "
                             "to be created by calling set_shape.")
        cdef unsigned int ndim = len(self.shape)

        # These arrays must be allocated at compile time, so we make separate
        # branches for different dimensionalities.
        cdef np.npy_intp nP1[1]
        cdef np.npy_intp nP2[2]
        cdef np.npy_intp nP3[3]
        cdef np.ndarray arr
        if len(self.shape) == 1:
            nP1[0] = self.shape[0]
            arr = np.PyArray_SimpleNewFromData(
                ndim, nP1, self.var_typenum, self.get())
        elif len(self.shape) == 2:
            nP2[0] = self.shape[0]
            nP2[1] = self.shape[1]
            arr = np.PyArray_SimpleNewFromData(
                ndim, nP2, self.var_typenum, self.get())
        elif len(self.shape) == 3:
            nP3[0] = self.shape[0]
            nP3[1] = self.shape[1]
            nP3[2] = self.shape[2]
            arr = np.PyArray_SimpleNewFromData(
                ndim, nP3, self.var_typenum, self.get())

        arr.setflags(write=False)

        self.set_as_base(arr)
        return arr


def resolve_arrays(array_names):
    """Decorator that ensures that all ManagedArrays are released to C++ if
    possible to minimize memory reallocations.

    Args:
        array_names (str or list(str)): ManagedArray attributes that should be
                                        released and reacquired.
    Returns:
        callable: A function that behaves as a decorator for a compute call.
    """
    if type(array_names) is str:
        array_names = [array_names]

    def decorator(func):
        """The wrapper is the actual decorator that is called on the compute
        function.

        Args:
            func (callable): The compute function to manage arrays for.

        Returns:
            callable: A compute function that manages arrays.
        """
        @wraps(func)
        def acquire_and_compute(self, *args, **kwargs):
            """This function is the replacement for compute.

            Args:
                *args: Any positional arguments to the compute call.
                **kwargs: Any keyword arguments to the compute call.

            Returns:
                callable: A compute function that manages arrays.
            """
            # If other objects (e.g. NumPy arrays) are referencing this one,
            # then we reallocate a new Python wrapper object. Otherwise, we
            # relinquish control of the underlying array to the C++ class for
            # its computation.  In either case, the Python wrapper class
            # reacquires ownership at the end.
            cdef freud.util.ManagedArrayManager array
            for array_name in array_names:
                refcount = sys.getrefcount(getattr(self, array_name))
                array = <freud.util.ManagedArrayManager> getattr(
                    self, array_name)
                if refcount <= 2:
                    array.release()
                else:
                    setattr(self, array_name,
                            freud.util.ManagedArrayManager.init(
                                array.sourceptr.uint_ptr,
                                arr_type_t.UNSIGNED_INT))
            ret_val = func(self, *args, **kwargs)

            # Store the array locally again.
            for array_name in array_names:
                array = <freud.util.ManagedArrayManager> getattr(
                    self, array_name)
                array.acquire()

            return ret_val
        return acquire_and_compute
    return decorator
