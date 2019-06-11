# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Methods used throughout freud for convenience

import logging
import numpy as np
import freud.box
from functools import partial

logger = logging.getLogger(__name__)


cdef class Compute:
    def __cinit__(self):
        self._called_compute = {"compute": False}

    @staticmethod
    def _compute(key="compute"):
        def _compute_with_key(func, key):
            def wrapper(self, *args, **kwargs):
                self._called_compute[key] = True
                func(self, *args, **kwargs)
            return wrapper
        return partial(_compute_with_key, key=key)

    @staticmethod
    def _computed_property(key="compute"):
        def _computed_property_with_key(func, key="compute"):
            @property
            def wrapper(self, *args, **kwargs):
                if not self._called_compute[key]:
                    raise AttributeError("Property not computed. "
                                         "Call {key} first.".format(key=key))
                return func(self, *args, **kwargs)
            return wrapper
        return partial(_computed_property_with_key, key=key)

    @staticmethod
    def _reset(func):
        def wrapper(self, *args, **kwargs):
            for k in self._called_compute:
                self._called_compute[k] = False
            func(self, *args, **kwargs)
        return wrapper


def convert_array(array, dimensions=None, dtype=np.float32):
    """Function which takes a given array, checks the dimensions,
    and converts to a supplied dtype and/or makes the array
    contiguous.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        array (:class:`numpy.ndarray`): Array to check and convert.
        dimensions (int): Expected dimensions of the array. If 'None',
            no dimensionality check will be done (Default value = 'None').
        dtype: code:`dtype` to convert the array to if :code:`array.dtype`
            is different. If `None`, :code:`dtype` will not be changed
            (Default value = `numpy.float32`).

    Returns:
        :class:`numpy.ndarray`: Array.
    """
    array = np.asarray(array)

    if dimensions is not None and array.ndim != dimensions:
        raise TypeError("array.ndim = {}; expected ndim = {}".format(
            array.ndim, dimensions))
    return np.require(array, dtype=dtype, requirements=['C'])


def convert_box(box):
    """Function which takes a box-like object and attempts to convert it to
    :class:`freud.box.Box`. Existing :class:`freud.box.Box` objects are
    used directly.

    .. moduleauthor:: Bradley Dice <bdice@bradleydice.com>

    Args:
        box (box-like object (see :meth:`freud.box.Box.from_box`)): Box to
            check and convert if needed.

    Returns:
        :class:`freud.box.Box`: freud box.
    """
    if not isinstance(box, freud.box.Box):
        try:
            box = freud.box.Box.from_box(box)
        except ValueError:
            raise
    return box
