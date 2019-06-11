# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Methods used throughout freud for convenience

import logging
import numpy as np
import freud.box

logger = logging.getLogger(__name__)


def convert_array(array, dimensions=None, dtype=np.float32,
                  shape=None):
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

    return_arr = np.require(array, dtype=dtype, requirements=['C'])

    if shape is not None:
        if array.ndim != len(shape):
            raise TypeError("array.ndim = {}; expected ndim = {}".format(
                return_arr.ndim, len(shape)))

        for i, s in enumerate(shape):
            if s is not None and return_arr.shape[i] != s:
                shape_str = "(" + ", ".join(str(i) if i is not None
                                            else "..." for i in shape) + ")"
                raise RuntimeError('array.shape= {}; expected shape = {}'
                                   .format(return_arr.shape, shape_str))

    return return_arr


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
