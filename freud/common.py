# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Methods used throughout freud for convenience

import logging
import numpy as np
import freud.box

logger = logging.getLogger(__name__)


def convert_array(array, dimensions, dtype=None,
                  contiguous=True, array_name=None):
    """Function which takes a given array, checks the dimensions,
    and converts to a supplied dtype and/or makes the array
    contiguous as required by the user.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    Args:
        array (:class:`numpy.ndarray`): Array to check and convert.
        dimensions (int): Expected dimensions of the array.
        dtype: code:`dtype` to convert the array to if :code:`array.dtype`
            is different. If `None`, :code:`dtype` will not be changed.
            (Default value = None).
        contiguous (bool): Whether to cast the array to a contiguous (Default
            value = True).
        array. Default behavior casts to a contiguous array.
        array_name (str): Name of the array, used for errors (Default value =
            None).

    Returns:
        py:class:`numpy.ndarray`: Array.
    """
    array = np.asarray(array)

    if array.ndim != dimensions:
        raise TypeError("{}.ndim = {}; expected ndim = {}".format(
            array_name or "array", array.ndim, dimensions))
    requirements = None
    if contiguous:
        if not array.flags.contiguous:
            msg = 'Converting supplied array to contiguous.'
            logger.info(msg)
        requirements = ["C"]
    if dtype is not None and dtype != array.dtype:
        msg = 'Converting supplied array dtype {} to dtype {}.'.format(
            array.dtype, dtype)
        logger.info(msg)
    return np.require(array, dtype=dtype, requirements=requirements)


def convert_box(box):
    """Function which takes a box-like object and attempts to convert it to
    :class:`freud.box.Box`. Existing :class:`freud.box.Box` objects are
    used directly.

    .. moduleauthor:: Bradley Dice <bdice@umich.edu>

    Args:
      box (box-like object (see :meth:`freud.box.Box.from_box`)): Box to
          check and convert if needed.

    Returns:
      py:class:`freud.box.Box`: freud box.
    """
    if not isinstance(box, freud.box.Box):
        try:
            box = freud.box.Box.from_box(box)
        except ValueError:
            raise
    return box
