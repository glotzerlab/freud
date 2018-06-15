# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

# Methods used throughout freud for convenience

import logging
import numpy as np

logger = logging.getLogger(__name__)


def convert_array(array, dimensions, dtype=None,
                  contiguous=True, dim_message=None):
    """
    Function which takes a given array, checks the dimensions,
    and converts to a supplied dtype and/or makes the array
    contiguous as required by the user.

    .. moduleauthor:: Eric Harper <harperic@umich.edu>

    :param array: Array to check and convert
    :param dimensions: Expected dimensions of the array
    :param dtype: dtype to convert the array to if array dtype
                is different. If `None` dtype will not be changed.
    :param contiguous: whether to cast the array to a contiguous
                array. Default behavior casts to a contiguous array
    :param dim_message: passed message to log if the dimensions do
                not match; allows for easier debugging
    :type array: :py:class:`numpy.ndarray`
    :type dimensions: int
    :type dtype: :py:class:`numpy.dtype`
    :type contiguous: bool
    :type dim_message: str
    :return: array
    :rtype: :py:class:`numpy.ndarray`
    """
    array = np.asarray(array)

    if array.ndim != dimensions:
        if dim_message is not None:
            logger.warning(dim_message)
        raise TypeError("array.ndim = {}; expected ndim = {}".format(
            array.ndim, dimensions))
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
