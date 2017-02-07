## \package freud.common
#
# Methods used throughout freud for convenience
import logging
logger = logging.getLogger(__name__)
import numpy as np

def convert_array(array, dimensions, dtype=None, contiguous=True):
    """
    Function which takes a given array, checks the dimensions, and converts to a supplied dtype and/or makes the array
    contiguous as required by the user.
    """
    if array.ndim != dimensions:
        raise TypeError("Wrong dimensions for supplied array!")
    requirements = None
    if contiguous == True:
        if array.flags.contiguous == False:
            msg = 'converting supplied array to contiguous'
            logger.warning(msg)
        requirements = ["C"]
    if dtype is not None and dtype != array.dtype:
        msg = 'converting supplied array dtype {} to dtype {}'.format(array.dtype, dtype)
        logger.warning(msg)
    return np.require(array, dtype=dtype, requirements=requirements)
