## \package freud.common
#
# Methods used throughout freud for convenience
import numpy as np

def convert_array(array, dimensions, dtype=None, contiguous=True):
    global DTYPE
    if dtype is None:
        dtype = DTYPE
    assert dtype is not None
    if array.shape != dimensions:
        raise TypeError("Wrong dimensions!")
    if contiguous:
        return np.ascontiguous(array, dtype=dtype)
    else:
        return np.asarray(array, dtype=dtype)
