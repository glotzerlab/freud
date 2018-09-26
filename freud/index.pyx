# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The index module exposes the :math:`1`-dimensional indexer utilized in freud at
the C++ level.  At the C++ level, freud utilizes flat arrays to represent
multidimensional arrays. :math:`N`-dimensional arrays with :math:`n_i` elements
in each dimension :math:`i` are represented as :math:`1`-dimensional arrays
with :math:`\prod_{i=1}^N n_i` elements.
"""

import numpy as np
import warnings
from freud.errors import FreudDeprecationWarning

from freud.util cimport _Index1D
cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Index2D:
    R"""freud-style indexer for flat arrays.

    Once constructed, the object provides direct access to the flat index
    equivalent:

    - Constructor Calls:

        Initialize with all dimensions identical::

            freud.index.Index2D(w)

        Initialize with each dimension specified::

            freud.index.Index2D(w, h)

    .. note::

        freud indexes column-first i.e. :code:`Index2D(i, j)` will return the
        :math:`1`-dimensional index of the :math:`i^{th}` column and the
        :math:`j^{th}` row. This is the opposite of what occurs in a
        numpy array, in which :code:`array[i, j]` returns the element in the
        :math:`i^{th}` row and the :math:`j^{th}` column.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        w (unsigned int): Width of 2D array (number of columns).
        h (unsigned int): Height of 2D array (number of rows).

    Attributes:
        num_elements (unsigned int): Number of elements in the array.

    Example::

        index = Index2D(10)
        i = index(3, 5)
    """
    cdef _Index1D.Index2D * thisptr

    def __cinit__(self, w, h=None):
        if h is not None:
            self.thisptr = new _Index1D.Index2D(w, h)
        else:
            self.thisptr = new _Index1D.Index2D(w)

    def __dealloc__(self):
        del self.thisptr

    def __call__(self, i, j):
        R"""
        Args:
            i (unsigned int): Column index.
            j (unsigned int): Row index.

        Returns:
            unsigned int: Index in flat (*e.g.* :math:`1`-dimensional) array.
        """
        return self.thisptr.getIndex(i, j)

    @property
    def num_elements(self):
        return self.thisptr.getNumElements()

    def getNumElements(self):
        warnings.warn("The getNumElements function is deprecated in favor "
                      "of the num_elements class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_elements

cdef class Index3D:
    R"""freud-style indexer for flat arrays.

    Once constructed, the object provides direct access to the flat index
    equivalent:

    - Constructor Calls:

        Initialize with all dimensions identical::

            freud.index.Index3D(w)

        Initialize with each dimension specified::

            freud.index.Index3D(w, h, d)

    .. note:: freud indexes column-first i.e. Index3D(i, j, k) will return the
              :math:`1`-dimensional index of the :math:`i^{th}` column,
              :math:`j^{th}` row, and the :math:`k^{th}` frame. This is the
              opposite of what occurs in a numpy array, in which
              :code:`array[i, j, k]` returns the element in the :math:`i^{th}`
              frame, :math:`j^{th}` row, and the :math:`k^{th}` column.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    Args:
        w (unsigned int): Width of 2D array (number of columns).
        h (unsigned int): Height of 2D array (number of rows).
        d (unsigned int): Depth of 2D array (number of frames).

    Attributes:
        num_elements (unsigned int): Number of elements in the array.

    Example::

        index = Index3D(10)
        i = index(3, 5, 4)
    """
    cdef _Index1D.Index3D * thisptr

    def __cinit__(self, w, h=None, d=None):
        if h is not None:
            self.thisptr = new _Index1D.Index3D(w, h, d)
        else:
            self.thisptr = new _Index1D.Index3D(w)

    def __dealloc__(self):
        del self.thisptr

    def __call__(self, i, j, k):
        R"""
        Args:
            i (unsigned int): Column index.
            j (unsigned int): Row index.
            k (unsigned int): Frame index.

        Returns:
            unsigned int: Index in flat (*e.g.* :math:`1`-dimensional) array.
        """
        return self.thisptr.getIndex(i, j, k)

    @property
    def num_elements(self):
        return self.thisptr.getNumElements()

    def getNumElements(self):
        warnings.warn("The getNumElements function is deprecated in favor "
                      "of the num_elements class attribute and will be "
                      "removed in a future version of freud.",
                      FreudDeprecationWarning)
        return self.num_elements
