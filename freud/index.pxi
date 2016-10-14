cimport freud.util._Index1D as Index1D

import numpy as np
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Index2D:
    """Freud-style indexer for flat arrays.

    Freud utilizes "flat" arrays at the C++ level i.e. an :math:`n`-dimensional array with :math:`n_i` elements in each \
    index is represented as a :math:`1`-dimensional array with :math:`\prod\limits_i n_i` elements.

    .. note:: Freud indexes column-first i.e. Index2D(i, j) will return the :math:`1`-dimensional index of the \
    :math:`i^{th}` column and the :math:`j^{th}` row. This is the opposite of what occurs in a numpy array, in which \
    array[i, j] returns the element in the :math:`i^{th}` row and the :math:`j^{th}` column

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    :param w: width of 2D array (number of columns)
    :param h: height of 2D array (number of rows)
    :type w: unsigned int
    :type h: unsigned int

    - Constructor Calls:

        Initialize with all dimensions identical::

            freud.index.Index2D(w)

        Initialize with each dimension specified::

            freud.index.Index2D(w, h)
    """
    cdef Index1D.Index2D *thisptr

    def __cinit__(self, w, h=None):
        if h is not None:
            self.thisptr = new Index1D.Index2D(w, h)
        else:
            self.thisptr = new Index1D.Index2D(w)

    def __dealloc__(self):
        del self.thisptr

    def __call__(self, i, j):
        """
        :param i: column index
        :param j: row index
        :type i: unsigned int
        :type j: unsigned int
        :return: :math:`1`-dimensional index in flat array
        :rtype: unsigned int
        """
        return self.thisptr.getIndex(i, j)

    def getNumElements(self):
        """
        :return: number of elements in the array
        :rtype: unsigned int
        """
        return self.thisptr.getNumElements()

cdef class Index3D:
    """Freud-style indexer for flat arrays.

    Freud utilizes "flat" arrays at the C++ level i.e. an :math:`n`-dimensional array with :math:`n_i` elements in each \
    index is represented as a :math:`1`-dimensional array with :math:`\\prod\\limits_i n_i` elements.

    .. note:: Freud indexes column-first i.e. Index3D(i, j, k) will return the :math:`1`-dimensional index of the \
    :math:`i^{th}` column, :math:`j^{th}` row, and the :math:`k^{th}` frame. This is the opposite of what occurs in a \
    numpy array, in which array[i, j, k] returns the element in the :math:`i^{th}` frame, :math:`j^{th}` row, and the \
    :math:`k^{th}` column.

    .. moduleauthor:: Joshua Anderson <joaander@umich.edu>

    :param w: width of 2D array (number of columns)
    :param h: height of 2D array (number of rows)
    :param d: depth of 2D array (number of frames)
    :type w: unsigned int
    :type h: unsigned int
    :type d: unsigned int

    - Constructor Calls:

        Initialize with all dimensions identical::

            freud.index.Index3D(w)

        Initialize with each dimension specified::

            freud.index.Index3D(w, h, d)
    """
    cdef Index1D.Index3D *thisptr

    def __cinit__(self, w, h=None, d=None):
        if h is not None:
            self.thisptr = new Index1D.Index3D(w, h, d)
        else:
            self.thisptr = new Index1D.Index3D(w)

    def __dealloc__(self):
        del self.thisptr

    def __call__(self, i, j, k):
        """
        :param i: column index
        :param j: row index
        :param k: frame index
        :type i: unsigned int
        :type j: unsigned int
        :type k: unsigned int
        :return: :math:`1`-dimensional index in flat array
        :rtype: unsigned int
        """
        return self.thisptr.getIndex(i, j, k)

    def getNumElements(self):
        """
        :return: number of elements in the array
        :rtype: unsigned int
        """
        return self.thisptr.getNumElements()

