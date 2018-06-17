============
Index Module
============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.index.Index2D
    freud.index.Index3D

.. rubric:: Details

The index module exposes the :math:`1`-dimensional indexer utilized in freud at the C++ level.
At the C++ level, freud utilizes "flat" arrays, i.e. an :math:`n`-dimensional array with :math:`n_i` elements in each index is represented as a :math:`1`-dimensional array with :math:`\prod\limits_i n_i` elements.

Index2D
=======

.. autoclass:: freud.index.Index2D(*args)
    :members:

    .. automethod:: freud.index.Index2D.__call__(self, i, j)

Index3D
=======

.. autoclass:: freud.index.Index3D(*args)
    :members:

    .. automethod:: freud.index.Index3D.__call__(self, i, j, k)
