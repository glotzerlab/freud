==============
Index Module
==============

The index module exposes the :math:`1`-dimensional indexer utilized in freud at the C++ level.

Freud utilizes "flat" arrays at the C++ level i.e. an :math:`n`-dimensional array with :math:`n_i` elements in each index is represented as a :math:`1`-dimensional array with :math:`\prod\limits_i n_i` elements.


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
