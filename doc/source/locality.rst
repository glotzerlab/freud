===============
Locality Module
===============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.locality.NeighborList
    freud.locality.IteratorLinkCell
    freud.locality.LinkCell
    freud.locality.NearestNeighbors

.. rubric:: Details

.. automodule:: freud.locality
    :synopsis: Data structures for finding neighboring points.

.. autoclass:: freud.locality.NeighborList
   :members:

.. autoclass:: freud.locality.IteratorLinkCell()
   :members:

.. autoclass:: freud.locality.LinkCell(box, cell_width)
   :members:

.. autoclass:: freud.locality.NearestNeighbors(rmax, n_neigh, scale=1.1, strict_cut=False)
   :members:
