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

Neighbor List
=============

.. autoclass:: freud.locality.NeighborList
   :members:

Cell Lists
==========

.. autoclass:: freud.locality.IteratorLinkCell()
   :members:

.. autoclass:: freud.locality.LinkCell(box, cell_width)
   :members: compute, getCell, getCellNeighbors, itercell

Nearest Neighbors
=================

.. autoclass:: freud.locality.NearestNeighbors(rmax, n_neigh, scale=1.1, strict_cut=False)
   :members: compute, getNeighborList, getNeighbors, getRsq, setCutMode
