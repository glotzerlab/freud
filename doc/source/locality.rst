===============
Locality Module
===============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.locality.NeighborQuery
    freud.locality.NeighborQueryResult
    freud.locality.NeighborList
    freud.locality.IteratorLinkCell
    freud.locality.LinkCell
    freud.locality.NearestNeighbors
    freud.locality.AABBQuery

.. rubric:: Details

.. automodule:: freud.locality
    :synopsis: Data structures for finding neighboring points.

Neighbor List
=============

.. autoclass:: freud.locality.NeighborList
   :members:

Neighbor Querying
=================

.. autoclass:: freud.locality.NeighborQuery
   :members:

.. autoclass:: freud.locality.NeighborQueryResult
   :members:

.. autoclass:: freud.locality.AABBQuery(box, points)
   :members:
   :inherited-members: queryBall

.. autoclass:: freud.locality.LinkCell(box, cell_width, points=None)
   :members: compute, getCell, getCellNeighbors, itercell, query, queryBall

.. autoclass:: freud.locality.IteratorLinkCell()
   :members:

Nearest Neighbors
=================

.. autoclass:: freud.locality.NearestNeighbors(r_max, num_neighbors, scale=1.1, strict_cut=False)
   :members: compute, getNeighborList, getNeighbors, getRsq
