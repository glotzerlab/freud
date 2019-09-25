===============
Locality Module
===============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.locality.NeighborList
    freud.locality.NeighborQuery
    freud.locality.NeighborQueryResult
    freud.locality.AABBQuery
    freud.locality.LinkCell
    freud.locality.IteratorLinkCell

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

.. autoclass:: freud.locality.LinkCell(box, cell_width, points=None)
   :members:

.. autoclass:: freud.locality.IteratorLinkCell()
   :members:
