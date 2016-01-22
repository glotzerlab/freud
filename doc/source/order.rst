.. contents:: Freud order

==============
Order Module
==============

The order module contains functions which deal with the order of the system


Bond Order
==========

.. autoclass:: freud.order.BondOrder(rmax, k, n, nBinsT, nBinsP)
    :members:

Entropic Bonding
================

.. autoclass:: freud.order.EntropicBonding(xmax, ymax, nNeighbors, bondMap)
    :members:

.. autoclass:: freud.order.EntropicBondingRT(rmax, nNeighbors, bondMap)
    :members:

Order Parameters
================

Lorem Ipsum

.. autoclass:: freud.order.HexOrderParameter(rmax, k, n)
    :members:

.. autoclass:: freud.order.LocalDescriptors(box, nNeigh, lmax, rmax)
    :members:

.. autoclass:: freud.order.TransOrderParameter(rmax, k, n)
    :members:

.. autoclass:: freud.order.LocalQl(box, rmax, l, rmin)
    :members:

.. autoclass:: freud.order.LocalQlNear(box, rmax, l, kn)
    :members:

.. autoclass:: freud.order.LocalWl(box, rmax, l)
    :members:
