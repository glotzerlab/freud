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

.. autoclass:: freud.order.CubaticOrderParameter(t_initial, t_final, scale, n_replicates, seed)
    :members:

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

.. autoclass:: freud.order.LocalWlNear(box, rmax, l, kn)
    :members:

.. autoclass:: freud.order.SolLiq(box, rmax, Qthreshold, Sthreshold, l)
    :members:

.. autoclass:: freud.order.SolLiqNear(box, rmax, Qthreshold, Sthreshold, l)
    :members:

.. autoclass:: freud.order.MatchEnv(box, rmax, k)
    :members:

.. autoclass:: freud.order.Pairing2D(rmax, k, compDotTol)
    :members:
