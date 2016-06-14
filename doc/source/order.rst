.. contents:: Freud order

==============
Order Module
==============

The order module contains functions which deal with the order of the system


Bond Order
==========

.. autoclass:: freud.order.BondOrder(rmax, k, n, nBinsT, nBinsP)
    :members:

Order Parameters
================

Lorem Ipsum

Cubatic Order Parameter
=======================

.. autoclass:: freud.order.CubaticOrderParameter(t_initial, t_final, scale, n_replicates, seed)
    :members:

Hexatic Order Parameter
=======================

.. autoclass:: freud.order.HexOrderParameter(rmax, k, n)
    :members:

Local Descriptors
=================

.. autoclass:: freud.order.LocalDescriptors(box, nNeigh, lmax, rmax)
    :members:

Translational Order Parameter
=============================

.. autoclass:: freud.order.TransOrderParameter(rmax, k, n)
    :members:

Local :math:`Q_l`
=================

.. autoclass:: freud.order.LocalQl(box, rmax, l, rmin)
    :members:

Nearest Neighbors Local :math:`Q_l`
===================================

.. autoclass:: freud.order.LocalQlNear(box, rmax, l, kn)
    :members:

Local :math:`W_l`
=================

.. autoclass:: freud.order.LocalWl(box, rmax, l)
    :members:

Nearest Neighbors Local :math:`W_l`
===================================

.. autoclass:: freud.order.LocalWlNear(box, rmax, l, kn)
    :members:

Solid-Liquid Order Parameter
============================

.. autoclass:: freud.order.SolLiq(box, rmax, Qthreshold, Sthreshold, l)
    :members:

Nearest Neighbors Solid-Liquid Order Parameter
==============================================

.. autoclass:: freud.order.SolLiqNear(box, rmax, Qthreshold, Sthreshold, l)
    :members:

Environment Matching
====================

.. autoclass:: freud.order.MatchEnv(box, rmax, k)
    :members:

Pairing
=======

.. note::
    This module is deprecated is is replaced with :doc:`bond`

.. autoclass:: freud.order.Pairing2D(rmax, k, compDotTol)
    :members:
