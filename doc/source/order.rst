============
Order Module
============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.order.CubaticOrderParameter
    freud.order.NematicOrderParameter
    freud.order.HexOrderParameter
    freud.order.TransOrderParameter
    freud.order.LocalQl
    freud.order.LocalQlNear
    freud.order.LocalWl
    freud.order.LocalWlNear
    freud.order.SolLiq
    freud.order.SolLiqNear
    freud.order.BondOrder
    freud.order.LocalDescriptors
    freud.order.MatchEnv
    freud.order.Pairing2D
    freud.order.AngularSeparation

.. rubric:: Details

The order module contains functions which compute order parameters for the whole system or individual particles.
Order parameters take bond order data and interpret it in some way to quantify the degree of order in a system using a scalar value.
This is often done through computing spherical harmonics of the bond order diagram, which are the spherical analogue of Fourier Transforms.

.. autoclass:: freud.order.CubaticOrderParameter(t_initial, t_final, scale, n_replicates, seed)
    :members:

.. autoclass:: freud.order.NematicOrderParameter(u)
    :members:

.. autoclass:: freud.order.HexOrderParameter(rmax, k, n)
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

Deprecated Classes
==================

The below functions have all either been deprecated or moved to the :doc:`environment` module

Environment Matching
--------------------

.. autoclass:: freud.order.MatchEnv(box, rmax, k)
    :members:

Pairing
-------

.. note::
    This module is deprecated and is replaced with :doc:`bond`.

.. autoclass:: freud.order.Pairing2D(rmax, k, compDotTol)
    :members:

Bond Order
----------

.. autoclass:: freud.order.BondOrder(rmax, k, n, nBinsT, nBinsP)
    :members:

Local Descriptors
-----------------

.. autoclass:: freud.order.LocalDescriptors(box, nNeigh, lmax, rmax)
    :members:

Angular Separation
-----------------

.. autoclass:: freud.order.AngularSeparation(box, rmax, n)
    :members:

