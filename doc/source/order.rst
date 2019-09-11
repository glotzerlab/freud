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
    freud.order.Steinhardt
    freud.order.SolLiq
    freud.order.SolLiqNear
    freud.order.RotationalAutocorrelation

.. rubric:: Details

.. automodule:: freud.order
    :synopsis: Compute order parameters.

Cubatic Order Parameter
=======================

.. autoclass:: freud.order.CubaticOrderParameter(t_initial, t_final, scale, n_replicates, seed)
    :members: compute

Nematic Order Parameter
=======================

.. autoclass:: freud.order.NematicOrderParameter(u)
    :members: compute

Hexatic Order Parameter
=======================

.. autoclass:: freud.order.HexOrderParameter(r_max, k=6, num_neighbors=0)
    :members: compute

Translational Order Parameter
=============================

.. autoclass:: freud.order.TransOrderParameter(r_max, k=6.0, num_neighbors=0)
    :members: compute

Steinhardt Order Parameters
===========================

.. autoclass:: freud.order.Steinhardt(l, r_min, r_max, average=False, norm=False, Wl=False, num_neighbors=0)
    :members: compute, plot

Solid-Liquid Order Parameter
============================

.. autoclass:: freud.order.SolLiq(box, r_max, Qthreshold, Sthreshold, l)
    :members: compute, computeSolLiqNoNorm, computeSolLiqVariant

.. autoclass:: freud.order.SolLiqNear(box, r_max, Qthreshold, Sthreshold, l, num_neighbors=12)
    :members: compute, computeSolLiqNoNorm, computeSolLiqVariant

Rotational Autocorrelation
==========================

.. autoclass:: freud.order.RotationalAutocorrelation(l)
    :members: compute
