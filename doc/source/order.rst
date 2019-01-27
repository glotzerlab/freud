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

.. rubric:: Details

.. automodule:: freud.order
    :synopsis: Compute order parameters

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

.. autoclass:: freud.order.HexOrderParameter(rmax, k, n)
    :members: compute

Translational Order Parameter
=============================

.. autoclass:: freud.order.TransOrderParameter(rmax, k, n)
    :members: compute

Steinhardt :math:`Q_l` Order Parameter
======================================

.. autoclass:: freud.order.LocalQl(box, rmax, l, rmin)
    :members: compute, computeAve, computeAveNorm, computeNorm, setBox

.. autoclass:: freud.order.LocalQlNear(box, rmax, l, kn)
    :members: compute, computeAve, computeAveNorm, computeNorm, setBox

Steinhardt :math:`W_l` Order Parameter
======================================

.. autoclass:: freud.order.LocalWl(box, rmax, l)
    :members: compute, computeAve, computeAveNorm, computeNorm, setBox

.. autoclass:: freud.order.LocalWlNear(box, rmax, l, kn)
    :members: compute, computeAve, computeAveNorm, computeNorm, setBox

Solid-Liquid Order Parameter
============================

.. autoclass:: freud.order.SolLiq(box, rmax, Qthreshold, Sthreshold, l)
    :members: compute, computeSolLiqNoNorm, computeSolLiqVariant

.. autoclass:: freud.order.SolLiqNear(box, rmax, Qthreshold, Sthreshold, l, kn)
    :members: compute, computeSolLiqNoNorm, computeSolLiqVariant
