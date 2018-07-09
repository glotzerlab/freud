.. _environment:

==================
Environment Module
==================

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.environment.BondOrder
    freud.environment.LocalDescriptors
    freud.environment.MatchEnv
    freud.environment.Pairing2D
    freud.environment.AngularSeparation

.. rubric:: Details

.. automodule:: freud.environment
    :synopsis: Analyze local particle environments

.. autoclass:: freud.environment.MatchEnv(box, rmax, k)
    :members:

.. note::
    This module is deprecated and is replaced with :doc:`bond`.

.. autoclass:: freud.environment.Pairing2D(rmax, k, compDotTol)
    :members:

.. autoclass:: freud.environment.BondOrder(rmax, k, n, nBinsT, nBinsP)
    :members:

.. autoclass:: freud.environment.LocalDescriptors(box, nNeigh, lmax, rmax)
    :members:

.. autoclass:: freud.environment.AngularSeparation(box, rmax, n)
    :members:

