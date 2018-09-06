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

.. autoclass:: freud.environment.BondOrder(rmax, k, n, nBinsT, nBinsP)
    :members: accumulate, compute, reset

.. autoclass:: freud.environment.LocalDescriptors(num_neighbors, lmax, rmax, negative_m=True)
    :members: compute, computeNList

.. autoclass:: freud.environment.MatchEnv(box, rmax, k)
    :members: cluster, getEnvironment, isSimilar, matchMotif, minRMSDMotif, minimizeRMSD, setBox

.. note::
    Pairing2D is deprecated and is replaced with :doc:`bond`.

.. autoclass:: freud.environment.Pairing2D(rmax, k, compDotTol)
    :members: compute

.. autoclass:: freud.environment.AngularSeparation(box, rmax, n)
    :members: computeGlobal, computeNeighbor
