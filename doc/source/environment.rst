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
    freud.environment.AngularSeparation

.. rubric:: Details

.. automodule:: freud.environment
    :synopsis: Analyze local particle environments

Bond Order
==========

.. autoclass:: freud.environment.BondOrder(rmax, k, n, nBinsT, nBinsP)
    :members: accumulate, compute, reset

Local Descriptors
=================

.. autoclass:: freud.environment.LocalDescriptors(num_neighbors, lmax, rmax, negative_m=True)
    :members: compute, computeNList

Match Environments
==================

.. autoclass:: freud.environment.MatchEnv(box, rmax, k)
    :members: cluster, getEnvironment, isSimilar, matchMotif, minRMSDMotif, minimizeRMSD, setBox

Angular Separation
==================

.. autoclass:: freud.environment.AngularSeparation(box, rmax, n)
    :members: computeGlobal, computeNeighbor
