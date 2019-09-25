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
    freud.environment.LocalBondProjection

.. rubric:: Details

.. automodule:: freud.environment
    :synopsis: Analyze local particle environments.

Bond Order
==========

.. autoclass:: freud.environment.BondOrder(r_max, num_neighbors, n_bins_theta, n_bins_phi)
    :members: accumulate, compute, reset

Local Descriptors
=================

.. autoclass:: freud.environment.LocalDescriptors(num_neighbors, l_max, r_max, negative_m=True)
    :members: compute

Match Environments
==================

.. autoclass:: freud.environment.MatchEnv(box, r_max, num_neighbors)
    :members: cluster, getEnvironment, isSimilar, matchMotif, minRMSDMotif, minimizeRMSD, plot

Angular Separation
==================

.. autoclass:: freud.environment.AngularSeparation(r_max, num_neighbors)
    :members: computeGlobal, computeNeighbor

Local Bond Projection
=====================

.. autoclass:: freud.environment.LocalBondProjection(r_max, num_neighbors)
    :members: compute
