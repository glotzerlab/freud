.. _environment:

=================
Environment Module
=================

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.environment.BondOrder
    freud.environment.LocalDescriptors
    freud.environment.MatchEnv
    freud.environment.Pairing2D
    freud.environment.AngularSeparation

.. rubric:: Details

The environment module contains functions which characterize the local environments of particles in the system.
These methods use the positions and orientations of particles in the local neighborhood of a given particle to characterize the particle environment.

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

