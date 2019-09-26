==============
Density Module
==============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.density.CorrelationFunction
    freud.density.GaussianDensity
    freud.density.LocalDensity
    freud.density.RDF

.. rubric:: Details

.. automodule:: freud.density
    :synopsis: Analyze system density.

Correlation Functions
=====================

.. autoclass:: freud.density.CorrelationFunction(bins, r_max)
    :members: accumulate, compute, reset, plot

Gaussian Density
================

.. autoclass:: freud.density.GaussianDensity(width, r_max, sigma)
    :members: compute, plot

Local Density
=============

.. autoclass:: freud.density.LocalDensity(r_max, volume, diameter)
    :members: compute

Radial Distribution Function
============================

.. autoclass:: freud.density.RDF(r_max, dr, r_min=0)
    :members: accumulate, compute, reset, plot
