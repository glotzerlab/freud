==============
Density Module
==============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.density.FloatCF
    freud.density.ComplexCF
    freud.density.GaussianDensity
    freud.density.LocalDensity
    freud.density.RDF

.. rubric:: Details

.. automodule:: freud.density
    :synopsis: Analyze system density.

Correlation Functions
=====================

.. autoclass:: freud.density.FloatCF(rmax, dr)
    :members: accumulate, compute, reset, plot

.. autoclass:: freud.density.ComplexCF(rmax, dr)
    :members: accumulate, compute, reset, plot

Gaussian Density
================

.. autoclass:: freud.density.GaussianDensity(\*args)
    :members: compute, plot

Local Density
=============

.. autoclass:: freud.density.LocalDensity(r_cut, volume, diameter)
    :members: compute

Radial Distribution Function
============================

.. autoclass:: freud.density.RDF(rmax, dr, rmin=0)
    :members: accumulate, compute, reset, plot
