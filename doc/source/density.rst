==============
Density Module
==============

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.density.GaussianDensity
    freud.density.LocalDensity
    freud.density.RDF
    freud.density.ComplexCF
    freud.density.FloatCF

.. rubric:: Details

.. automodule:: freud.density
    :synopsis: Analyze system density.

Correlation Functions
=====================

.. autoclass:: freud.density.FloatCF(rmax, dr)
    :members:

.. autoclass:: freud.density.ComplexCF(rmax, dr)
    :members:

Gaussian Density
================

.. autoclass:: freud.density.GaussianDensity(*args)
    :members:

Local Density
=============

.. autoclass:: freud.density.LocalDensity(r_cut, volume, diameter)
    :members:

Radial Distribution Function
============================

.. autoclass:: freud.density.RDF(rmax, dr, rmin=0)
    :members:
