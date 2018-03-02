==============
Density Module
==============

The density module contains functions which deal with the density of the system


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
