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

The density module contains various classes relating to the density of the system.
These functions allow evaluation of particle distributions with respect to other particles. 

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
