.. contents:: Freud density

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

.. autoclass:: freud.density.GaussianDensity
    :members:

    .. method:: __init__(self, width, r_cut, dr)

    Initialize with all dimensions identical

    .. method:: __init__(self, width_x, width_y, width_z, r_cut, dr)

    Initialize with specific dimensions

Local Density
=============

.. autoclass:: freud.density.LocalDensity(r_cut, volume, diameter)
    :members:

Radial Distribution Function
============================

.. autoclass:: freud.density.RDF(rmax, dr)
    :members:
