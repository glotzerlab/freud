.. contents:: Freud density

==============
Density Module
==============

The density module contains functions which deal with the density of the system


Correlation Functions
=====================

.. autoclass:: freud.density.FloatCF
    :members:

    .. method:: __init__(self, rmax, dr)

.. autoclass:: freud.density.ComplexCF
    :members:

    .. method:: __init__(self, rmax, dr)

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

.. autoclass:: freud.density.LocalDensity
    :members:

    .. method:: __init__(self, r_cut, volume, diameter)

Radial Distribution Function
============================

.. autoclass:: freud.density.RDF
    :members:

    .. method:: __init__(self, rmax, dr)
