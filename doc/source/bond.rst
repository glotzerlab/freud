.. contents:: Freud bond

===========
Bond Module
===========

The bond modules allows for the computation of bonds as defined by a map. Depending on the coordinate system desired, \
either a two or three dimensional array is supplied, with each element containing the bond index which corresponds to \
the pair geometry of that element. The user also provides a list of indices to track, so that not all bond indices \
contained in the bond map need to be tracked in computation.

The bonding module is designed to take in arrays using the same coordinate systems in the :doc:`pmft` in freud.

.. note::
    the coordinate system in which the calculation is performed is not the same as the coordinate system in which \
    particle positions and orientations should be supplied; only certain coordinate systems are available for certain \
    particle positions and orientations:

    * 2D particle coordinates (position: [x, y, 0], orientation: :math:`\theta`):
        * X, Y
        * X, Y, :math:`\theta_2`
        * :math:`r`, :math:`\theta_1`, :math:`\theta_2`

    * 3D particle coordinates -> X, Y, Z

Bonding Analysis
================

.. autoclass:: freud.bond.BondingAnalysis(num_particles, num_bonds)
    :members:

Coordinate System: :math:`x`, :math:`y`
=======================================

.. autoclass:: freud.bond.BondingXY2D(x_max, y_max, bond_map, bond_list)
    :members:

Coordinate System: :math:`x`, :math:`y`, :math:`\theta_2`
=========================================================

.. autoclass:: freud.bond.BondingXYT(x_max, y_max, bond_map, bond_list)
    :members:

Coordinate System: :math:`r`, :math:`\theta_1`, :math:`\theta_2`
================================================================

.. autoclass:: freud.bond.BondingR12(r_max, bond_map, bond_list)
    :members:

Coordinate System: :math:`x`, :math:`y`, :math:`z`
==================================================

.. autoclass:: freud.bond.BondingXYZ(x_max, y_max, z_max, bond_map, bond_list)
    :members:
