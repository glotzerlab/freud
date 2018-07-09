===========
Bond Module
===========

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.bond.BondingAnalysis
    freud.bond.BondingR12
    freud.bond.BondingXY2D
    freud.bond.BondingXYT
    freud.bond.BondingXYZ

.. rubric:: Details

.. automodule:: freud.bond
    :synopsis: Calculate bonds

.. autoclass:: freud.bond.BondingAnalysis(num_particles, num_bonds)
    :members:
    :inherited-members:

Bond Computation
================

The below classes all perform computation of bonds in the system.
Bonds are computed for all particles to determine which other particles are in which bonding sites at any time.
The different classes are specialized for particular coordinate systems in both 2D and 3D

Coordinate System: :math:`r`, :math:`\theta_1`, :math:`\theta_2`
----------------------------------------------------------------

.. autoclass:: freud.bond.BondingR12(r_max, bond_map, bond_list)
    :members:
    :inherited-members:

Coordinate System: :math:`x`, :math:`y`
---------------------------------------

.. autoclass:: freud.bond.BondingXY2D(x_max, y_max, bond_map, bond_list)
    :members:
    :inherited-members:

Coordinate System: :math:`r`, :math:`y`, :math:`\theta_1`
---------------------------------------------------------

.. autoclass:: freud.bond.BondingXYT(x_max, y_max, bond_map, bond_list)
    :members:
    :inherited-members:

Coordinate System: :math:`r`, :math:`y`, :math:`z`
---------------------------------------------------------------

.. autoclass:: freud.bond.BondingXYZ(x_max, y_max, z_max, bond_map, bond_list)
    :members:
    :inherited-members:
