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
    :members: compute, initialize

.. autoclass:: freud.bond.BondingR12(r_max, bond_map, bond_list)
    :members: compute

.. autoclass:: freud.bond.BondingXY2D(x_max, y_max, bond_map, bond_list)
    :members: compute

.. autoclass:: freud.bond.BondingXYT(x_max, y_max, bond_map, bond_list)
    :members: compute

.. autoclass:: freud.bond.BondingXYZ(x_max, y_max, z_max, bond_map, bond_list)
    :members: compute
