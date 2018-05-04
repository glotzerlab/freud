===========
PMFT Module
===========

The PMFT Module allows for the calculation of the Potential of Mean Force and Torque (PMFT) [Cit2]_ in a number of \
different coordinate systems.

.. note::
    The coordinate system in which the calculation is performed is not the same as the coordinate system in which \
    particle positions and orientations should be supplied. Only certain coordinate systems are available for certain \
    particle positions and orientations:

    * 2D particle coordinates (position: [x, y, 0], orientation: :math:`\theta`):
        * :math:`X`, :math:`Y`
        * :math:`X`, :math:`Y`, :math:`\theta_2`
        * :math:`R`, :math:`\theta_1`, :math:`\theta_2`

    * 3D particle coordinates: :math:`X`, :math:`Y`, :math:`Z`

Coordinate System: :math:`x`, :math:`y`, :math:`\theta_2`
=========================================================

.. autoclass:: freud.pmft.PMFTXYT(x_max, y_max, n_x, n_y, n_t)
    :members:
    :inherited-members:

Coordinate System: :math:`x`, :math:`y`
=======================================

.. autoclass:: freud.pmft.PMFTXY2D(x_max, y_max, n_x, n_y)
    :members:
    :inherited-members:

Coordinate System: :math:`r`, :math:`\theta_1`, :math:`\theta_2`
================================================================

.. autoclass:: freud.pmft.PMFTR12(r_max, n_r, n_t1, n_t2)
    :members:
    :inherited-members:

Coordinate System: :math:`x`, :math:`y`, :math:`z`
==================================================

.. autoclass:: freud.pmft.PMFTXYZ(x_max, y_max, z_max, n_x, n_y, n_z)
    :members:
    :inherited-members:
