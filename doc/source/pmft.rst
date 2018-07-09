===========
PMFT Module
===========

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.pmft.PMFTR12
    freud.pmft.PMFTXY2D
    freud.pmft.PMFTXYT
    freud.pmft.PMFTXYZ

.. rubric:: Details

.. automodule:: freud.pmft
    :synopsis: Compute potentials of mean force and torque

Coordinate System: :math:`x`, :math:`y`, :math:`\theta_1`
---------------------------------------------------------

.. autoclass:: freud.pmft.PMFTXYT(x_max, y_max, n_x, n_y, n_t)
    :members:
    :inherited-members:

Coordinate System: :math:`x`, :math:`y`
---------------------------------------

.. autoclass:: freud.pmft.PMFTXY2D(x_max, y_max, n_x, n_y)
    :members:
    :inherited-members:

Coordinate System: :math:`r`, :math:`\theta_1`, :math:`\theta_2`
----------------------------------------------------------------

.. autoclass:: freud.pmft.PMFTR12(r_max, n_r, n_t1, n_t2)
    :members:
    :inherited-members:

Coordinate System: :math:`x`, :math:`y`, :math:`z`
--------------------------------------------------

.. autoclass:: freud.pmft.PMFTXYZ(x_max, y_max, z_max, n_x, n_y, n_z)
    :members:
    :inherited-members:
