===========
PMFT Module
===========

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    freud.pmft.PMFTR12
    freud.pmft.PMFTXYT
    freud.pmft.PMFTXY2D
    freud.pmft.PMFTXYZ

.. rubric:: Details

.. automodule:: freud.pmft
    :synopsis: Compute potentials of mean force and torque.

PMFT :math:`\left(r, \theta_1, \theta_2\right)`
===============================================

.. autoclass:: freud.pmft.PMFTR12(r_max, n_r, n_t1, n_t2)
    :members: accumulate, compute, reset

PMFT :math:`\left(x, y\right)`
==============================

.. autoclass:: freud.pmft.PMFTXY2D(x_max, y_max, n_x, n_y)
    :members: accumulate, compute, reset

PMFT :math:`\left(x, y, \theta\right)`
======================================

.. autoclass:: freud.pmft.PMFTXYT(x_max, y_max, n_x, n_y, n_t)
    :members: accumulate, compute, reset

PMFT :math:`\left(x, y, z\right)`
=================================

.. autoclass:: freud.pmft.PMFTXYZ(x_max, y_max, z_max, n_x, n_y, n_z)
    :members: accumulate, compute, reset
