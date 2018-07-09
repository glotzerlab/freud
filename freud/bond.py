# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

R"""
The bond module allows for the computation of bonds as defined by a map.
Depending on the coordinate system desired, either a two or three dimensional array is supplied, with each element containing the bond index mapped to the pair geometry of that element.
The user provides a list of indices to track, so that not all bond indices contained in the bond map need to be tracked in computation.

The bond module is designed to take in arrays using the same coordinate systems in the :doc:`pmft` in freud.

.. note::
    The coordinate system in which the calculation is performed is not the same as the coordinate system in which particle positions and orientations should be supplied.
    Only certain coordinate systems are available for certain particle positions and orientations:

    * 2D particle coordinates (position: [:math:`x`, :math:`y`, :math:`0`], orientation: :math:`\theta`):
        * :math:`X`, :math:`Y`
        * :math:`X`, :math:`Y`, :math:`\theta_2`
        * :math:`r`, :math:`\theta_1`, :math:`\theta_2`

    * 3D particle coordinates:
        * :math:`X`, :math:`Y`, :math:`Z`
"""

from ._freud import BondingAnalysis
from ._freud import BondingR12
from ._freud import BondingXY2D
from ._freud import BondingXYT
from ._freud import BondingXYZ
