# Copyright (c) 2010-2018 The Regents of the University of Michigan
# This file is part of the freud project, released under the BSD 3-Clause License.

R"""
The PMFT Module allows for the calculation of the Potential of Mean Force and Torque (PMFT) [vanAndersKlotsa2014]_ [vanAndersAhmed2014]_ in a number of different coordinate systems.
The PMFT is defined as the negative algorithm of positional correlation function (PCF).
A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
The resulting values are accumulated in a PCF array listing the value of the PCF at a discrete set of points.
The specific points are determined by the particular coordinate system used to represent the system.

.. note::
    The coordinate system in which the calculation is performed is not the same as the coordinate system in which particle positions and orientations should be supplied. Only certain coordinate systems are available for certain particle positions and orientations:

    * 2D particle coordinates (position: [x, y, 0], orientation: :math:`\theta`):
        * :math:`r`, :math:`\theta_1`, :math:`\theta_2`.
        * :math:`x`, :math:`y`.
        * :math:`x`, :math:`y`, :math:`\theta`.

    * 3D particle coordinates:
        * :math:`x`, :math:`y`, :math:`z`.
"""

from ._freud import PMFTR12
from ._freud import PMFTXYT
from ._freud import PMFTXY2D
from ._freud import PMFTXYZ
